import shutil
import cv2
import numpy as np
import io
import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import zipfile
import base64
from application.selecionar_melhor_rosto_usecase import SelecionarMelhorRostoUseCase
from application.converter_pdf_usecase import ConverterPDFUseCase
from application.comparar_documento_usecase import CompararDocumentoUseCase
from infrastructure.adapters.documento_repository_local import DocumentoRepositoryLocal
from infrastructure.service.pdf_text_extractor import PDFTextExtractor
from application.extrair_texto_usecase import ExtrairTextoUseCase
from infrastructure.adapters.face_detector_mediapipe import FaceDetectorMediapipe
from domain.services.alinhamento_service import AlinhamentoService

router = APIRouter()

converter_pdf_usecase = ConverterPDFUseCase()
repo = DocumentoRepositoryLocal()
usecase = CompararDocumentoUseCase(repo)
extractor = PDFTextExtractor()
extrair_texto = ExtrairTextoUseCase(repo, extractor)
detector = FaceDetectorMediapipe(min_detection_confidence=0.5)
alinhador = AlinhamentoService()
use_case = SelecionarMelhorRostoUseCase(detector, alinhador, step_degrees=1)

@router.post("/extrair-melhor-rosto")
async def extrair_melhor_rosto(file: UploadFile = File(...), step_degrees: int = 1):
    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Imagem inválida")

    use_case = SelecionarMelhorRostoUseCase(detector, alinhador, step_degrees=step_degrees)
    melhor_img, todos_os_rostos = use_case.executar(img)

    if melhor_img is None:
        raise HTTPException(status_code=404, detail="Nenhum rosto encontrado")

    # Criar buffer de memória para o ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        # Adiciona a melhor imagem
        ok, encoded = cv2.imencode(".jpg", melhor_img)
        if not ok:
            raise HTTPException(status_code=500, detail="Falha ao codificar melhor imagem")
        zip_file.writestr("melhor_rosto.jpg", encoded.tobytes())

        # Adiciona todas as imagens detectadas
        for idx, r in enumerate(todos_os_rostos):
            ok, encoded = cv2.imencode(".jpg", r["face"])
            if ok:
                zip_file.writestr(f"rosto_{idx}.jpg", encoded.tobytes())

    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": "attachment; filename=rostos.zip"}
    )


@router.post("/converter-pdf")
async def converter_pdf(file: UploadFile = File(...)):
    conteudo = await file.read()
    caminho_temp = f"temp/{file.filename}"
    with open(caminho_temp, "wb") as f:
        f.write(conteudo)

    imagens = converter_pdf_usecase.executar(caminho_temp)
    return {"Imagens_geradas": imagens}

@router.post("/comparar/{documento_id}")
async def comparar(documento_id: str, file: UploadFile = File(...)):
    temp_pdf = f"temp_{file.filename}"
    with open(temp_pdf, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    resultado = use_case.executar(documento_id, temp_pdf)

    os.remove(temp_pdf)
    return {"documento_id": documento_id, "igual": resultado}

@router.get("/extrair-texto/{doc_id}")
def extrair_texto(doc_id: str):
    texto = use_case.executar(doc_id)
    return {"documento_id": doc_id, "texto": texto}
