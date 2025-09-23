import shutil
import os
from fastapi import FastAPI, UploadFile, File
from application.extrair_foto_usecase import ExtrairFotoUseCase
from application.converter_pdf_usecase import ConverterPDFUseCase
from application.comparar_documento_usecase import CompararDocumentoUseCase
from infrastructure.adapters.documento_repository_local import DocumentoRepositoryLocal
from infrastructure.service.pdf_text_extractor import PDFTextExtractor
from application.extrair_texto_usecase import ExtrairTextoUseCase

app = FastAPI()
use_case = ExtrairFotoUseCase()
converter_pdf_usecase = ConverterPDFUseCase()
repo = DocumentoRepositoryLocal()
usecase = CompararDocumentoUseCase(repo)
extractor = PDFTextExtractor()
extrair_texto = ExtrairTextoUseCase(repo, extractor)

@app.post("/extrair-rosto")
async def extrair_rosto(file: UploadFile = File(...)):
    os.makedirs("temp", exist_ok=True)
    caminho_temp = f"temp/{file.filename}"
    conteudo = await file.read()
    with open(caminho_temp, "wb") as f:
        f.write(conteudo)

    saida = use_case.executar(caminho_temp)
    os.remove(caminho_temp)
    return {"Rostos_extraidos": saida}

@app.post("/converter-pdf")
async def converter_pdf(file: UploadFile = File(...)):
    conteudo = await file.read()
    caminho_temp = f"temp/{file.filename}"
    with open(caminho_temp, "wb") as f:
        f.write(conteudo)

    imagens = converter_pdf_usecase.executar(caminho_temp)
    return {"Imagens_geradas": imagens}

@app.post("/comparar/{documento_id}")
async def comparar(documento_id: str, file: UploadFile = File(...)):
    temp_pdf = f"temp_{file.filename}"
    with open(temp_pdf, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    resultado = use_case.executar(documento_id, temp_pdf)

    os.remove(temp_pdf)
    return {"documento_id": documento_id, "igual": resultado}

@app.get("/extrair-texto/{doc_id}")
def extrair_texto(doc_id: str):
    texto = use_case.executar(doc_id)
    return {"documento_id": doc_id, "texto": texto}