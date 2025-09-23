import cv2
from infrastructure.service.pdf_converter import PDFConverter

class CompararDocumentoUseCase:
    def __init__(self, documento_repository):
        self.documento_repository = documento_repository
        self.converter = PDFConverter()

    def executar(self, documento_id: str, pdf_path: str) -> bool:
        documento_salvo = self.documento_repository.obter_por_id(documento_id)

        imagens_base = self.converter.converter_temporario(documento_salvo.caminho)
        imagens_recebido = self.converter.converter_temporario(pdf_path)

        if len(imagens_base) != len(imagens_recebido):
            return False
        
        # Comparar página a página
        for img_base, img_nova in zip(imagens_base, imagens_recebido):
            base = cv2.imread(img_base)
            nova = cv2.imread(img_nova)

            # Se tamanhos diferentes -> não é igual
            if base.shape != nova.shape:
                return False

            # Comparação pixel a pixel
            diferenca = cv2.absdiff(base, nova)
            if diferenca.sum() != 0:  # se houver diferença
                return False

        return True