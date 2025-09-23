from infrastructure.service.pdf_text_extractor import PDFTextExtractor

class ExtrairTextoUseCase:
    def __init__(self, repo, extractor: PDFTextExtractor):
        self.repo = repo
        self.extractor = extractor
        
    def executar(self, documento_id: str) -> str:
        documento = self.repo.obter_por_id(documento_id)
        return self.extractor.extrair_texto(documento.caminho)