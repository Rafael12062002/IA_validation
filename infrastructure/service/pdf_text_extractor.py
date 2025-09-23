import fitz

class PDFTextExtractor:
    def extrair_texto(self, pdf_path: str) -> str:
        texto = []
        with fitz.open(pdf_path) as doc:
            for pagina in doc:
                texto.append(pagina.get_text())
        return "\n".join(texto)