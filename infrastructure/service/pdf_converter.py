from pdf2image import convert_from_path
import tempfile
import os

class PDFConverter:
    def converter_para_imagens(self, pdf_path: str, pasta_saida="saida_pdf"):
        os.makedirs(pasta_saida, exist_ok=True)
        imagens = convert_from_path(pdf_path, dpi=300)

        arquivos = []
        for i, img in enumerate(imagens):
            out_path = os.path.join(pasta_saida, f"pagina_{i + 1}.jpj")
            img.save(out_path, "JPEG")
            arquivos.append(out_path)
        return arquivos
    
    def converter_temporario(self, pdf_path: str):
        temp_dir = tempfile.mkdtemp(prefix="pdf_temp_")
        imagens = convert_from_path(pdf_path, dpi=300)

        arquivos = []
        for i, img in enumerate(imagens):
            out_path = os.path.join(temp_dir, f"pagina_{i + 1}.jpg")
            img.save(out_path, "JPEG")
            arquivos.append(out_path)
        return arquivos