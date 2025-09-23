import fitz  # pymupdf
import os

PDF_PATH = r"candidato3.pdf"
OUT_DIR = r"saida_pdf"
os.makedirs(OUT_DIR, exist_ok=True)

doc = fitz.open(PDF_PATH)
for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    pix = page.get_pixmap(dpi=200)   # ajustar dpi conforme necessidade
    out = os.path.join(OUT_DIR, f"pagina_{page_num+5}.jpg")
    pix.save(out)
    print("Salvo:", out)
doc.close()
