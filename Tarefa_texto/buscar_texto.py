import cv2
import numpy as np
import os
os.environ['TESSDATA_PREFIX'] = r"C:\Program Files\Tesseract-OCR\tessdata"
import pytesseract

# Se o tesseract não estiver no PATH do Windows, configure aqui:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Binariza levemente para achar coords
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bw < 255))
    if coords.size == 0:
        return image
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_for_ocr(img_bgr, target_width=1200):
    # 1) deskew
    img = deskew(img_bgr)

    # 2) converte p/ cinza e aplica CLAHE (melhora contraste)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # 3) reduzir ruído preservando bordas
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.medianBlur(gray, 3)

    # 4) aumentar resolução (melhora leitura de fontes pequenas)
    h, w = gray.shape
    if w < target_width:
        scale = target_width / w
        gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

    # 5) binarização adaptativa (bom para iluminação desigual)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, blockSize=31, C=15)

    # 6) operações morfológicas (remove pequenos ruídos)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    # opcional: invert (se fundo escuro)
    # opened = cv2.bitwise_not(opened)

    return opened

def ocr_with_confidence(img_bin, lang='por', psms=[6, 3, 4, 11]):
    """
    Tenta vários PSMs e retorna o texto com maior confiança média.
    """
    best_text = ""
    best_conf = -1
    best_psm = None

    for psm in psms:
        cfg = f'--oem 3 --psm {psm}'
        data = pytesseract.image_to_data(img_bin, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)
        # coletar confs válidos (inteiros)
        confs = [c for c in data['conf'] if c != -1]
        avg_conf = sum(confs)/len(confs) if confs else -1
        text = pytesseract.image_to_string(img_bin, lang=lang, config=cfg)
        # debug:
        # print(f"PSM {psm} -> avg conf = {avg_conf}, len text = {len(text)}")
        if avg_conf > best_conf:
            best_conf = avg_conf
            best_text = text
            best_psm = psm

    return {"psm": best_psm, "avg_conf": best_conf, "text": best_text}

if __name__ == "__main__":
    img_path = "saida_pdf/pagina_5.jpg"      # troque pelo seu arquivo
    img = cv2.imread(img_path)
    pre = preprocess_for_ocr(img)
    resultado = ocr_with_confidence(pre, lang='por', psms=[7,6,4,11])  # experimente ordens diferentes
    print("Melhor PSM:", resultado['psm'])
    print("Conf média estimada:", resultado['avg_conf'])
    print("Texto extraído:\n")
    print(resultado['text'])

    # Visualizar a imagem pré-processada (apenas para debug)
    cv2.imshow("Pré-processada", pre)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Falar com o Miguel, pegar exemplos reais da base de dados e carregar e atualizar a aplicação. Esses exemplos, considerar rg, cpf e compara os documentos e validar se são iguais