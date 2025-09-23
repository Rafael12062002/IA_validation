import cv2
import os
import numpy as np

INPUT = "saida_pdf/pagina_6.jpg"
OUT_DIR = "saida"
OUT_RG = os.path.join(OUT_DIR, "rg_completo.jpg")
OUT_FOTO = os.path.join(OUT_DIR, "foto_rg.jpg")

os.makedirs(OUT_DIR, exist_ok=True)

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    eq = cv2.equalizeHist(gray)

    blur = cv2.GaussianBlur(eq, (3,3), 0)
    return blur

def detectar_por_contornos(img, debug=False):
    proc = preprocess(img)
    # binarização adaptativa (mais robusta a iluminação)
    th = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 9)
    # dilatar para juntar componentes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dil = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    contornos, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []
    h_img, w_img = proc.shape

    for cnt in contornos:
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h
        # filtros: tamanho mínimo e proporção razoável (ajuste conforme seu RG)
        if area < 2000:          # evita pequenos ruídos
            continue
        proporcao = h / float(w) if w>0 else 0
        # esperamos foto retangular vertical; ajuste se seu RG for diferente
        if 1.0 < proporcao < 3.0 and w > w_img*0.08 and h > h_img*0.08:
            candidatos.append((x,y,w,h,area))

    # ordenar por área (maior primeiro) — preferir a maior candidata
    candidatos = sorted(candidatos, key=lambda x: x[4], reverse=True)
    if debug:
        print("Candidatos por contorno:", candidatos)
    if candidatos:
        x,y,w,h,_ = candidatos[0]
        return img[y:y+h, x:x+w], (x,y,w,h)
    return None, None

def detectar_por_face(img, cascade_path=None, debug=False):
    # tenta detectar face e retornar box expandida
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cascade_path is None:
        # caminho comum quando OpenCV foi instalado via pip
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        if debug: print("Cascade not found:", cascade_path)
        return None, None
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
    if debug: print("Faces encontradas:", faces)
    if len(faces) == 0:
        return None, None
    # pegar a maior face e expandir levemente para incluir enquadramento
    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    x,y,w,h = faces[0]
    pad_x = int(w * 0.35)
    pad_y = int(h * 0.6)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img.shape[1], x + w + pad_x)
    y2 = min(img.shape[0], y + h + pad_y)
    return img[y1:y2, x1:x2], (x1,y1,x2-x1,y2-y1)

def recorte_manual(img):
    # janela interativa para selecionar ROI (pressione ENTER ou SPACE para confirmar, ESC para cancelar)
    r = cv2.selectROI("Selecione a foto (ENTER para confirmar)", img, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Selecione a foto (ENTER para confirmar)")
    x,y,w,h = r
    if w==0 or h==0:
        return None, None
    return img[y:y+h, x:x+w], (x,y,w,h)

def salvar_imagem(path, img):
    cv2.imwrite(path, img)
    print("Salvo:", path)

def main():
    img = cv2.imread(INPUT)
    if img is None:
        print("Erro: não consegui abrir", INPUT)
        return

    # salva RG completo sempre
    salvar_imagem(OUT_RG, img)
    print("Foto do rg completo salva")

    # 1) tentativa por contorno
    foto, box = detectar_por_contornos(img, debug=True)
    if foto is not None:
        salvar_imagem(OUT_FOTO, foto)
        print("Foto detectada por contorno e salva em", OUT_FOTO)
        return

    # 2) tentativa por detecção de face (fallback)
    foto, box = detectar_por_face(img, debug=True)
    if foto is not None:
        salvar_imagem(OUT_FOTO, foto)
        print("Foto detectada por detecção de face e salva em", OUT_FOTO)
        return

    # 3) fallback manual
    print("Foto não encontrada automaticamente. Abra a seleção manual para recortar.")
    foto, box = recorte_manual(img)
    if foto is not None:
        salvar_imagem(OUT_FOTO, foto)
        print("Foto cortada manualmente e salva em", OUT_FOTO)
    else:
        print("Operação cancelada — foto não salva.")

if __name__ == "__main__":
    main()