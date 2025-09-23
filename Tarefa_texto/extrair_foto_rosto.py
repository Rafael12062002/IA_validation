import cv2
import mediapipe as mp
import os

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def salvar_rosto(img_path, pasta_saida = "rostos", margem=0.3):
    os.makedirs(pasta_saida, exist_ok=True)
    img = cv2.imread(img_path)
    if img is None:
        print("Erro ao carregar a imagem: ", img_path)
        return
    
    h, w, _= img.shape

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        igm_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        resultados = face_detection.process(igm_rgb)

        if not resultados.detections:
            print("Nenhum rosto encontrado")
            return
        
        for i, det in enumerate(resultados.detections):
            bboxC = det.location_data.relative_bounding_box
            x_min = int(bboxC.xmin * w)
            y_min = int(bboxC.ymin * h)
            largura = int(bboxC.width * w)
            altura = int(bboxC.height * h)

            # Garante que não saia dos limites
            x_min = max(0, x_min - int(largura * margem))
            y_min = max(0, y_min - int(altura * margem))
            x_max = min(w, x_min + largura + int(largura * margem * 2))
            y_max = min(h, y_min + altura + int(altura * margem * 2))

            # Recorta o rosto
            rosto = img[y_min:y_max, x_min:x_max]

            # Salva o rosto
            out_path = os.path.join(pasta_saida, f"rosto_{i+1}.jpg")
            cv2.imwrite(out_path, rosto)
            print(f"Rosto {i+1} salvo em: {out_path}")


# Exemplo de uso
salvar_rosto("../saida_pdf/pagina_1.jpg")