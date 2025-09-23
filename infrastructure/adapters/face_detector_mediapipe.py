import cv2
import mediapipe as mp
import os

class FaceDetector:

    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    def detectar(self, img_path, pasta_saida="rostos", margem=0.5):
        img = cv2.imread(img_path)
        if img is None:
            return[]
        
        os.makedirs(pasta_saida, exist_ok=True)
        resultados = []

        angulos = [0, 90, 180, 270]
        detections = None
        img_final = img
        altura, largura, _= img.shape

        with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
            for ang in angulos:
                if ang == 90:
                    img_rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                elif ang == 180:
                    img_rot = cv2.rotate(img, cv2.ROTATE_180)
                elif ang == 270:
                    img_rot = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                else:
                    img_rot = img

                img_rgb = cv2.cvtColor(img_rot, cv2.COLOR_BGR2RGB)
                detections = detector.process(img_rgb)

                if detections.detections:
                    img_final = img_rot
                    break

            if detections and detections.detections:
                altura, largura, _= img_final.shape
                for i, detection in enumerate(detections.detections):
                    bbox = detection.location_data.relative_bounding_box
                    x_min = int(bbox.xmin * largura)
                    y_min = int(bbox.ymin * altura)
                    box_larg = int(bbox.width * largura)
                    box_alt = int(bbox.height * altura)

                    # margem estilo 3x4
                    x_min = max(0, x_min - int(box_larg * margem))
                    y_min = max(0, y_min - int(box_alt * margem))
                    x_max = min(largura, x_min + box_larg + int(box_larg * margem * 2))
                    y_max = min(altura, y_min + box_alt + int(box_alt * margem * 3))

                    rosto = img_final[y_min:y_max, x_min:x_max]
                    out_path = os.path.join(pasta_saida, f"rosto_{i+1}.jpg")
                    cv2.imwrite(out_path, rosto)
                    resultados.append(out_path)

        return resultados