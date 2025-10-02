import cv2
import numpy as np

class SelecionarMelhorRostoUseCase:
    def __init__(self, detector, alinhador, step_degrees=1):
        """
        detector: objeto com método detectar(imagem) -> lista de rostos
        alinhador: objeto com método avaliar(face_roi) -> score float
                   deve ter também eye_cascade para endireitar rosto
        """
        self.detector = detector
        self.alinhador = alinhador
        self.step_degrees = step_degrees

    def endireitar_rosto(self, face_roi):
        """
        Corrige a inclinação do rosto usando a posição dos olhos
        """
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes = self.alinhador.eye_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30)
        )

        if len(eyes) >= 2:
            # pegar os dois olhos mais à esquerda/direita
            eyes_sorted = sorted(eyes, key=lambda e: e[0])
            left_eye = eyes_sorted[0]
            right_eye = eyes_sorted[-1]

            left_center = (left_eye[0] + left_eye[2]//2, left_eye[1] + left_eye[3]//2)
            right_center = (right_eye[0] + right_eye[2]//2, right_eye[1] + right_eye[3]//2)

            dy = right_center[1] - left_center[1]
            dx = right_center[0] - left_center[0]
            angle = np.degrees(np.arctan2(dy, dx))

            h, w = face_roi.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), -angle, 1)
            face_roi = cv2.warpAffine(face_roi, M, (w, h))

        # garante retrato final
        h, w = face_roi.shape[:2]
        if w > h:
            face_roi = cv2.rotate(face_roi, cv2.ROTATE_90_CLOCKWISE)

        return face_roi

    def executar(self, imagem):
        melhor_score = -1
        melhor_face = None

        todos_os_rostos = []
        
        # Testa apenas ângulos principais do documento para detectar rosto
        for angulo in range(0, 360, self.step_degrees):
            print(f"[DEBUG] Testando ângulo {angulo}")
            # faz a rotação e roda o detector
            M = cv2.getRotationMatrix2D(
                (imagem.shape[1]//2, imagem.shape[0]//2),
                angulo, 1
            )
            imagem_rot = cv2.warpAffine(imagem, M, (imagem.shape[1], imagem.shape[0]))
            rostos = self.detector.detectar(imagem_rot)

            for rosto in rostos:
                if isinstance(rosto, dict) and "box" in rosto:
                    x, y, w, h = rosto["box"]
                elif isinstance(rosto, dict) and "bbox" in rosto:
                    x, y, w, h = rosto["bbox"]
                else:
                    x, y, w, h = rosto
                x, y, w, h = map(int, (x, y, w, h))

                margin_y = int(0.4 * h)
                margin_x = int(0.3 * w)
                x1 = max(x - margin_x, 0)
                y1 = max(y - margin_y, 0)
                x2 = min(x + w + margin_x, imagem_rot.shape[1])
                y2 = min(y + h + margin_y, imagem_rot.shape[0])

                face_roi = imagem_rot[y1:y2, x1:x2]
                if face_roi.size == 0:
                    continue

                # Endireita o rosto usando olhos
                face_roi = self.endireitar_rosto(face_roi)
                face_roi = cv2.resize(face_roi, (300, 400))

                score = self.alinhador.avaliar(face_roi)

                todos_os_rostos.append({
                    "face": face_roi,
                    "score": score,
                    "angulo": angulo
                })

                if score > melhor_score:
                    melhor_score = score
                    melhor_face = face_roi.copy()
                    melhor_angulo = angulo

        return melhor_face, todos_os_rostos