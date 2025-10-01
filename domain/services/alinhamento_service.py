# domain/services/alinhamento_service.py
import cv2
import numpy as np
import os

class AlinhamentoService:
    """
    Avalia um face_roi (numpy BGR) e retorna um score float (quanto mais alinhado ao retrato, maior).
    Usa cascades para detectar olhos/nariz/boca (quando disponíveis). Fallbacks razoáveis.
    """

    def __init__(self):
        base = cv2.data.haarcascades
        self.eye_cascade = cv2.CascadeClassifier(os.path.join(base, "haarcascade_eye.xml"))
        # alguns OpenCVs não têm esses arquivos; tentamos carregar e, se não funcionarem, deixamos None
        self.nose_cascade = None
        self.mouth_cascade = None
        for fname in ("haarcascade_mcs_nose.xml", "haarcascade_mcs_nose.xml"):
            p = os.path.join(base, fname)
            if os.path.exists(p):
                self.nose_cascade = cv2.CascadeClassifier(p)
                break
        # mouth: fallback para smile cascade
        p_smile = os.path.join(base, "haarcascade_smile.xml")
        if os.path.exists(p_smile):
            self.mouth_cascade = cv2.CascadeClassifier(p_smile)

    def _centers_from_rects(self, rects):
        out = []
        for (x,y,w,h) in rects:
            cx = x + w/2.0
            cy = y + h/2.0
            out.append((cx, cy))
        return out

    def avaliar(self, face_roi) -> float:
        """
        face_roi: numpy BGR (apenas a região da face)
        retorna: score float entre 0 e 1 (maior = melhor alinhamento)
        """
        if face_roi is None or face_roi.size == 0:
            return 0.0

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if w > h:
            face_roi = cv2.rotate(face_roi, cv2.ROTATE_90_CLOCKWISE)
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]

        if w == 0:
            return 0.0

        # detect eyes
        eyes = []
        try:
            eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(int(w*0.08), int(h*0.08)))
        except Exception:
            eyes = []

        # detect nose (se available)
        nose = []
        if self.nose_cascade:
            nose = self.nose_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(int(w*0.05), int(h*0.05)))

        # detect mouth (fallback smile)
        mouth = []
        if self.mouth_cascade:
            mouth = self.mouth_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=10, minSize=(int(w*0.08), int(h*0.05)))

        # prepara centros
        eye_centers = self._centers_from_rects(eyes)
        nose_centers = self._centers_from_rects(nose)
        mouth_centers = self._centers_from_rects(mouth)

        # heurística robusta:
        centers_x = []
        centers_y = []
        if len(eye_centers) >= 2:
            sorted_eyes = sorted(eye_centers, key=lambda c: c[0])
            left_eye = sorted_eyes[0]        # mais à esquerda
            right_eye = sorted_eyes[-1]      # mais à direita
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            angulo_olhos = np.degrees(np.arctan2(dy, dx))
            penalidade_angulo = max(0.0, 1.0 - abs(angulo_olhos) / 30.0)  
            # até 30 graus de inclinação aceito
            #score *= penalidade_angulo
        elif len(eye_centers) == 1:
            centers_x.append(eye_centers[0][0]); centers_y.append(eye_centers[0][1])

        if nose_centers:
            centers_x.append(nose_centers[0][0]); centers_y.append(nose_centers[0][1])
        if mouth_centers:
            # escolher boca mais baixa (maior cy)
            mouth_centers_sorted = sorted(mouth_centers, key=lambda c: c[1])
            centers_x.append(mouth_centers_sorted[-1][0]); centers_y.append(mouth_centers_sorted[-1][1])

        # Se não tiver features suficientes, damos um score baixo mas não zero
        if len(centers_x) < 2:
            # fallback: estima alinhamento com base na proporção do rosto (se altura > largura -> retrato)
            ratio = h / (w + 1e-6)
            # alvo aproximado: 4x3 => ratio ~ 4/3 ~ 1.33 (para 3x4 crop vertical); normaliza
            score = max(0.0, min(1.0, (ratio - 0.8) / (1.5 - 0.8)))
            return float(score * 0.4)  # fraco porque não temos landmarks

        # cálculo: desvio padrão das posições x / largura (quanto menor, mais alinhado)
        xs = np.array(centers_x)
        std_x = float(xs.std())
        std_norm = std_x / (w + 1e-6)  # menor é melhor

        # vertical ordering penalization (esperado eyes < nose < mouth)
        vert_penalty = 1.0
        try:
            ys = np.array(centers_y)
            # se temos 3 pontos e a ordem não faz sentido, penaliza
            if len(ys) >= 3:
                # olhos normalmente têm as menores Ys
                if not (ys.min() < np.median(ys) < ys.max()):
                    vert_penalty = 0.85
        except Exception:
            vert_penalty = 0.9

        # transforma em score 0..1
        score = 1.0 / (1.0 + std_norm * 6.0)  # fator 6: controla quão rápido decai
        score *= vert_penalty
        # multiplicador conforme quantas features temos (mais features -> mais confiança)
        feature_mul = min(1.0, len(centers_x) / 3.0)
        score *= (0.6 + 0.4 * feature_mul)  # entre 0.6 e 1.0 de multiplicador

        score = max(0.0, min(1.0, float(score)))
        return score
