import cv2
from domain.entities.rosto import Rosto

class DetectarRostoCascade:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_nose.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml')

    def detectar(self, imagem):
        gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        rostos = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if rostos is None or len(rostos) == 0:
            return []

        rostos = []
        for (x, y, w, h) in rostos:
            rosto_img = imagem[y:y+h, x:x+w]
            rosto_gray = gray[y:y+h, x:x+w]

            olhos = self.eye_cascade.detectMultiScale(rosto_gray)
            nariz = self.nose_cascade.detectMultiScale(rosto_gray)
            boca = self.mouth_cascade.detectMultiScale(rosto_gray)

            olhos_centros = [(ex + ew//2, ey + eh//2) for (ex, ey, ew, eh) in olhos]
            nariz_centros = [(nx + nw//2, ny + nh//2) for (nx, ny, nw, nh) in nariz]
            boca_centros = [(mx + mw//2, my + mh//2) for (mx, my, mw, mh) in boca]

            rosto = Rosto(imagem=rosto_img, olhos=olhos_centros, nariz=nariz_centros, boca=boca_centros)
            rostos.append(rosto)

        return rostos