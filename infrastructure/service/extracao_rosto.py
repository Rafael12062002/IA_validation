from infrastructure.adapters.face_detector_mediapipe import FaceDetector

class ExtracaoRostoService:
    def __init__(self):
        self.detector = FaceDetector()

    def extrair(self, caminho_imagem):
        return self.detector.detectar(caminho_imagem)