# infrastructure/adapters/face_detector_mediapipe.py
import cv2
import mediapipe as mp
from typing import List, Tuple

class FaceDetectorMediapipe:
    """
    adapter: receber numpy.ndarray BGR e retornar List[ (x,y,w,h), ... ]
    """

    def __init__(self, min_detection_confidence: float = 0.5, model_selection: int = 1):
        self.mp_fd = mp.solutions.face_detection
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        # NÃO manter estado com a solução aberta para permitir múltiplas chamadas sem bloquear
        # Usaremos contexto 'with' em cada chamada (suficiente para este caso)

    def detectar(self, image_bgr) -> List[Tuple[int,int,int,int]]:
        """
        image_bgr: numpy array no formato BGR (cv2)
        retorna: lista de (x,y,w,h) (inteiros). Retorna [] se nada encontrado.
        """
        out = []
        if image_bgr is None:
            return out

        try:
            img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            return out

        h, w = image_bgr.shape[:2]
        with self.mp_fd.FaceDetection(model_selection=self.model_selection,
                                      min_detection_confidence=self.min_detection_confidence) as detector:
            results = detector.process(img_rgb)
            detections = results.detections
            if not detections:
                return out

            for det in detections:
                bbox_rel = det.location_data.relative_bounding_box
                # Alguns valores podem ser negativos (fora do frame). Clamp depois.
                x = int(round(bbox_rel.xmin * w))
                y = int(round(bbox_rel.ymin * h))
                bw = int(round(bbox_rel.width * w))
                bh = int(round(bbox_rel.height * h))

                # clamp para a imagem
                x = max(0, x)
                y = max(0, y)
                bw = max(1, min(bw, w - x))
                bh = max(1, min(bh, h - y))

                out.append((x, y, bw, bh))

        return out
