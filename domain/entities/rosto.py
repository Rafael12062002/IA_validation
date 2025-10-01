class Rosto:
    def __init__(self, imagem, olhos = None, nariz = None, boca = None):
        self.imagem = imagem
        self.olhos = olhos or []
        self.nariz = nariz or []
        self.boca = boca or []