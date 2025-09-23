from infrastructure.service.extracao_rosto import ExtracaoRostoService

class ExtrairFotoUseCase:
    def __init__(self):
        self.service = ExtracaoRostoService()

    def executar(self, caminho_imagem):
        return self.service.extrair(caminho_imagem)