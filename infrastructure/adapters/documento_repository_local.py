import os
from domain.entities.documento import Documento
from domain.repositories.documento_repository import DocumentoRepository

class DocumentoRepositoryLocal(DocumentoRepository):
    def __init__(self, pasta_base="documentos_pdf"):
        self.pasta_base = pasta_base

    def obter_por_id(self, id: str) -> Documento:
        caminho = os.path.join(self.pasta_base, f"{id}.pdf")
        if not os.path.exists(caminho):
            raise FileNotFoundError(f"Documento {id} não encontrado")
        return Documento(id, caminho)