from abc import ABC, abstractmethod
from domain.entities.documento import Documento

class DocumentoRepository(ABC):
    @abstractmethod
    def obter_por_id(self, id: str) -> Documento:
        pass