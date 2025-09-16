from abc import ABC, abstractmethod
import torch

class AbstractModel(ABC):
    @abstractmethod
    def embed(self, dna_sequence:str) -> torch.Tensor:
        pass