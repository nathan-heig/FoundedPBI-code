from abc import ABC, abstractmethod
import torch

from pbi_utils.embeddings_merging_strategies.abstract_merger_strategy import AbstractMergerStrategy

class AbstractModel(ABC):

    @abstractmethod
    def __init__(self, merging_strategy: AbstractMergerStrategy) -> None:
        self.merging_strategy = merging_strategy
        super().__init__()

    @abstractmethod
    def embed(self, dna_sequence:str) -> torch.Tensor:
        pass

    def name(self) -> str:
        return f"{type(self).__name__}-{self.merging_strategy.name()}"
    
    def __repr__(self):
        return f"{type(self).__name__}(merging_strategy={self.merging_strategy})"