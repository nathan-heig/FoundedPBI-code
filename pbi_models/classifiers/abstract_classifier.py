import torch
from torch import nn
import torch.nn.functional as F
from pbi_utils.logging import Logging
from abc import ABC, abstractmethod

logger = Logging()

class AbstractClassifier(nn.Module, ABC):

    @abstractmethod
    def __init__(self, bacterium_embed_dim: int, phage_embed_dim: int):
        super().__init__()

    @abstractmethod
    def forward(self, bacterium_emb: torch.Tensor, phage_emb: torch.Tensor):
        """
        Inputs:
            bacterium_emb: [batch, emb_dim]
            phage_emb:     [batch, emb_dim]
        Returns:
            logits: [batch, num_classes]
        """
        pass
    
    def name(self) -> str:
        return f"{type(self).__name__}"
    
    def __repr__(self):
        return f"{type(self).__name__}()"