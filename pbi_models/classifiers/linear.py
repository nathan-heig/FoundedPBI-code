import torch
from torch import nn
import torch.nn.functional as F
from pbi_utils.logging import Logging
from pbi_models.classifiers.abstract_classifier import AbstractClassifier

logger = Logging()

class LinearClassifier(AbstractClassifier):
    def __init__(self, bacterium_embed_dim: int, phage_embed_dim: int):
        super().__init__(bacterium_embed_dim, phage_embed_dim)

        self.linear = nn.Linear(bacterium_embed_dim + phage_embed_dim, 2)
        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, bacterium_emb: torch.Tensor, phage_emb: torch.Tensor):
        """
        Inputs:
            bacterium_emb: [batch, emb_dim]
            phage_emb:     [batch, emb_dim]
        Returns:
            logits: [batch, num_classes]
        """
        x = torch.cat([bacterium_emb, phage_emb], dim=1)  # concat along features

        logits = self.softmax(self.linear(x))

        return logits