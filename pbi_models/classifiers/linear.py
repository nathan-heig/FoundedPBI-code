import torch
from torch import nn
from pbi_utils.logging import Logging
from pbi_models.classifiers.abstract_classifier import AbstractNNClassifier

logger = Logging()

class LinearClassifier(AbstractNNClassifier):
    """
    A simple linear classifier that concatenates bacterium and phage embeddings
    and passes them through a single linear layer to output logits for binary classification.
    
    Does not perform well, but serves as a baseline model.
    """

    def __init__(self, bacterium_embed_dim: int, phage_embed_dim: int):
        super().__init__(bacterium_embed_dim, phage_embed_dim)

        self.linear = nn.Linear(bacterium_embed_dim + phage_embed_dim, 2)
        

    def forward(self, bacterium_emb: torch.Tensor, phage_emb: torch.Tensor):
        """
        Inputs:
            bacterium_emb: [batch, emb_dim]
            phage_emb:     [batch, emb_dim]
        Returns:
            logits: [batch, num_classes]
        """
        x = torch.cat([bacterium_emb, phage_emb], dim=1)  # concat along features

        x = self.linear(x)

        return x