import torch
from torch import nn
import torch.nn.functional as F
from pbi_utils.logging import Logging
from pbi_models.classifiers.abstract_classifier import AbstractNNClassifier

logger = Logging()


class BasicClassifier(AbstractNNClassifier):
    """
    A basic feedforward neural network classifier that concatenates bacterium and phage embeddings,
    passes them through a hidden layer with ReLU activation and batch normalization, and outputs logits for
    binary classification.

    Only implemented because it was the first one. It is not recommended for production use.
    """

    def __init__(
        self, bacterium_embed_dim: int, phage_embed_dim: int, hidden_dim: int = 256
    ):
        super().__init__(bacterium_embed_dim, phage_embed_dim)

        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(bacterium_embed_dim + phage_embed_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, bacterium_emb: torch.Tensor, phage_emb: torch.Tensor):
        """
        Inputs:
            bacterium_emb: [batch, emb_dim]
            phage_emb:     [batch, emb_dim]
        Returns:
            logits: [batch, num_classes]
        """
        x = torch.cat([bacterium_emb, phage_emb], dim=1)  # concat along features
        x = F.relu(self.bn1(self.fc1(x)))
        logits = self.fc2(x)
        return logits

    def name(self):
        return f"BasicClassifier({self.hidden_dim})"

    def __repr__(self):
        return f"BasicClassifier(hidden_dim={self.hidden_dim})"
