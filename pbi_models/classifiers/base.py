import torch
from torch import nn
import torch.nn.functional as F
from pbi_utils.logging import Logging

logger = Logging(__name__)

class BasicClassifier(nn.Module):
    def __init__(self, bacterium_embed_dim: int, phage_embed_dim: int, hidden_dim: int = 256):
        super().__init__()
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