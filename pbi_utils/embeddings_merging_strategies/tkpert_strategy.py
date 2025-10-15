from typing import Literal
from pbi_utils.embeddings_merging_strategies.abstract_merger_strategy import AbstractMergerStrategy
import torch
import numpy as np
from scipy.stats import beta


class TKPertStrategy(AbstractMergerStrategy):
    def __init__(self, J: int = 16, gamma: float = 20, merging_strategy: Literal["avg", "concat"] = "concat") -> None:
        super().__init__()
        self.J = J
        self.gamma = gamma
        self.merging_strategy = merging_strategy

    def merge(self, sentences: list[str], embeddings: torch.Tensor) -> torch.Tensor:
        """Merge embeddings of sentences into a single embedding by concatenating the embeddings weighted by TK-PERT weights.
        Args:
            sentences (list[str]): List of sentences (splitted dna chunks).
            embeddings (torch.Tensor): Tensor of shape (N, D) where N is the number of sentences and D is the embedding dimension.
        Returns:
            torch.Tensor: Merged embedding of shape (1, D).
        """
        embed = self.tk_pert_embedding(embeddings)
        
        return embed
    
    # ---------- PERT-related utilities ----------
    # Coauthored by ChatGPT
    def pert_pdf(self, x, min_val:float=0.0, mode:float=0.5, max_val:float=1.0):
        """Modified PERT probability density function."""
        alpha = 1 + self.gamma * ((mode - min_val) / (max_val - min_val))
        beta_param = 1 + self.gamma * ((max_val - mode) / (max_val - min_val))
        return beta.pdf(x, alpha, beta_param)
    
    def tk_pert_weights(self, num_segments: int) -> torch.Tensor:
        """Compute TK-PERT positional weights for each segment and each window."""
        xs = np.linspace(0, 1, num_segments)
        centers = np.linspace(0, 1, self.J)
        weights = []
        for c in centers:
            w = self.pert_pdf(xs, 0.0, c, 1.0)
            w = np.maximum(w, 1e-12)
            w = w / w.sum()
            weights.append(w)
        weights = np.stack(weights)  # (J, N)
        return torch.tensor(weights, dtype=torch.float32)
    
    def tk_pert_embedding(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute the TK-PERT concatenated embedding from subsequence embeddings."""
        num_segments, _ = embeddings.shape
        W = self.tk_pert_weights(num_segments).to(embeddings.device)
        weighted_parts = torch.matmul(W, embeddings)  # (J, dim)
        weighted_parts = torch.nn.functional.normalize(weighted_parts, dim=1)
        
        if self.merging_strategy == "avg":
            return torch.mean(weighted_parts, dim=0, keepdim=True)  # shape: (1, dim)
        else:  # "concat"
            return weighted_parts.flatten().unsqueeze(0)  # shape: (1, J * dim,)
        
    def name(self) -> str:
        return f"TKPert-{self.merging_strategy}-J{self.J}-g{self.gamma}"
    
    def __repr__(self):
        return f"TKPertStrategy(J={self.J}, gamma={self.gamma}, merging_strategy='{self.merging_strategy}')"