from pbi_utils.embeddings_merging_strategies.abstract_merger_strategy import AbstractMergerStrategy
import torch

class MaxStrategy(AbstractMergerStrategy):
    def merge(self, sentences: list[str], embeddings: torch.Tensor) -> torch.Tensor:
        """Merge embeddings of sentences into a single embedding by returning the maximum value at each position.
        Args:
            sentences (list[str]): List of sentences (splitted dna chunks).
            embeddings (torch.Tensor): Tensor of shape (N, D) where N is the number of sentences and D is the embedding dimension.
        Returns:
            torch.Tensor: Merged embedding of shape (1, D).
        """
        max_embed, _ = torch.max(embeddings, dim=0)
        
        return max_embed