from pbi_utils.embeddings_merging_strategies.abstract_merger_strategy import AbstractMergerStrategy
import torch

class AverageStrategy(AbstractMergerStrategy):
    """
    Merge embeddings by averaging them.
    """

    def merge(self, sentences: list[str], embeddings: torch.Tensor) -> torch.Tensor:
        return torch.mean(embeddings, dim=0)