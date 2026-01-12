from pbi_utils.embeddings_merging_strategies.abstract_merger_strategy import (
    AbstractMergerStrategy,
)
import torch


class TruncateStrategy(AbstractMergerStrategy):
    """
    Merge embeddings by truncating to the first embedding.
    """

    def merge(self, sentences: list[str], embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings[0].unsqueeze(0)


class BottomTruncateStrategy(AbstractMergerStrategy):
    """
    Merge embeddings by truncating to the last embedding.
    """

    def merge(self, sentences: list[str], embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings[-1].unsqueeze(0)


class TopBottomTruncateStrategy(AbstractMergerStrategy):
    """
    Merge embeddings by concatenating the first and last embeddings.
    """

    def merge(self, sentences: list[str], embeddings: torch.Tensor) -> torch.Tensor:
        return torch.cat([embeddings[0], embeddings[-1]], dim=0).unsqueeze(0)
