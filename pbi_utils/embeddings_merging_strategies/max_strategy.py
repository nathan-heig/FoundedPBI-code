from pbi_utils.embeddings_merging_strategies.abstract_merger_strategy import (
    AbstractMergerStrategy,
)
import torch


class MaxStrategy(AbstractMergerStrategy):
    """
    Merge embeddings by taking the maximum value across all embeddings.
    """

    def merge(self, sentences: list[str], embeddings: torch.Tensor) -> torch.Tensor:
        max_embed, _ = torch.max(embeddings, dim=0)

        return max_embed
