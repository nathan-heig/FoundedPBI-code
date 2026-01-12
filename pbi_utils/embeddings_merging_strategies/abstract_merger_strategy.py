from abc import ABC, abstractmethod
import torch


class AbstractMergerStrategy(ABC):
    """
    Abstract class for merging strategies of embeddings. All merging strategies should inherit from this class and implement the merge method.
    """

    @abstractmethod
    def merge(self, sentences: list[str], embeddings: torch.Tensor) -> torch.Tensor:
        """Merge embeddings of sentences into a single embedding.
        Arguments:
            sentences (list[str]): List of sentences (splitted dna chunks).
            embeddings (torch.Tensor): Tensor of shape (N, D) where N is the number of sentences and D is the embedding dimension.
        Returns:
            torch.Tensor: Merged embedding of shape (1, D).
        """
        pass

    def name(self) -> str:
        return type(self).__name__

    def __repr__(self):
        return f"{type(self).__name__}()"
