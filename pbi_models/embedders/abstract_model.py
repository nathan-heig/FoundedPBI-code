from abc import ABC, abstractmethod
import torch

from pbi_utils.embeddings_merging_strategies.abstract_merger_strategy import AbstractMergerStrategy
from pbi_utils.embeddings_merging_strategies.truncate_strategy import TruncateStrategy
from pbi_utils.utils import clean_gpu
from tqdm import tqdm

class AbstractModel(ABC):

    @abstractmethod
    def __init__(self, max_seq_len: int, merging_strategy: AbstractMergerStrategy = TruncateStrategy(), overlap: int = 0) -> None:
        self.merging_strategy = merging_strategy
        self.overlap = int(overlap)
        self.max_seq_len = int(float(max_seq_len))
        super().__init__()

    def embed(self, dna_sequence:str) -> torch.Tensor:
        """Compute the embedding for a DNA sequence.
        The sequence is split into overlapping (or not) subsequences, tokenized using the function _encode that the child class must implement,
        and then the embeddings for each subsequence are computed using the function _compute_single_embedding that the child class must also implement.
        Finally, the embeddings are merged using the specified merging strategy.
        """
        # Manually split the sequences
        sequences = self._split_sequence(dna_sequence)

        # Only keep the first chunk if using TruncateStrategy. Bad practice, but much faster
        if self.merging_strategy.name() == "TruncateStrategy":
            sequences = [sequences[0]]

        clean_gpu()
        # Get embeddings for each subsequence
        tokens = self._encode(sequences)
        embeddings = self._compute_batch_embeddings(tokens)
        embeddings = embeddings.squeeze(1)

        # Merge the embeddings using the specified strategy
        merged_embedding = self.merging_strategy.merge(sequences, embeddings)
        clean_gpu()
        
        return merged_embedding

    def _compute_batch_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        embeddings_list = []

        # Batch size is not useful, as it takes almost the same time as doing it one by one and it uses much more memory

        if tokens.shape[0] > 50:
            tokens = tqdm(tokens, desc="Embedding chunks") # type: ignore

        for sentence in tokens:
            embeddings = self._compute_single_embedding(sentence.unsqueeze(0))
            embeddings_list.append(embeddings)

        embeddings = torch.stack(embeddings_list, dim=0)

        return embeddings

    # Divide sequence into overlapping subsequences
    def _split_sequence(self, sequence: str) -> list[str]:
        step = self.max_seq_len - self.overlap
        subsequences = [sequence[i:i+self.max_seq_len] for i in range(0, len(sequence), step)]
        return subsequences

    @abstractmethod
    def _compute_single_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """Compute the embedding for a single tokenized sequence.
        Args:
            tokens (torch.Tensor): Tokenized sequence of size [1, self.max_seq_len].
        Returns:
            torch.Tensor: Embedding of the sequence of size [1, embedding_size].
        """
        pass

    @abstractmethod
    def _encode(self, dna_sequences: list[str]) -> torch.Tensor:
        """Encode a list of DNA sequences into token tensors.
        Args:
            dna_sequences (list[str]): List of DNA sequences of size < self.max_seq_len.
        Returns:
            torch.Tensor: Tokenized sequences tensor of size [batch_size, self.max_seq_len]."""
        pass

    def name(self) -> str:
        return f"{type(self).__name__}-{self.merging_strategy.name()}"
    
    def __repr__(self):
        return f"{type(self).__name__}(merging_strategy={self.merging_strategy})"