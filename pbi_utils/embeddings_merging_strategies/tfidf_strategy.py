from pbi_utils.embeddings_merging_strategies.abstract_merger_strategy import (
    AbstractMergerStrategy,
)
import torch
import numpy as np
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer,
    TfidfTransformer,
)


class TfidfStrategy(AbstractMergerStrategy):
    """
    Merge embeddings by performing a weighted average with TF-IDF weights.
    """

    def __init__(self, k: int = 6) -> None:
        """
        Initialize the TF-IDF merging strategy.

        :param k: k-mer size for TF-IDF computation.
        :type k: int
        """
        super().__init__()
        self.k = k

    def merge(self, sentences: list[str], embeddings: torch.Tensor) -> torch.Tensor:
        # Get TF-IDF weights for each subsequence
        weights = self._get_subsequence_weights(sentences)

        # Perform weighted average using tfidf weights
        weights = torch.tensor(
            weights, dtype=embeddings.dtype, device=embeddings.device
        )
        weights = weights / weights.sum()

        weighted_embed = torch.sum(embeddings * weights.unsqueeze(1), dim=0).unsqueeze(
            0
        )

        return weighted_embed

    # Convert subsequences into k-mer "documents" (Add space between k-mers) to use with TfidfVectorizer
    def _get_kmers(self, seq: str) -> str:
        return " ".join(seq[i : i + self.k] for i in range(len(seq) - self.k + 1))

    def _get_subsequence_weights(self, subsequences: list[str]) -> np.ndarray:
        # Co-authored by ChatGPT.

        docs = [self._get_kmers(subseq) for subseq in subsequences]

        # Compute TF-IDF matrix
        vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"[^ ]+")
        tfidf_matrix = vectorizer.fit_transform(docs)

        # Compute weight per subsequence (mean TF-IDF). We could also use sum, max, etc.
        weights = np.asarray(tfidf_matrix.mean(axis=1)).flatten()  # type: ignore

        # Normalize weights so they sum to 1
        if weights.sum() > 0:
            weights = weights / weights.sum()

        return weights


class Tf4idfStrategy(TfidfStrategy):
    """
    Merge embeddings by performing a weighted average with TF4-IDF weights.
    TF4-IDF is a variant of TF-IDF that achieves better results in natural text, as described in https://arxiv.org/pdf/2304.14796.
    """

    def _get_subsequence_weights(self, subsequences: list[str]) -> np.ndarray:
        """
        Get TF4-IDF weights for each subsequence. Variant of TF-IDF described in https://arxiv.org/pdf/2304.14796, wich achieves better results in natural text.
        """
        docs = [self._get_kmers(subseq) for subseq in subsequences]

        # Compute raw term frequencies using CountVectorizer
        vectorizer = CountVectorizer(analyzer="word", token_pattern=r"[^ ]+")
        term_counts = vectorizer.fit_transform(docs).toarray().astype(float)  # type: ignore # shape: (num_docs, num_terms)

        # Apply TF4 formula
        max_freq_per_doc = term_counts.max(axis=1, keepdims=True)
        max_freq_per_doc[max_freq_per_doc == 0] = 1  # avoid div by zero
        tf4 = 0.4 + 0.6 * (term_counts / max_freq_per_doc)

        # Compute IDF
        transformer = TfidfTransformer(
            norm=None, use_idf=True, smooth_idf=True, sublinear_tf=False
        )
        transformer.fit(term_counts)
        idf = transformer.idf_  # shape: (num_terms,)

        # Combine TF4 and IDF
        tfidf_tf4 = tf4 * idf  # (num_docs, num_terms)

        # Aggregate to one weight per subsequence (mean TF4-IDF per doc). As before, we could also use sum, max, etc.
        weights = tfidf_tf4.mean(axis=1)

        # Normalize weights
        weights = np.maximum(
            weights, 0
        )  # avoid negative weights (which should never happen, but just in case)
        weights = weights / weights.sum()

        return weights
