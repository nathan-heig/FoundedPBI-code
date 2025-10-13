import numpy as np
from tqdm import tqdm
from pbi_models.embedders.abstract_model import AbstractModel
import torch
from pbi_utils.logging import Logging, logging
from typing import Literal
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForMaskedLM
from scipy.stats import beta

logger = Logging()

class NT2(AbstractModel):
    MODEL_NAMES = Literal["nucleotide-transformer-2.5b-multi-species", "nucleotide-transformer-2.5b-1000g", "nucleotide-transformer-500m-human-ref", "nucleotide-transformer-500m-1000g", "nucleotide-transformer-v2-50m-multi-species", "nucleotide-transformer-v2-50m-3mer-multi-species", "nucleotide-transformer-v2-100m-multi-species", "nucleotide-transformer-v2-500m-multi-species", "nucleotide-transformer-v2-250m-multi-species"]
    def __init__(self, device: str = "cpu", model_name: MODEL_NAMES = "nucleotide-transformer-v2-50m-multi-species"):

        self.device = device


        # Not show internal transformers logging messages
        current_log_level = logging.root.level
        Logging.set_logging_level()
        
        self.tokenizer = AutoTokenizer.from_pretrained(f"InstaDeepAI/{model_name}", trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(f"InstaDeepAI/{model_name}", trust_remote_code=True)
        
        Logging.set_logging_level(current_log_level)


        self.model.to(self.device)

        self.max_seq_len = self.tokenizer.model_max_length

        logger.debug(f"Max sequence length for Nucleotide Transformer: {self.max_seq_len}")

    def embed(self, dna_sequence: str) -> torch.Tensor:
        tokens = self._encode(dna_sequence)
        
        return self._embed_from_tokens(tokens)

    def _embed_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        # Compute the embeddings
        attention_mask = tokens != self.tokenizer.pad_token_id
        output = self.model(
            tokens,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Compute sequences embeddings
        embeddings = output['hidden_states'][-1].detach()

        # Add embed dimension axis
        attention_mask_unsq = torch.unsqueeze(attention_mask, dim=-1)

        # Compute mean embeddings per sequence
        mean_embed = torch.sum(attention_mask_unsq*embeddings, axis=-2)/torch.sum(attention_mask_unsq, axis=1) # type: ignore

        return mean_embed

    def _encode(self, dna_sequence: str) -> torch.Tensor:
        if len(dna_sequence) > self.max_seq_len:
            logger.debug(f"Found DNA sequence longer than max length ({len(dna_sequence)} vs {self.max_seq_len}), truncating it")
            # dna_sequence = dna_sequence[:self.max_seq_len]
        
        return self.tokenizer.batch_encode_plus([dna_sequence], return_tensors="pt", padding="max_length", max_length = self.max_seq_len, truncation=True)["input_ids"].to(self.device)
    
class NT2_sentence_avg(NT2):
    """ Embeds a DNA sequence by splitting it into chunks of max_seq_len, embedding each chunk with NT2, and averaging the embeddings. """

    def _compute_batch_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        embeddings_list = []

        # Batch size is not useful, as it takes almost the same time as doing it one by one and it uses much more memory

        if tokens.shape[0] > 50:
            tokens = tqdm(tokens, desc="Embedding chunks") # type: ignore

        for sentence in tokens:
            embeddings = super()._embed_from_tokens(sentence.unsqueeze(0))
            embeddings_list.append(embeddings)

        embeddings = torch.stack(embeddings_list, dim=0)

        return embeddings

    def embed(self, dna_sequence: str) -> torch.Tensor:
        tokens = self._encode(dna_sequence)

        embeddings = self._compute_batch_embeddings(tokens)
        
        # Average the embeddings of the tokens in the sequence
        mean_embed = torch.mean(embeddings, dim=0)

        return mean_embed

    def _encode(self, dna_sequence: str) -> torch.Tensor:
        # Tokenize the entire sequence at once
        tokens_ids = self.tokenizer.batch_encode_plus([dna_sequence], return_tensors="pt")["input_ids"].to(self.device)

        # Split the sequence in chunks of max_length
        tokens_list = []
        for token_ids in tokens_ids:
            for i in range(0, len(token_ids), self.max_seq_len):
                tokens_list.append(token_ids[i : i + self.max_seq_len])

            # Pad with zeros if the last chunk is smaller than self.max_seq_len
            tokens_list[-1] = torch.nn.functional.pad(tokens_list[-1], (0, self.max_seq_len - len(tokens_list[-1])), value=self.tokenizer.pad_token_id)
        tokens_ids = torch.stack(tokens_list)
        return tokens_ids
    
class NT2_sentence_max(NT2_sentence_avg):
    """ Embeds a DNA sequence by splitting it into chunks of max_seq_len, embedding each chunk with NT2, and taking the max of the embeddings. """

    def embed(self, dna_sequence: str) -> torch.Tensor:
        tokens = self._encode(dna_sequence)

        embeddings = self._compute_batch_embeddings(tokens)
        
        # Get the max embeddings of the tokens in the sequence
        max_embed, _ = torch.max(embeddings, dim=0)
        
        return max_embed

class NT2_sentence_tfidf(NT2_sentence_avg):
    """ Embeds a DNA sequence by splitting it into overlapping chunks of max_seq_len, embedding each chunk with NT2, and averaging the embeddings weighted by TF-IDF weights. """

    def embed(self, dna_sequence: str) -> torch.Tensor:
        max_len = (self.max_seq_len - 1) * 6 # NT tokenizes the sequence as 6-mers, and max_seq_len is for the tokenized sequence. (-1 for the special tokens)

        sequences, weights = self.get_subsequence_weights(dna_sequence, max_len, 6, 0)

        tokens = self._encode(sequences)

        embeddings = self._compute_batch_embeddings(tokens)

        embeddings = embeddings.squeeze(1)

        # Perform weighted average using tfidf weights
        weights = torch.tensor(weights, dtype=embeddings.dtype, device=embeddings.device)
        weights = weights / weights.sum()

        weighted_embed = torch.sum(embeddings * weights.unsqueeze(1), dim=0).unsqueeze(0)
        
        return weighted_embed

    def get_subsequence_weights(self, sequence: str, max_len: int, k: int = 6, overlap: int = 0) -> tuple[list[str], np.ndarray]:
        """
        Splits a DNA sequence into overlapping subsequences of up to max_len, computes TF-IDF weights based on k-mer representation, and returns (subsequences, weights).
        Co-authored by ChatGPT.
        """

        # Divide sequence into overlapping subsequences
        step = max_len - overlap
        subsequences = [sequence[i:i+max_len] for i in range(0, len(sequence), step)]
        
        # Convert subsequences into k-mer "documents" (Add space between k-mers)
        def kmers(seq: str) -> str:
            return " ".join(seq[i:i+k] for i in range(len(seq) - k + 1))
        
        docs = [kmers(subseq) for subseq in subsequences]
        
        # Compute TF-IDF matrix
        vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'[^ ]+')
        tfidf_matrix = vectorizer.fit_transform(docs)
        
        # Compute weight per subsequence (mean TF-IDF)
        weights = np.asarray(tfidf_matrix.mean(axis=1)).flatten() # type: ignore
        
        # Normalize weights so they sum to 1
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        return subsequences, weights

    def _encode(self, dna_sequence: list[str]) -> torch.Tensor:
        # Tokenize the entire sequence at once
        tokens_ids = self.tokenizer.batch_encode_plus(dna_sequence, return_tensors="pt", padding=True)["input_ids"].to(self.device)
        
        return tokens_ids
    
class NT2_sentence_TKPERT(NT2_sentence_avg):
    """ Embeds a DNA sequence by splitting it into overlapping chunks of max_seq_len, embedding each chunk with NT2, and concatenating the embeddings weighted by TK-PERT weights. """

    def __init__(self, device: str = "cpu", model_name: NT2.MODEL_NAMES = "nucleotide-transformer-v2-50m-multi-species", J: int = 16, gamma: float = 20, merging_strategy: Literal["avg", "concat"] = "concat"):
        """ J: Number of windows. gamma: Shape parameter for the PERT distribution. """
        super().__init__(device, model_name)

        self.J = J
        self.gamma = gamma
        self.merging_strategy = merging_strategy

    def embed(self, dna_sequence: str) -> torch.Tensor:
        tokens = self._encode(dna_sequence)

        embeddings = self._compute_batch_embeddings(tokens).squeeze(1)
        
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
        elif self.merging_strategy == "concat":
            return weighted_parts.flatten().unsqueeze(0)  # shape: (1, J * dim,)
        else:
            raise ValueError(f"Unknown merging strategy: {self.merging_strategy}")