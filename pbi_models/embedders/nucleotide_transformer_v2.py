import numpy as np
from tqdm import tqdm
from pbi_models.embedders.abstract_model import AbstractModel
import torch
from pbi_utils.logging import Logging, logging
from typing import Literal
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pbi_utils.embeddings_merging_strategies.abstract_merger_strategy import AbstractMergerStrategy
from pbi_utils.embeddings_merging_strategies.truncate_strategy import TruncateStrategy

logger = Logging()

class NT2(AbstractModel):
    MODEL_NAMES = Literal["nucleotide-transformer-2.5b-multi-species", "nucleotide-transformer-2.5b-1000g", "nucleotide-transformer-500m-human-ref", "nucleotide-transformer-500m-1000g", "nucleotide-transformer-v2-50m-multi-species", "nucleotide-transformer-v2-50m-3mer-multi-species", "nucleotide-transformer-v2-100m-multi-species", "nucleotide-transformer-v2-500m-multi-species", "nucleotide-transformer-v2-250m-multi-species"]
    model_name2short_name = {
        "nucleotide-transformer-2.5b-multi-species": "v1-2.5B-MS",
        "nucleotide-transformer-2.5b-1000g": "v1-2.5B-1KG",
        "nucleotide-transformer-500m-human-ref": "v1-500M-HR",
        "nucleotide-transformer-500m-1000g": "v1-500M-1KG",
        "nucleotide-transformer-v2-50m-multi-species": "50M",
        "nucleotide-transformer-v2-50m-3mer-multi-species": "50M-3mer",
        "nucleotide-transformer-v2-100m-multi-species": "100M",
        "nucleotide-transformer-v2-250m-multi-species": "250M",
        "nucleotide-transformer-v2-500m-multi-species": "500M"
    }
    
    def __init__(self, merging_strategy: AbstractMergerStrategy = TruncateStrategy(), overlap: int = 0, device: str = "cpu", model_name: MODEL_NAMES = "nucleotide-transformer-v2-50m-multi-species"):

        self.device = device
        self.overlap = overlap
        self.merging_strategy = merging_strategy
        self.model_name = model_name


        # Not show internal transformers logging messages
        current_log_level = logging.root.level
        Logging.set_logging_level()
        
        self.tokenizer = AutoTokenizer.from_pretrained(f"InstaDeepAI/{model_name}", trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(f"InstaDeepAI/{model_name}", trust_remote_code=True)
        
        Logging.set_logging_level(current_log_level)


        self.model.to(self.device)

        self.max_seq_len = (self.tokenizer.model_max_length - 1) * 6 # NT tokenizes the sequence as 6-mers, and max_seq_len is for the tokenized sequence. (-1 for the special tokens)


        logger.debug(f"Max sequence length for Nucleotide Transformer: {self.max_seq_len}")

    def embed(self, dna_sequence: str) -> torch.Tensor:

        # Manually split the sequences
        sequences = self._split_sequence(dna_sequence)

        # Only keep the first chunk if using TruncateStrategy. Bad practice, but much faster
        if self.merging_strategy.name() == "TruncateStrategy":
            sequences = [sequences[0]]

        # Get embeddings for each subsequence
        tokens = self._encode(sequences)
        embeddings = self._compute_batch_embeddings(tokens)
        embeddings = embeddings.squeeze(1)

        # Merge the embeddings using the specified strategy
        merged_embedding = self.merging_strategy.merge(sequences, embeddings)
        
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

    def _compute_single_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
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

    # Divide sequence into overlapping subsequences
    def _split_sequence(self, sequence: str) -> list[str]:
        step = self.max_seq_len - self.overlap
        subsequences = [sequence[i:i+self.max_seq_len] for i in range(0, len(sequence), step)]
        return subsequences
    
    def _encode(self, dna_sequence: list[str]) -> torch.Tensor:
        # Tokenize the entire sequence at once
        tokens_ids = self.tokenizer.batch_encode_plus(dna_sequence, return_tensors="pt", padding=True)["input_ids"].to(self.device)
        
        return tokens_ids

    def name(self) -> str:
        return f"NT2-{self.merging_strategy.name()}-{self.model_name2short_name[self.model_name]}"
    
    def __repr__(self):
        return f"NT2(merging_strategy={self.merging_strategy}, overlap={self.overlap}, model_name='{self.model_name}')"