import numpy as np
from tqdm import tqdm
from pbi_models.embedders.abstract_model import AbstractModel
import torch
from pbi_utils.logging import Logging, logging
from typing import Literal

from transformers import AutoTokenizer, AutoModelForMaskedLM

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
        tokens = self.__encode(dna_sequence)
        
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

    def __encode(self, dna_sequence: str) -> torch.Tensor:
        if len(dna_sequence) > self.max_seq_len:
            logger.debug(f"Found DNA sequence longer than max length ({len(dna_sequence)} vs {self.max_seq_len}), truncating it")
            # dna_sequence = dna_sequence[:self.max_seq_len]
        
        return self.tokenizer.batch_encode_plus([dna_sequence], return_tensors="pt", padding="max_length", max_length = self.max_seq_len, truncation=True)["input_ids"].to(self.device)
    
class NT2_sentence_avg(NT2):
    def embed(self, dna_sequence: str) -> torch.Tensor:
        tokens = self.__encode(dna_sequence)

        embeddings_list = []

        # Batch size is not useful, as it takes almost the same time as doing it one by one and it uses much more memory

        if tokens.shape[0] > 50:
            tokens = tqdm(tokens, desc="Embedding chunks")

        for sentence in tokens:
            embeddings = super()._embed_from_tokens(sentence.unsqueeze(0))
            embeddings_list.append(embeddings)

        embeddings = torch.stack(embeddings_list, dim=0)
        
        # Average the embeddings of the tokens in the sequence
        mean_embed = torch.mean(embeddings, dim=0)

        return mean_embed

    def __encode(self, dna_sequence: str) -> torch.Tensor:
        # Tokenize the entire sequence at once
        tokens_ids = self.tokenizer.batch_encode_plus([dna_sequence], return_tensors="pt")["input_ids"].to(self.device)

        # Split the sequence in chunks of max_length
        tokens_list = []
        for token_ids in tokens_ids:
            for i in range(0, len(token_ids), self.max_seq_len):
                # Pad with zeros if the last chunk is smaller than self.max_seq_len
                tokens_list.append(token_ids[i : i + self.max_seq_len])

                # l = torch.zeros(size=self.max_seq_len)
                # l[: min(self.max_seq_len, len(token_ids) - i)] = token_ids[i : i + self.max_seq_len]
                # tokens_list.append(l)
            tokens_list[-1] = torch.nn.functional.pad(tokens_list[-1], (0, self.max_seq_len - len(tokens_list[-1])), value=self.tokenizer.pad_token_id)
        tokens_ids = torch.stack(tokens_list)
        return tokens_ids