from pbi_models.embedders.abstract_model import AbstractModel
import torch
from pbi_utils.logging import Logging, logging
from typing import Literal

# Not show internal transformers logging messages
current_log_level = logging.root.level
Logging.set_logging_level()
from transformers import AutoTokenizer, AutoModelForMaskedLM
Logging.set_logging_level(current_log_level)

logger = Logging(__name__)

class NT2(AbstractModel):
    MODEL_NAMES = Literal["nucleotide-transformer-2.5b-multi-species", "nucleotide-transformer-2.5b-1000g", "nucleotide-transformer-500m-human-ref", "nucleotide-transformer-500m-1000g", "nucleotide-transformer-v2-50m-multi-species", "nucleotide-transformer-v2-50m-3mer-multi-species", "nucleotide-transformer-v2-100m-multi-species", "nucleotide-transformer-v2-500m-multi-species", "nucleotide-transformer-v2-250m-multi-species"]
    def __init__(self, device: str = "cpu", model_name: MODEL_NAMES = "nucleotide-transformer-v2-50m-multi-species") -> None:

        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(f"InstaDeepAI/{model_name}", trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(f"InstaDeepAI/{model_name}", trust_remote_code=True)

        self.model.to(self.device)

        self.max_seq_len = self.tokenizer.model_max_length

        logger.debug(f"Max sequence length for megaDNA: {self.max_seq_len}")

    def embed(self, dna_sequence: str) -> torch.Tensor:
        tokens = self.__encode(dna_sequence)
        
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
        mean_embed = torch.sum(attention_mask_unsq*embeddings, axis=-2)/torch.sum(attention_mask_unsq, axis=1)

        return mean_embed

    def __encode(self, dna_sequence: str) -> torch.Tensor:
        if len(dna_sequence) > self.max_seq_len:
            logger.debug(f"Found DNA sequence longer than max length ({len(dna_sequence)} vs {self.max_seq_len}), truncating it")
            # dna_sequence = dna_sequence[:self.max_seq_len]
        
        return self.tokenizer.batch_encode_plus([dna_sequence], return_tensors="pt", padding="max_length", max_length = self.max_seq_len, truncation=True)["input_ids"].to(self.device)