from typing import Literal
from pbi_models.embedders.abstract_model import AbstractModel
import torch
from pbi_utils.logging import Logging, logging
from pbi_utils.utils import clean_gpu

# Not show internal transformers logging messages
current_log_level = logging.root.level
Logging.set_logging_level()
from transformers import AutoModelForCausalLM, AutoTokenizer
Logging.set_logging_level(current_log_level)

logger = Logging()

class EVO(AbstractModel):
    MODEL_NAMES = Literal["evo-1.5-8k-base", "evo-1-8k-base", "evo-1-131k-base"]
    def __init__(self, model_name: MODEL_NAMES, device: str = "cpu", max_seq_len: int = 2**10, load_model: bool = True, **kwargs) -> None:

        self.device = device
        self.load_model = load_model

        if self.load_model:
            self.tokenizer = AutoTokenizer.from_pretrained(f"togethercomputer/{model_name}", trust_remote_code=True, **kwargs)
            self.model = AutoModelForCausalLM.from_pretrained(f"togethercomputer/{model_name}", trust_remote_code=True)
            
            self.model.to(self.device)
            self.model.eval()

        self.max_seq_len = max_seq_len

        logger.debug(f"Max sequence length for EVO: {self.max_seq_len}")

    def embed(self, dna_sequence: str) -> torch.Tensor:
        if not self.load_model:
            raise RuntimeError("Model not loaded. If you want to compute embeddings, please set load_model to True when initializing the class.")
        clean_gpu()
        with torch.no_grad():
            tokens = self._encode(dna_sequence)
            # Compute the embeddings
            output = self.model(tokens)[0] # [1, sequence_length, 768]

        # embedding with mean pooling
        mean_embed = torch.mean(output[0], dim=0)

        clean_gpu()

        return mean_embed

    def _encode(self, dna_sequence: str) -> torch.Tensor:
        if len(dna_sequence) > self.max_seq_len:
            logger.debug(f"Found DNA sequence longer than max length ({len(dna_sequence)} vs {self.max_seq_len}), truncating it")
            dna_sequence = dna_sequence[:self.max_seq_len]
        
        return self.tokenizer(dna_sequence, return_tensors = 'pt')["input_ids"].to(self.device)