from pbi_models.embedders.abstract_model import AbstractModel
import torch
from pbi_utils.logging import Logging, logging
from pbi_utils.utils import clean_gpu
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig

logger = Logging()

class DNABERT2(AbstractModel):
    def __init__(self, source_code_path: str, device: str = "cpu", max_seq_len: int = 2**15) -> None:

        self.device = device
        
        # Not show internal transformers logging messages
        current_log_level = logging.root.level
        Logging.set_logging_level()

        self.tokenizer = AutoTokenizer.from_pretrained(source_code_path, trust_remote_code=True)
        config = BertConfig.from_pretrained(source_code_path)
        self.model = AutoModel.from_pretrained(source_code_path, trust_remote_code=True, config=config)

        Logging.set_logging_level(current_log_level)

        
        self.model.to(self.device)
        self.model.eval()

        self.max_seq_len = max_seq_len

        logger.debug(f"Max sequence length for DNABERT2: {self.max_seq_len}")

    def embed(self, dna_sequence: str) -> torch.Tensor:
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