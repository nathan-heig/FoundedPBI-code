from pbi_models.abstract_model import AbstractModel
import torch
import numpy as np
import os
import urllib.request
from pbi_utils.logging import Logging
from functools import reduce
logger = Logging(__name__)

class MegaDNA(AbstractModel):
    def __init__(self, weights_path: str, device: str = "cpu") -> None:
        if not os.path.isfile(weights_path):
            logger.warning(f"Weights not found, downloading them again and saving them to: {weights_path}")
            os.makedirs(weights_path.rsplit("/", maxsplit=1)[0], exist_ok=True)
            urllib.request.urlretrieve("https://huggingface.co/lingxusb/megaDNA_updated/resolve/main/megaDNA_phage_145M.pt", weights_path)

        self.device = device
        self.model = torch.load(weights_path, map_location=torch.device(device))

        self.max_seq_len = reduce(lambda x, y: x*y, self.model.max_seq_len)-1
        logger.debug(f"Max sequence length for megaDNA: {self.max_seq_len}")

    def embed(self, dna_sequence: str) -> torch.Tensor:
        input_seq = self.__encode(dna_sequence)
        output = self.model(input_seq, return_value = 'embedding')

        # In the paper, they concatenated all three embeddings, here I'm only returning the last one
        last_embed = output[-1]
        mean_embed = last_embed.mean(dim=1).squeeze(0)
        return mean_embed

    def __vocabulary(self, char: str) -> int:
        match char:
            case "**": return 0
            case "A": return 1
            case "T": return 2
            case "C": return 3
            case "G": return 4
            case "#": return 5
            case _: return 0

    def __encode(self, dna_sequence: str) -> torch.Tensor:
        if len(dna_sequence) > self.max_seq_len:
            logger.debug(f"Found DNA sequence longer than max length ({len(dna_sequence)} vs {self.max_seq_len}), trimming it")
            dna_sequence = dna_sequence[:self.max_seq_len]
        elif len(dna_sequence) < self.max_seq_len:
            old_seq = dna_sequence
            dna_sequence = [0] * self.max_seq_len
            dna_sequence[:len(old_seq)] = old_seq # type: ignore


        return torch.tensor([self.__vocabulary(nt) for nt in dna_sequence] + [self.__vocabulary("#")], device=self.device).unsqueeze(dim=0)