from pbi_models.abstract_model import AbstractModel
import torch
import numpy as np

class MegaDNA(AbstractModel):
    def __init__(self, weights_path: str, device: str = "cpu") -> None:
        self.device = device
        self.model = torch.load(weights_path, map_location=torch.device(device))

    def embed(self, dna_sequence: str) -> torch.Tensor:
        input_seq = self.__encode(dna_sequence)
        output = self.model(input_seq, return_value = 'embedding')

        # In the paper, they concatenated all three embeddings, here I'm only returning the last one
        last_embed = output[-1]
        mean_embed = last_embed.mean(dim=1).squeeze(0)
        return mean_embed

    def __encode(self, dna_sequence: str) -> torch.Tensor:
        VOCABULARY = {"**":0, "A":1, "T":2, "C":3, "G":4, "#":5}
        return torch.tensor([VOCABULARY[nt] for nt in dna_sequence] + [VOCABULARY["#"]]).unsqueeze(dim=0).to(self.device)