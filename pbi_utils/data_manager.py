from colorama import Fore
import numpy as np
import pandas as pd
import subprocess
import torch
import h5py
import os
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from typing import Tuple
from pbi_utils.logging import Logging

logger = Logging(__name__)

class EmbeddingsManager(ABC):
    @abstractmethod
    def save_embedding(self, id: int, embedding: torch.Tensor, overwrite: bool = False) -> None:
        pass

    @abstractmethod
    def load_embedding(self, id: int, remove: bool = False) -> torch.Tensor | None:
        pass

    @abstractmethod
    def remove_key(self, id: int) -> None:
        pass

class H5pyEmbeddingsManager(EmbeddingsManager):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        os.makedirs(file_path.rsplit("/", maxsplit=1)[0], exist_ok=True)
    
    def save_embedding(self, id: int, embedding: torch.Tensor, overwrite: bool = False):
        id = str(id) # type: ignore

        # ensure CPU + numpy
        data = embedding.detach().cpu().numpy()

        with h5py.File(self.file_path, "a") as f:
            if not overwrite and id in f:
                print(f"{id} already exists, skipping it. To overwrite the value, use overwrite=True")
            else:
                f.create_dataset(id, data=data, compression="gzip")

    def load_embedding(self, id: int, remove: bool = False) -> torch.Tensor | None:
        id = str(id) # type: ignore
        with h5py.File(self.file_path, "r") as f:
            if id not in f:
                print(f"{id} not found")
                embed = None
            else:
                embed = torch.Tensor(f[id][:]).flatten()
                if remove:
                    self.remove_key(id)

        return embed
    
    def remove_key(self, id: int): 
        id = str(id) # type: ignore
        with h5py.File(self.file_path, "a") as f:
            if id in f:
                del f[id]
            else:
                print(f"{id} not found")

class InputManager(ABC):
    @abstractmethod
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pass

class PerphectDataInput(InputManager):
    def __init__(self, base_path: str) -> None:
        self.base_path = base_path if base_path[-1] == "/" else base_path + "/"
        self.bacteria_path = os.path.join(base_path, "bacteria_df.csv")
        self.phages_path = os.path.join(base_path, "phages_df.csv")
        self.couples_path = os.path.join(base_path, "couples_df.csv")

        if not os.path.isfile(self.bacteria_path):
            logger.warning(f"Bacteria file not found. Path tried: {self.bacteria_path}")
        if not os.path.isfile(self.phages_path):
            logger.warning(f"Phages file not found. Path tried: {self.phages_path}")
        if not os.path.isfile(self.couples_path):
            logger.warning(f"Couples file not found. Path tried: {self.couples_path}")

    
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        # wc = int(subprocess.run([f"cat {self.base_path}*.csv | wc -l"], capture_output=True, shell=True).stdout) # line count of the files, but read_csv does not reach the end idk why

        with tqdm() as bar:
            couples_df = pd.read_csv(self.couples_path, skiprows=lambda x: bar.update(1) and False) # type: ignore
            bacteria_df = pd.read_csv(self.bacteria_path, skiprows=lambda x: bar.update(1) and False) # type: ignore
            phages_df = pd.read_csv(self.phages_path, skiprows=lambda x: bar.update(1) and False) # type: ignore
        return bacteria_df, phages_df, couples_df