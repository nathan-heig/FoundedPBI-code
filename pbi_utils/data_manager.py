from colorama import Fore
import numpy as np
import pandas as pd
import subprocess
import torch
import h5py
import os
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from typing import Tuple, List
from pbi_utils.logging import Logging

logger = Logging(__name__)

class EmbeddingsManager(ABC):
    @abstractmethod
    def save_embedding(self, id: int, embedding: torch.Tensor, model_name:str, overwrite: bool = False) -> None:
        pass

    @abstractmethod
    def load_embedding(self, id: int, model_name:str, remove: bool = False, device: str = "cpu") -> torch.Tensor | None:
        pass

    @abstractmethod
    def save_embeddings_batch(self, ids: List[int], embeddings: List[torch.Tensor], model_name:str, overwrite: bool = False) -> None:
        pass

    @abstractmethod
    def load_embedding_batch(self, ids: List[int], model_name:str, remove: bool = False, device: str = "cpu") -> List[torch.Tensor]:
        pass

    @abstractmethod
    def remove_key(self, id: int | str, model_name:str, ignore_not_found: bool) -> None:
        pass

class H5pyEmbeddingsManager(EmbeddingsManager):
    def __init__(self, base_path: str) -> None:
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        logger.info(f"Embeddings will be stored or read from {base_path}")
    
    def save_embedding(self, id: int, embedding: torch.Tensor, model_name: str, overwrite: bool = False):
        id = str(id) # type: ignore

        # ensure CPU + numpy
        data = embedding.detach().cpu().float().numpy()

        with h5py.File(os.path.join(self.base_path, model_name + ".h5"), "a") as f:
            if not overwrite and id in f:
                print(f"{id} already exists, skipping it. To overwrite the value, use overwrite=True")
            else:
                self.remove_key(id, model_name, ignore_not_found=True)
                f.create_dataset(id, data=data, compression="gzip")
    
    def save_embeddings_batch(self, ids: List[int], embeddings: List[torch.Tensor], model_name: str, overwrite: bool = False):
        logger.debug(f"Saving {len(ids)} embeddings for model {model_name} to {self.base_path}")
        with h5py.File(os.path.join(self.base_path, model_name + ".h5"), "a") as f:
            for id, embedding in tqdm(zip(ids, embeddings), total=len(ids), desc="Saving embeddings"):
                id = str(id) # type: ignore
                # ensure CPU + numpy
                data = embedding.detach().cpu().float().numpy()
                if not overwrite and id in f:
                    print(f"{id} already exists, skipping it. To overwrite the value, use overwrite=True")
                else:
                    self.remove_key(id, model_name, ignore_not_found=True)
                    f.create_dataset(id, data=data, compression="gzip")

    def load_embedding(self, id: int, model_name: str, remove: bool = False, device: str = "cpu") -> torch.Tensor | None:
        id = str(id) # type: ignore
        with h5py.File(os.path.join(self.base_path, model_name + ".h5"), "r") as f:
            if id not in f:
                print(f"{id} not found")
                embed = None
            else:
                embed = torch.Tensor(f[id][:]).flatten().to(device=device) # type: ignore
                if remove:
                    self.remove_key(id, model_name)

        return embed
    
    def load_embedding_batch(self, ids: List[int], model_name: str, remove: bool = False, device: str = "cpu") -> List[torch.Tensor]:
        logger.debug(f"Loading {len(ids)} embeddings for model {model_name} from {self.base_path}")
        result = []
        with h5py.File(os.path.join(self.base_path, model_name + ".h5"), "r") as f:
            for id in tqdm(ids, desc="Loading embeddings"):
                id = str(id) # type: ignore
                if id not in f:
                    print(f"{id} not found")
                else:
                    result.append(torch.Tensor(f[id][:]).flatten().to(device=device)) # type: ignore
                    if remove:
                        self.remove_key(id, model_name)
        return result
    
    def remove_key(self, id: int | str, model_name: str, ignore_not_found: bool = False): 
        id = str(id) # type: ignore
        with h5py.File(os.path.join(self.base_path, model_name + ".h5"), "a") as f:
            if id in f:
                del f[id]
            elif not ignore_not_found:
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

        logger.info(f"Perphect input files will be read from {base_path}")

    
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        # wc = int(subprocess.run([f"cat {self.base_path}*.csv | wc -l"], capture_output=True, shell=True).stdout) # line count of the files, but read_csv does not reach the end idk why

        logger.info("Reading csv files...")
        with tqdm() as bar:
            couples_df = pd.read_csv(self.couples_path, skiprows=lambda x: bar.update(1) and False) # type: ignore
            bacteria_df = pd.read_csv(self.bacteria_path, skiprows=lambda x: bar.update(1) and False) # type: ignore
            phages_df = pd.read_csv(self.phages_path, skiprows=lambda x: bar.update(1) and False) # type: ignore
        return bacteria_df, phages_df, couples_df