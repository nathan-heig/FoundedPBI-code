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

logger = Logging()

class EmbeddingsManager(ABC):
    @abstractmethod
    def save_embedding(self, id: int, embedding: torch.Tensor, model_name:str, overwrite: bool = False) -> None:
        pass

    @abstractmethod
    def load_embedding(self, id: int, model_name:str, remove: bool = False, device: str = "cpu") -> torch.Tensor | None:
        pass

    @abstractmethod
    def save_embeddings_batch(self, ids: List[int], embeddings: List[torch.Tensor], model_name:str, overwrite: bool = False, silent: bool = False) -> None:
        pass

    @abstractmethod
    def load_embedding_batch(self, ids: List[int], model_name:str, remove: bool = False, device: str = "cpu", silent: bool = False) -> List[torch.Tensor]:
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
    
    def save_embeddings_batch(self, ids: List[int], embeddings: List[torch.Tensor], model_name: str, overwrite: bool = False, silent: bool = False):
        emb_shape = embeddings[0].shape
        eq_shape = list(map(lambda x: x.shape == emb_shape, embeddings))
        all_eq_shape = np.all(eq_shape)

        logger.debug(f"Saving {len(ids)} embeddings for model {model_name} to {self.base_path}. " + f"Shape: {emb_shape}" if all_eq_shape else f"Found different shapes at position 0 ({emb_shape}) and {eq_shape.index(False)} ({embeddings[eq_shape.index(False)].shape})")
        with h5py.File(os.path.join(self.base_path, model_name + ".h5"), "a") as f:
            for id, embedding in tqdm(zip(ids, embeddings), total=len(ids), desc="Saving embeddings", disable=silent):
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
    
    def load_embedding_batch(self, ids: List[int], model_name: str, remove: bool = False, device: str = "cpu", silent: bool = False) -> List[torch.Tensor]:
        logger.debug(f"Loading {len(ids)} embeddings for model {model_name} from {self.base_path}")
        result = []
        with h5py.File(os.path.join(self.base_path, model_name + ".h5"), "r") as f:
            for id in tqdm(ids, desc="Loading embeddings", disable=silent):
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
    def __init__(self, input_paths) -> None:
        self.bacteria_path = input_paths.bacteria_df
        self.phages_path = input_paths.phages_df
        self.couples_path = input_paths.couples_df

        if not os.path.isfile(self.bacteria_path):
            logger.warning(f"Bacteria file not found. Path tried: {self.bacteria_path}")
        if not os.path.isfile(self.phages_path):
            logger.warning(f"Phages file not found. Path tried: {self.phages_path}")
        if not os.path.isfile(self.couples_path):
            logger.warning(f"Couples file not found. Path tried: {self.couples_path}")

        logger.info(f"Perphect input files will be read from {self.bacteria_path}, {self.phages_path} and {self.couples_path}")

    
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        # wc = int(subprocess.run([f"cat {self.base_path}*.csv | wc -l"], capture_output=True, shell=True).stdout) # line count of the files, but read_csv does not reach the end idk why

        logger.info("Reading csv files...")
        couples_df = pd.read_csv(self.couples_path)
        bacteria_df = pd.read_csv(self.bacteria_path)
        phages_df = pd.read_csv(self.phages_path)
        return bacteria_df, phages_df, couples_df