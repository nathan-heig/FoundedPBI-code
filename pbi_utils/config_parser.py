import yaml
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from pbi_utils.embeddings_merging_strategies import *
from pbi_models.embedders import *
from pbi_models.embedders.abstract_model import AbstractModel
from pbi_utils.embeddings_merging_strategies.abstract_merger_strategy import AbstractMergerStrategy
from pbi_utils.logging import Logging
import os

logger = Logging()


class StrategyConfig(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)
    strategy: StrategyConfig | str

class InputConfig(BaseModel):
    bacteria_df: str
    phages_df: str
    couples_df: str

class YAMLConfig(BaseModel):
    input_perphect: str | InputConfig # The path of the folder containing the input data in the Perphect format. The folder must contain the following files: `bacteria_df.csv` (with the columns: `bacterium_id,bacterium_sequence,sequence_length`), `phages_df.csv` (with the columns: `phage_id,phage_sequence,sequence_length`) and `couples_df.csv` (with the columns: `id,bacterium_id,phage_id,interaction_type`, where `interaction_type` is either 1 or 0)
    embeddings_dir: str = "data/embeddings" # The path where the embeddings will be stored and read from (default = data/embeddings)
    num_gpu: int = 1 # Number of GPUs available in the system. If 0, the model is run on the CPU (default = 0)
    gpu_id: int = 0 # Index of GPU to be employed (if 'num_gpu' == 1) (default = 0)
    use_cached_embeddings: bool = False # Do not calculate embeddings, use cached ones. If set, `embeddings_dir` must point to a correct dir with the existing embeddings
    no_train: bool = False # Do not train or test the model, just compute the embeddings
    epochs: int = 10 # Number of epochs to train the classifier model
    batch_size: int = 128 # Batch size to use for training and testing of the classifier
    learning_rate: float = 1e-3 # Learning rate to use for training with Adam optimizer
    phages_embedding_models: List[ModelConfig] = Field(default_factory=list) # Name and parameters of the embedding model to use for the phages sequences. Use this flag multiple times to use multiple models
    bacteria_embedding_models: List[ModelConfig] = Field(default_factory=list) # Name and parameters of the embedding model to use for the bacteria sequences. Use this flag multiple times to use multiple models

class Config:
    def __init__(self, yaml_config: YAMLConfig):
        if isinstance(yaml_config.input_perphect, str):
            yaml_config.input_perphect = InputConfig(
                bacteria_df=os.path.join(yaml_config.input_perphect, "bacteria_df.csv"),
                phages_df=os.path.join(yaml_config.input_perphect, "phages_df.csv"),
                couples_df=os.path.join(yaml_config.input_perphect, "couples_df.csv"),
            )
        self.input_perphect = yaml_config.input_perphect
        self.embeddings_dir = yaml_config.embeddings_dir
        self.num_gpu = yaml_config.num_gpu
        self.gpu_id = yaml_config.gpu_id
        self.use_cached_embeddings = yaml_config.use_cached_embeddings
        self.no_train = yaml_config.no_train
        self.epochs = yaml_config.epochs
        self.batch_size = yaml_config.batch_size
        self.learning_rate = yaml_config.learning_rate
        self.device = "cpu" if self.num_gpu == 0 else f"cuda:{self.gpu_id}"
        self.phages_embedding_models = self._parse_models(yaml_config.phages_embedding_models)
        self.bacteria_embedding_models = self._parse_models(yaml_config.bacteria_embedding_models)

    def _parse_models(self, models_config: List[ModelConfig]) -> List[AbstractModel]:
        models = []
        for model_config in models_config:
            if isinstance(model_config.strategy, str):
                merging_strategy = self._get_instance_from_string(model_config.strategy)()
            else:
                merging_strategy = self._get_instance_from_string(model_config.strategy.name)(**model_config.strategy.params)

            model_params = model_config.params
            model_params["merging_strategy"] = merging_strategy
            model_params["device"] = self.device
            model_params["load_model"] = not self.use_cached_embeddings

            model = self._get_instance_from_string(model_config.name)(**model_params)
            models.append(model)
        return models

    def _get_instance_from_string(self, class_name: str):
        if class_name in globals() and (issubclass(globals()[class_name], AbstractModel) or issubclass(globals()[class_name], AbstractMergerStrategy)):
            model_class = globals()[class_name]
            return model_class
        else:
            raise ValueError(f"Class {class_name} not found or is not a subclass of AbstractModel or AbstractMergerStrategy")

    def __repr__(self):
        return (f"Config(input_perphect={self.input_perphect}, embeddings_dir={self.embeddings_dir}, "
                f"num_gpu={self.num_gpu}, gpu_id={self.gpu_id}, use_cached_embeddings={self.use_cached_embeddings}, "
                f"no_train={self.no_train}, epochs={self.epochs}, batch_size={self.batch_size}, "
                f"learning_rate={self.learning_rate}, phages_embedding_models={self.phages_embedding_models}, "
                f"bacteria_embedding_models={self.bacteria_embedding_models})")
    
def parse_config(config_path: str) -> Config:
    with open(config_path, 'r') as f:
        # config_dict = yaml.safe_load(f)
        config_dict = yaml.safe_load(os.path.expandvars(f.read()))
    yaml_config = YAMLConfig(**config_dict)

    c = Config(yaml_config)

    logger.info(f"Configuration loaded from {config_path}: {c}")

    return c