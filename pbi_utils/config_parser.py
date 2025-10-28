import yaml
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal
from pbi_utils.embeddings_merging_strategies import *
from pbi_models.embedders import *
from pbi_models.classifiers import *
from pbi_models.classifiers.abstract_classifier import AbstractClassifier
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

class ClassifierConfig(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)

class TrainingConfig(BaseModel):
    do_train: bool = True # If false, do not train or test the model, just compute the embeddings
    epochs: int = 10 # Number of epochs to train the classifier model
    batch_size: int = 128 # Batch size to use for training and testing of the classifier
    learning_rate: float = 1e-3 # Learning rate to use for training with Adam optimizer
    weight_decay: float = 1e-4 # L2 Regularization for Adam optimizer

    k_folds_cv: int = 3 # Train the classifier model using `k_folds`-Fold cross validation

    patience_early_stopping: int = 10 # Number of epochs to wait without performance increase before the training is interrupted
    monitor_metric_early_stopping: Literal["f1", "loss"] = "f1" # The metric to decide wether the model is performing better or not
    
    patience_reduce_lr: int = 5 # Number of epochs to wait before reducing the learning rate
    monitor_metric_reduce_lr: Literal["f1", "loss"] = "f1" # The metric to decide wether the model is performing better or not
    multiplying_factor_reduce_lr: float = 0.5 # Multiply the learning rate with this after `patience_reduce_lr` epochs without improvement 
    


class YAMLConfig(BaseModel):
    input_perphect: str | InputConfig # The path of the folder containing the input data in the Perphect format. The folder must contain the following files: `bacteria_df.csv` (with the columns: `bacterium_id,bacterium_sequence,sequence_length`), `phages_df.csv` (with the columns: `phage_id,phage_sequence,sequence_length`) and `couples_df.csv` (with the columns: `id,bacterium_id,phage_id,interaction_type`, where `interaction_type` is either 1 or 0)
    embeddings_dir: str = "data/embeddings" # The path where the embeddings will be stored and read from (default = data/embeddings)
    num_gpu: int = 1 # Number of GPUs available in the system. If 0, the model is run on the CPU (default = 0)
    gpu_id: int = 0 # Index of GPU to be employed (if 'num_gpu' == 1) (default = 0)
    use_cached_embeddings: bool = False # Do not calculate embeddings, use cached ones. If set, `embeddings_dir` must point to a correct dir with the existing embeddings
    phages_embedding_models: List[ModelConfig] = Field(default_factory=list) # Name and parameters of the embedding model to use for the phages sequences. Use this flag multiple times to use multiple models
    bacteria_embedding_models: List[ModelConfig] = Field(default_factory=list) # Name and parameters of the embedding model to use for the bacteria sequences. Use this flag multiple times to use multiple models
    classifier: ClassifierConfig = Field(default_factory=lambda: ClassifierConfig(name="LinearClassifier", params={})) # Model to use to classify the embeddings. Must be a subclass of `AbstractClassifier`, implemented in `pbi_models.classifiers`
    torch_num_threads: int = -1 # Number of threads used by PyTorch. If -1, the maximum number of threads is used
    training_config: TrainingConfig

class Config:
    def __init__(self, yaml_config: YAMLConfig, raw_dict):
        self.raw_dict = raw_dict
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
        self.training_config = yaml_config.training_config
        self.device = "cpu" if self.num_gpu == 0 else f"cuda:{self.gpu_id}"
        self.phages_embedding_models = self._parse_models(yaml_config.phages_embedding_models)
        self.bacteria_embedding_models = self._parse_models(yaml_config.bacteria_embedding_models)
        self.classifier = self._get_instance_from_string(yaml_config.classifier.name, subclass_of=AbstractClassifier) # The classifier is not instantiated yet because we need to know the embedding dimensions first, and that is only known after loading them (in main.py)
        self.classifier_params = yaml_config.classifier.params
        self.torch_num_threads = yaml_config.torch_num_threads

    def _parse_models(self, models_config: List[ModelConfig]) -> List[AbstractModel]:
        models = []
        for model_config in models_config:
            if isinstance(model_config.strategy, str):
                merging_strategy = self._get_instance_from_string(model_config.strategy, subclass_of=AbstractMergerStrategy)()
            else:
                merging_strategy = self._get_instance_from_string(model_config.strategy.name, subclass_of=AbstractMergerStrategy)(**model_config.strategy.params)

            model_params = model_config.params
            model_params["merging_strategy"] = merging_strategy
            model_params["device"] = self.device
            model_params["load_model"] = not self.use_cached_embeddings

            model = self._get_instance_from_string(model_config.name, subclass_of=AbstractModel)(**model_params)
            models.append(model)
        return models

    def _get_instance_from_string(self, class_name: str, subclass_of: type):
        if class_name in globals() and issubclass(globals()[class_name], subclass_of):
            model_class = globals()[class_name]
            return model_class
        else:
            raise ValueError(f"Class {class_name} not found or is not a subclass of {subclass_of.__name__}.")

    def __repr__(self):
        return (f"Config(input_perphect={self.input_perphect}, embeddings_dir={self.embeddings_dir}, "
                f"num_gpu={self.num_gpu}, gpu_id={self.gpu_id}, use_cached_embeddings={self.use_cached_embeddings}, "
                f"training_config=TrainingConfig({self.training_config}), "
                f"phages_embedding_models={self.phages_embedding_models}, "
                f"bacteria_embedding_models={self.bacteria_embedding_models}, "
                f"torch_num_threads={self.torch_num_threads}, "
                f"classifier={self.classifier.__name__}({self.classifier_params})"
                f")"
                )
    
def parse_config(config_path: str | None, json_cli: str | None) -> Config:
    if config_path is not None:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(os.path.expandvars(f.read())) # Read file replacing env vars with their value

    elif json_cli is not None:
        config_dict = yaml.safe_load(json_cli)
    
    else:
        raise ValueError("Either config_path or json_cli must have a value.")
    
    yaml_config = YAMLConfig(**config_dict)

    c = Config(yaml_config, config_dict)

    logger.info(f"Configuration loaded from {config_path}: {c}")

    return c