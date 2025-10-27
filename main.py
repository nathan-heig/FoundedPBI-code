import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from pbi_utils.config_parser import parse_config
from pbi_utils.data_manager import H5pyEmbeddingsManager, PerphectDataInput, EmbeddingsManager
from pbi_utils.logging import Logging, INFO, DEBUG
from pbi_models.classifiers.base import BasicClassifier
from pbi_models.embedders.abstract_model import AbstractModel
from typing import Tuple, List
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchmetrics as tm
import argparse
import time
from pbi_utils.utils import Stats
from pbi_utils.embeddings_merging_strategies import *

tqdm.pandas() # Initialize tqdm with pandas
Logging.set_logging_level(DEBUG)
logger = Logging()

def create_embeddings(bacteria_models: List[AbstractModel], phages_models: List[AbstractModel], bacteria_df: pd.DataFrame, phages_df: pd.DataFrame, output_manager: EmbeddingsManager, overwrite: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"Creating embeddings for {len(bacteria_models)} bacteria models and {len(phages_models)} phage models...")
    # phages_encoded has columns: phage_id, embedding_MegaDNA, embedding_DNABert, etc.
    phages_embed_names = [f"embedding_{model.name()}" for model in phages_models]
    phages_encoded = pd.DataFrame(columns=["phage_id"] + phages_embed_names)
    phages_encoded["phage_id"] = phages_df["phage_id"]

    # Create all the embeddings for one model and then save them all at once
    for phages_model in phages_models:
        logger.debug(f"Creating phage embeddings for model {phages_model.name()}...")
        phages_encoded[f"embedding_{phages_model.name()}"] = phages_df.progress_apply(lambda row: phages_model.embed(row["phage_sequence"]), axis=1) # type: ignore
        output_manager.save_embeddings_batch(phages_encoded["phage_id"], phages_encoded[f"embedding_{phages_model.name()}"], model_name=phages_model.name(), overwrite=overwrite) # type: ignore
    
    # bacteria_encoded has columns: bacterium_id, embedding_MegaDNA, embedding_DNABert, etc.
    bacteria_embed_names = [f"embedding_{model.name()}" for model in bacteria_models]
    bacteria_encoded = pd.DataFrame(columns=["bacterium_id"] + bacteria_embed_names)
    bacteria_encoded["bacterium_id"] = bacteria_df["bacterium_id"]

    # Create all the embeddings for one model and then save them all at once
    for bacteria_model in bacteria_models:
        logger.debug(f"Creating bacteria embeddings for model {bacteria_model.name()}...")
        bacteria_encoded[f"embedding_{bacteria_model.name()}"] = bacteria_df.progress_apply(lambda row: bacteria_model.embed(row["bacterium_sequence"]), axis=1) # type: ignore
        output_manager.save_embeddings_batch(bacteria_encoded["bacterium_id"], bacteria_encoded[f"embedding_{bacteria_model.name()}"], model_name=bacteria_model.name(), overwrite=overwrite) # type: ignore
    
    return bacteria_encoded, phages_encoded

def make_dataset(couples_df: pd.DataFrame, bacteria_model_names: List[str], phages_model_names: List[str], output_manager: EmbeddingsManager, device: str) -> pd.DataFrame:
    result = couples_df.copy(deep=True)

    logger.info(f"Creating dataset (loading embeddings)...")

    bacteria_embeddings = []
    for bacteria_model in bacteria_model_names: 
        bacteria_embeddings.append(output_manager.load_embedding_batch(result["bacterium_id"].tolist(), model_name=bacteria_model, device=device))
    
    # The embeddings are concatenated to form 1 final embedding per bacterium/phage. 
    # TODO: One of the papers mentions that you can also simply add them, to reduce the final size, consider testing it.
    result["bacterium_embedding"] = pd.Series([torch.cat(embeds) for embeds in zip(*bacteria_embeddings)])

    phage_embeddings = []
    for phage_model in phages_model_names:
        phage_embeddings.append(output_manager.load_embedding_batch(result["phage_id"].tolist(), model_name=phage_model, device=device))
    result["phage_embedding"] = pd.Series([torch.cat(embeds) for embeds in zip(*phage_embeddings)])

    logger.debug(f"Final embedding size (bacteria): {len(result['bacterium_embedding'].iloc[0])}")
    logger.debug(f"Final embedding size (phages): {len(result['phage_embedding'].iloc[0])}")

    return result

def dataframe_to_tf_dataloader(df: pd.DataFrame, batch_size: int, device: str):
    dataset = TensorDataset(
        torch.stack(list(df["bacterium_embedding"])),
        torch.stack(list(df["phage_embedding"])),
        torch.tensor(df["interaction_type"].values, dtype=torch.long, device=device),
        )
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader

def train_model(train_df: pd.DataFrame, model: nn.Module, batch_size: int, learning_rate: float, epochs: int, device: str, use_multiple_gpu: bool = True) -> np.ndarray:
    dataloader = dataframe_to_tf_dataloader(train_df, batch_size=batch_size, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # metrics
    accuracy = tm.Accuracy(task="binary").to(device)
    recall = tm.Recall(task="binary").to(device)
    f1 = tm.F1Score(task="binary").to(device)
    cm = tm.ConfusionMatrix(task="binary").to(device)

    if use_multiple_gpu:
        model = nn.DataParallel(model)
    
    model.to(device)

    logger.info(f"Starting training for {epochs} epochs...")
    # Typical torch training loop
    for i in range(epochs):
        with tqdm(dataloader, unit="batch", mininterval=0) as tepoch:
            train_loss = 0.0
            for bact_emb, phg_emb, labels in tepoch:   # each [batch, emb_dim]
                tepoch.set_description(f"Epoch: {i+1}")

                optimizer.zero_grad()
                logits = model(bact_emb, phg_emb)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * bact_emb.size(0)

                predictions = logits.argmax(dim=1, keepdim=True).squeeze()
                acc = accuracy(predictions, labels)
                f1(predictions, labels)
                rec = recall(predictions, labels)
                cm(predictions, labels)

                tepoch.set_postfix(loss=loss.item(), accuracy=100. * acc.item(), recall=100. * rec.item())
    
    logger.info(f"Finished training")
    logger.info(f'Accuracy (train): {accuracy.compute()}')
    logger.info(f'Recall (train): {recall.compute()}')
    logger.info(f'F1 score (train): {f1.compute()}')
    logger.info(f"Loss (train): {loss}")
    cm_mat = cm.compute().cpu().numpy()[::-1, ::-1].T # Transpose anti diagonal (torchmetrics default: TN, FP, FN, TP)
    logger.info(f"Confusion Matrix (train) (TP, FP, FN, TN): {cm_mat[0][0], cm_mat[0][1], cm_mat[1][0], cm_mat[1][1]}")

    return cm_mat

def test_model(test_df: pd.DataFrame, model: nn.Module, batch_size: int, device: str) -> np.ndarray:
    logger.info(f"Starting testing...")
    dataloader = dataframe_to_tf_dataloader(test_df, batch_size, device)

    test_loss = 0.0
    correct, total = 0,0
    criterion = nn.CrossEntropyLoss()

    # metrics
    accuracy = tm.Accuracy(task="binary").to(device)
    recall = tm.Recall(task="binary").to(device)
    f1 = tm.F1Score(task="binary").to(device)
    cm = tm.ConfusionMatrix(task="binary").to(device)

    for bact_emb, phg_emb, labels in dataloader:
        logits = model(bact_emb, phg_emb)

        predictions = logits.argmax(dim=1, keepdim=True).squeeze()
        correct += (predictions == labels).sum().item()
        total += predictions.size(0)
        loss = criterion(logits,labels)
        test_loss += loss.item() * bact_emb.size(0)

        accuracy(predictions, labels)
        f1(predictions, labels)
        recall(predictions, labels)
        cm(predictions, labels)

    logger.info(f'Accuracy (test): {accuracy.compute()}')
    logger.info(f'Recall (test): {recall.compute()}')
    logger.info(f'F1 score (test): {f1.compute()}')
    logger.info(f'Loss (test): {test_loss/len(dataloader.dataset)}') # type: ignore
    cm_mat = cm.compute().cpu().numpy()[::-1, ::-1].T # Transpose anti diagonal (torchmetrics default: TN, FP, FN, TP)
    logger.info(f"Confusion Matrix (test) (TP, FP, FN, TN): {cm_mat[0][0], cm_mat[0][1], cm_mat[1][0], cm_mat[1][1]}")

    return cm_mat

if __name__ == "__main__":
    
    logger.info(f"Running: {' '.join(sys.argv)}")

    # Parse command line args
    parser = argparse.ArgumentParser(description="PBI Pipeline CLI")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    cli_args = parser.parse_args()
    config = parse_config(cli_args.config)

    # Set number of threads for PyTorch
    if config.torch_num_threads > 0:
        logger.debug(f"Setting PyTorch number of threads to {config.torch_num_threads}")
        torch.set_num_threads(config.torch_num_threads)

    # Load input data
    if config.input_perphect is not None:
        bacteria_df, phages_df, couples_df = PerphectDataInput(input_paths=config.input_perphect).load()

    device = config.device
    logger.debug(f"Running on device: {device}")

    output_manager = H5pyEmbeddingsManager(config.embeddings_dir)

    
    # Create embeddings (if not cached)
    if not config.use_cached_embeddings:
        create_embeddings(config.bacteria_embedding_models, config.phages_embedding_models, bacteria_df, phages_df, output_manager, overwrite=True)

    # Create datasets, train and test classifier
    if not config.no_train:
        # Create train and test datasets
        bacteria_model_names = [x.name() for x in config.bacteria_embedding_models]
        phages_model_names = [x.name() for x in config.phages_embedding_models]
        dataset = make_dataset(couples_df, bacteria_model_names, phages_model_names , output_manager, device)
        train, test = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True)

        stats = Stats(config)

        # Instantiate classifier
        bacterium_embed_size = len(train["bacterium_embedding"].iloc[0])
        phage_embed_size = len(train["phage_embedding"].iloc[0])
        model = BasicClassifier(bacterium_embed_size, phage_embed_size, hidden_dim=256)

        stats.update_classifier(model)

        t = time.perf_counter()
        cm = train_model(train, model, batch_size=config.batch_size, learning_rate=config.learning_rate, epochs=config.epochs, device=device, use_multiple_gpu=False)
        train_time = time.perf_counter() - t

        stats.update_train_results(cm, train_time)

        t = time.perf_counter()
        cm = test_model(test, model, batch_size=config.batch_size, device=device)
        test_time = time.perf_counter() - t

        stats.update_test_results(cm, test_time)

        stats.log(logger.info)






