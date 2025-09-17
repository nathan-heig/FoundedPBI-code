import pandas as pd
from tqdm import tqdm
from pbi_utils.data_manager import H5pyEmbeddingsManager, PerphectDataInput, EmbeddingsManager
from pbi_utils.logging import Logging, INFO, DEBUG
from pbi_models.megaDNA import MegaDNA
from pbi_models.classifiers.base import BasicClassifier
from typing import Tuple
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
import torchmetrics as tm

tqdm.pandas() # Initialize tqdm with pandas
Logging.set_logging_level(DEBUG)
logger = Logging(__name__)

def create_embeddings(bacteria_df: pd.DataFrame, phages_df: pd.DataFrame, output_manager: EmbeddingsManager, device: str, overwrite: bool = False):
    model = MegaDNA(weights_path="./data/weights/megaDNA_phage_145M.pt", device=device)

    bacteria_df.progress_apply(lambda row: output_manager.save_embedding(row["bacterium_id"], model.embed(row["bacterium_sequence"]), overwrite=overwrite), axis=1) # type: ignore
    phages_df.progress_apply(lambda row: output_manager.save_embedding(row["phage_id"], model.embed(row["phage_sequence"]), overwrite=overwrite), axis=1) # type: ignore

def load_embeddings(bacteria_df: pd.DataFrame, phages_df: pd.DataFrame, output_manager: EmbeddingsManager, device: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bacteria_encoded = pd.DataFrame(columns=["bacterium_id", "embedding"])
    bacteria_encoded[["bacterium_id", "embedding"]] = bacteria_df.progress_apply((lambda row: (row["bacterium_id"], output_manager.load_embedding(row["bacterium_id"], device=device))), axis=1, result_type="expand") # type: ignore

    phages_encoded = pd.DataFrame(columns=["phage_id", "embedding"])
    phages_encoded[["phage_id", "embedding"]] = phages_df.progress_apply((lambda row: (row["phage_id"], output_manager.load_embedding(row["phage_id"], device=device))), axis=1, result_type="expand") # type: ignore

    return bacteria_encoded, phages_encoded

def make_dataset(couples_df: pd.DataFrame, output_manager: EmbeddingsManager, device: str) -> pd.DataFrame:
    result = couples_df.copy(deep=True)

    logger.info(f"Creating dataset (loading embeddings)...")
    result["bacterium_embedding"] = couples_df.progress_apply(lambda row: output_manager.load_embedding(row["bacterium_id"], device=device), axis=1) # type: ignore
    result["phage_embedding"] = couples_df.progress_apply(lambda row: output_manager.load_embedding(row["phage_id"], device=device), axis=1) # type: ignore

    return result

def dataframe_to_tf_dataloader(df: pd.DataFrame, batch_size: int, device: str):
    dataset = TensorDataset(
        torch.stack(list(df["bacterium_embedding"])),
        torch.stack(list(df["phage_embedding"])),
        torch.tensor(df["interaction_type"].values, dtype=torch.long, device=device),
        )
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader

def train_model(train_df: pd.DataFrame, model: nn.Module, batch_size: int, epochs: int, device: str, use_multiple_gpu: bool = True):
    dataloader = dataframe_to_tf_dataloader(train_df, batch_size=batch_size, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # metrics
    accuracy = tm.Accuracy(task="binary")
    recall = tm.Recall(task="binary")
    f1 = tm.F1Score(task="binary")

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

                tepoch.set_postfix(loss=loss.item(), accuracy=100. * acc.item(), recall=100. * rec.item())
    
    logger.info(f"Finished training")
    logger.info(f'Accuracy (train): {accuracy.compute()}')
    logger.info(f'Recall (train): {recall.compute()}')
    logger.info(f'F1 score (train): {f1.compute()}')
    logger.info(f"Loss (train): {loss}")

def test_model(test_df: pd.DataFrame, model: nn.Module, batch_size: int, device: str):
    dataloader = dataframe_to_tf_dataloader(test_df, batch_size, device)

    test_loss = 0.0
    correct, total = 0,0
    criterion = nn.CrossEntropyLoss()

    # metrics
    accuracy = tm.Accuracy(task="binary")
    recall = tm.Recall(task="binary")
    f1 = tm.F1Score(task="binary")

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

    logger.info(f'Accuracy (test): {accuracy.compute()}')
    logger.info(f'Recall (test): {recall.compute()}')
    logger.info(f'F1 score (test): {f1.compute()}')
    logger.info(f'Loss (test): {test_loss/len(dataloader.dataset)}') # type: ignore

if __name__ == "__main__":
    bacteria_df, phages_df, couples_df = PerphectDataInput(base_path="data/perphect-data/public_data_set").load()

    output_manager = H5pyEmbeddingsManager("./data/embeddings/megaDNA.h5")

    # device = "cuda:0"
    device = "cpu"

    create_embeddings(bacteria_df, phages_df, output_manager, device=device, overwrite=True)

    dataset = make_dataset(couples_df, output_manager, device)

    train, test = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True)

    bacterium_embed_size = len(train["bacterium_embedding"].iloc[0])
    phage_embed_size = len(train["phage_embedding"].iloc[0])

    model = BasicClassifier(bacterium_embed_size, phage_embed_size, hidden_dim=256) # type: ignore

    train_model(train, model, batch_size=128, epochs=5, device=device, use_multiple_gpu=False)
    test_model(test, model, batch_size=128, device=device)






