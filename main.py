import pandas as pd
from tqdm import tqdm
from pbi_utils.data_manager import H5pyEmbeddingsManager, PerphectDataInput, EmbeddingsManager
from pbi_utils.logging import Logging, INFO
from pbi_models.megaDNA import MegaDNA
from pbi_models.classifiers.base import BasicClassifier
from typing import Tuple
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset, DataLoader

tqdm.pandas() # Initialize tqdm with pandas
Logging.set_logging_level(INFO)
logger = Logging(__name__)

def create_embeddings(bacteria_df: pd.DataFrame, phages_df: pd.DataFrame, output_manager: EmbeddingsManager):
    model = MegaDNA(weights_path="foundation-models-test/megaDNA/data/weights/megaDNA_phage_145M.pt", device="cpu")

    bacteria_df.progress_apply(lambda row: output_manager.save_embedding(row["bacterium_id"], model.embed(row["bacterium_sequence"])), axis=1) # type: ignore
    phages_df.progress_apply(lambda row: output_manager.save_embedding(row["phage_id"], model.embed(row["phage_sequence"])), axis=1) # type: ignore

def load_embeddings(bacteria_df: pd.DataFrame, phages_df: pd.DataFrame, output_manager: EmbeddingsManager) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bacteria_encoded = pd.DataFrame(columns=["bacterium_id", "embedding"])
    bacteria_encoded[["bacterium_id", "embedding"]] = bacteria_df.progress_apply((lambda row: (row["bacterium_id"], output_manager.load_embedding(row["bacterium_id"]))), axis=1, result_type="expand") # type: ignore

    phages_encoded = pd.DataFrame(columns=["phage_id", "embedding"])
    phages_encoded[["phage_id", "embedding"]] = phages_df.progress_apply((lambda row: (row["phage_id"], output_manager.load_embedding(row["phage_id"]))), axis=1, result_type="expand") # type: ignore

    return bacteria_encoded, phages_encoded

def make_dataset(couples_df: pd.DataFrame, output_manager: EmbeddingsManager) -> pd.DataFrame:
    result = couples_df.copy(deep=True)

    result["bacterium_embedding"] = couples_df.progress_apply(lambda row: output_manager.load_embedding(row["bacterium_id"]), axis=1) # type: ignore
    result["phage_embedding"] = couples_df.progress_apply(lambda row: output_manager.load_embedding(row["phage_id"]), axis=1) # type: ignore

    return result

def dataframe_to_tf_dataloader(df: pd.DataFrame, batch_size: int):
    dataset = TensorDataset(
        torch.stack(list(df["bacterium_embedding"])),
        torch.stack(list(df["bacterium_embedding"])),
        torch.tensor(df["interaction_type"].values, dtype=torch.long),
        )
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader

def train_model(train_df: pd.DataFrame, model, epochs: int):
    dataloader = dataframe_to_tf_dataloader(train_df, batch_size=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"Starting training for {epochs} epochs...")
    # Typical torch training loop
    for i in range(epochs):
        with tqdm(dataloader, unit="batch", mininterval=0) as tepoch:
            train_loss = 0.0
            for bact_emb, phg_emb, labels in tepoch:   # each [batch, emb_dim]
                tepoch.set_description(f"Epoch: {i}")

                optimizer.zero_grad()
                logits = model(bact_emb, phg_emb)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * bact_emb.size(0)

                predictions = logits.argmax(dim=1, keepdim=True).squeeze()
                correct = (predictions == labels).sum().item()
                accuracy = correct / bact_emb.size(0)

                tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
    logger.info(f"Finished training. Final training accuracy: {accuracy}, loss: {loss}")

def test_model(test_df: pd.DataFrame, model: nn.Module):
    dataloader = dataframe_to_tf_dataloader(test_df, 2)

    test_loss = 0.0
    correct, total = 0,0
    criterion = nn.CrossEntropyLoss()

    for bact_emb, phg_emb, labels in dataloader:
        logits = model(bact_emb, phg_emb)

        predictions = logits.argmax(dim=1, keepdim=True).squeeze()
        correct += (predictions == labels).sum().item()
        total += predictions.size(0)
        loss = criterion(logits,labels)
        test_loss += loss.item() * bact_emb.size(0)

    logger.info(f'Testing Loss:{test_loss/len(dataloader)}')
    logger.info(f'Correct Predictions: {correct}/{total} ({correct/total*100:.2f}%)')

if __name__ == "__main__":
    bacteria_df, phages_df, couples_df = PerphectDataInput(base_path="data/perphect-data/dummy").load()

    output_manager = H5pyEmbeddingsManager("./data/embeddings/megaDNA.h5")

    # create_embeddings(bacteria_df, phages_df, output_manager)
    dataset = make_dataset(couples_df, output_manager)

    train, test = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True)

    bacterium_embed_size = len(train["bacterium_embedding"].iloc[0])
    phage_embed_size = len(train["phage_embedding"].iloc[0])

    model = BasicClassifier(bacterium_embed_size, phage_embed_size, hidden_dim=256) # type: ignore

    train_model(train, model, epochs=5)
    test_model(test, model)






