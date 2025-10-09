import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from pbi_utils.data_manager import H5pyEmbeddingsManager, PerphectDataInput, EmbeddingsManager
from pbi_utils.logging import Logging, INFO, DEBUG
from pbi_models.embedders.megaDNA import MegaDNA
from pbi_models.embedders.nucleotide_transformer_v2 import NT2, NT2_sentence_avg
from pbi_models.embedders.dnabert2 import DNABERT2
from pbi_models.embedders.evo import EVO
from pbi_models.classifiers.base import BasicClassifier
from pbi_models.embedders.abstract_model import AbstractModel
from typing import Tuple, List, get_args
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
import torchmetrics as tm
import argparse
import time
from pbi_utils.utils import Stats

tqdm.pandas() # Initialize tqdm with pandas
Logging.set_logging_level(DEBUG)
logger = Logging()

def load_embedding_models(models_list, device: str) -> List[AbstractModel]:
    # TODO: Refactor this so it is easier to use and understand. Maybe give a way to define the number of arguments and get them automatically, etc
    models = []
    for model_info in models_list:
        # To add a new model, add the corresponding class & switch case. IMPORTANT: The `model_name` must match the class name (You can override the .name() method to change this).
        model_name = model_info[0]
        match model_name:
            case "MegaDNA":
                if len(model_info) > 1:
                    weights_path = model_info[1]
                else:
                    raise ValueError(f"MegaDNA model for bacteria requires a weights path. Usage: `--bacteria-embedding-model MegaDNA path/to/weights.pt`")
                models.append(MegaDNA(weights_path=weights_path, device=device))

            case "NT2":
                if len(model_info) > 1:
                    hf_model_name = model_info[1]
                    if hf_model_name not in get_args(NT2.MODEL_NAMES):
                        raise ValueError(f"Model name <{hf_model_name}> for Nucleotide Transformer is not recognized. The following names are supported: {get_args(NT2.MODEL_NAMES)}")
                    models.append(NT2(device, model_name=hf_model_name))
                else:
                    models.append(NT2(device))
            
            case "NT2_sentence_avg":
                if len(model_info) > 1:
                    hf_model_name = model_info[1]
                    if hf_model_name not in get_args(NT2_sentence_avg.MODEL_NAMES):
                        raise ValueError(f"Model name <{hf_model_name}> for Nucleotide Transformer is not recognized. The following names are supported: {get_args(NT2_sentence_avg.MODEL_NAMES)}")
                    models.append(NT2_sentence_avg(device, model_name=hf_model_name))
                else:
                    models.append(NT2_sentence_avg(device))
            
            
            case "DNABERT2":
                if len(model_info) > 1:
                    source_code_path = model_info[1]
                else:
                    raise ValueError(f"DNABERT2 model requires a path to the downloaded source code. You can download it from `https://huggingface.co/zhihan1996/DNABERT-2-117M?clone=true`, and fix the triton errors (change all `tl.dot(q, k, trans_b=True)` with `tl.dot(q, tl.trans(k))`)")
                if len(model_info) > 2:
                    max_seq_len = int(model_info[2])
                else:
                    max_seq_len = 2**15
                models.append(DNABERT2(source_code_path, device, max_seq_len))

            case "EVO":
                if len(model_info) > 1:
                    hf_model_name = model_info[1]
                    if hf_model_name not in get_args(EVO.MODEL_NAMES):
                        raise ValueError(f"Model name <{hf_model_name}> for EVO is not recognized. The following names are supported: {get_args(EVO.MODEL_NAMES)}")
                else:
                    raise ValueError(f"EVO model requires a model name. The following names are supported: {get_args(EVO.MODEL_NAMES)}")
                if len(model_info) > 2:
                    max_seq_len = int(model_info[2])
                else:
                    max_seq_len = 2**10
    
                models.append(EVO(hf_model_name, device, max_seq_len))

            case _:
                raise ValueError(f"Unknown embedding model: {model_name}")

    return models

def create_embeddings(bacteria_models: List[AbstractModel], phages_models: List[AbstractModel], bacteria_df: pd.DataFrame, phages_df: pd.DataFrame, output_manager: EmbeddingsManager, overwrite: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"Creating embeddings for {len(bacteria_models)} bacteria models and {len(phages_models)} phage models...")
    # phages_encoded has columns: phage_id, embedding_MegaDNA, embedding_DNABert, etc.
    phages_embed_names = [f"embedding_{model.name()}" for model in phages_models]
    phages_encoded = pd.DataFrame(columns=["phage_id"] + phages_embed_names)
    phages_encoded["phage_id"] = phages_df["phage_id"]

    # Create all the embeddings for one model and then save them all at once
    for phages_model in tqdm(phages_models, unit="models", desc="Creating phage embeddings"):
        phages_encoded[f"embedding_{phages_model.name()}"] = phages_df.progress_apply(lambda row: phages_model.embed(row["phage_sequence"]), axis=1) # type: ignore
        output_manager.save_embeddings_batch(phages_encoded["phage_id"], phages_encoded[f"embedding_{phages_model.name()}"], model_name=phages_model.name(), overwrite=overwrite) # type: ignore
    
    # bacteria_encoded has columns: bacterium_id, embedding_MegaDNA, embedding_DNABert, etc.
    bacteria_embed_names = [f"embedding_{model.name()}" for model in bacteria_models]
    bacteria_encoded = pd.DataFrame(columns=["bacterium_id"] + bacteria_embed_names)
    bacteria_encoded["bacterium_id"] = bacteria_df["bacterium_id"]

    # Create all the embeddings for one model and then save them all at once
    for bacteria_model in tqdm(bacteria_models, unit="models", desc="Creating bacteria embeddings"):
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

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Input data. Only allow 1 type of data input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-i", "--input-perphect", type=str, metavar="INPUT_DIR", help="The path of the folder containing the input data in the Perphect format. The folder must contain the following files: `bacteria_df.csv` (with the columns: `bacterium_id,bacterium_sequence,sequence_length`), `phages_df.csv` (with the columns: `phage_id,phage_sequence,sequence_length`) and `couples_df.csv` (with the columns: `id,bacterium_id,phage_id,interaction_type`, where `interaction_type` is either 1 or 0)")

    # Output
    parser.add_argument("-e", "--embeddings-dir", type=str, default="/data/embeddings", help="The path where the embeddings will be stored and read from (default = data/embeddings)")
    parser.add_argument("--use-cached-embeddings", action="store_true", help="Do not calculate embeddings, use cached ones. If set, `embeddings-dir` must point to a correct dir with the existing embeddings")
    # Train&Test
    parser.add_argument("--no-train", action="store_true", help="Do not train or test the model, just compute the embeddings")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the classifier model")
    parser.add_argument("-bs", "--batch-size", type=int, default=128, help="Batch size to use for training and testing of the classifier")
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3, help="Learning rate to use for training with Adam optimizer")

    parser.add_argument("--num-gpu", default=0, type=int, help="Number of GPUs available in the system. If 0, the model is run on the CPU (default = 0)")
    parser.add_argument("--gpu-id", default=0, type=int, help="Index of GPU to be employed (if 'num-gpu' == 1) (default = 0)")

    # Embedding models. Allow multiple attributes per model
    parser.add_argument("-pem", "--phages-embedding-model", type=str, nargs="+", required=True, action="append", help="Name and parameters of the embedding model to use for the phages sequences. Use this flag multiple times to use multiple models. To be usede like: `--phages-embedding-model MegaDNA path/to/weights.pt`")
    parser.add_argument("-bem", "--bacteria-embedding-model", type=str, nargs="+", required=True, action="append", help="Name and parameters of the embedding model to use for the bacteria sequences. Use this flag multiple times to use multiple models. To be usede like: `--phages-embedding-model MegaDNA path/to/weights.pt`")

    return parser.parse_args()

if __name__ == "__main__":
    
    logger.info(f"Running: {' '.join(sys.argv)}")

    args = parse_arguments()

    stats = Stats(args)

    if args.input_perphect is not None:
        bacteria_df, phages_df, couples_df = PerphectDataInput(base_path=args.input_perphect).load()

    device = "cpu" if args.num_gpu == 0 else f"cuda:{args.gpu_id}"
    logger.debug(f"Running on device: {device}")

    output_manager = H5pyEmbeddingsManager(args.embeddings_dir)

    
    # Create embeddings (if not cached)
    if not args.use_cached_embeddings:
        # Load embedding models
        bacteria_models = load_embedding_models(args.bacteria_embedding_model, device=device)
        phages_models = load_embedding_models(args.phages_embedding_model, device=device)
        create_embeddings(bacteria_models, phages_models, bacteria_df, phages_df, output_manager, overwrite=True)

    # Create datasets, train and test classifier
    if not args.no_train:
        # Create train and test datasets
        bacteria_model_names = [x[0] for x in args.bacteria_embedding_model]
        phages_model_names = [x[0] for x in args.phages_embedding_model]
        dataset = make_dataset(couples_df, bacteria_model_names, phages_model_names , output_manager, device)
        train, test = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True)

        # Instantiate classifier
        bacterium_embed_size = len(train["bacterium_embedding"].iloc[0])
        phage_embed_size = len(train["phage_embedding"].iloc[0])
        model = BasicClassifier(bacterium_embed_size, phage_embed_size, hidden_dim=256)

        stats.update_classifier(model)

        t = time.perf_counter()
        cm = train_model(train, model, batch_size=args.batch_size, learning_rate=args.learning_rate, epochs=args.epochs, device=device, use_multiple_gpu=False)
        train_time = time.perf_counter() - t

        stats.update_train_results(cm, train_time)

        t = time.perf_counter()
        cm = test_model(test, model, batch_size=args.batch_size, device=device)
        test_time = time.perf_counter() - t

        stats.update_test_results(cm, test_time)

        stats.log(logger.info)






