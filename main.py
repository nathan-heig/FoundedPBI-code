from collections import OrderedDict
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from pbi_models.classifiers.sklearn_classifier import SklearnClassifier
from pbi_utils.config_parser import TrainingConfig, parse_config
from pbi_utils.data_manager import H5pyEmbeddingsManager, PerphectDataInput, EmbeddingsManager
from pbi_utils.logging import Logging, INFO, DEBUG
from pbi_models.embedders.abstract_model import AbstractModel
from typing import Literal, Tuple, List
from sklearn.model_selection import KFold, train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchmetrics as tm
import argparse
import time
from pbi_utils.utils import Stats
from pbi_utils.embeddings_merging_strategies import *
from sklearn.metrics import confusion_matrix

tqdm.pandas() # Initialize tqdm with pandas
Logging.set_logging_level(DEBUG)
logger = Logging()

def create_embeddings(bacteria_models: List[AbstractModel], phages_models: List[AbstractModel], bacteria_df: pd.DataFrame, phages_df: pd.DataFrame, output_manager: EmbeddingsManager, overwrite: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"Creating embeddings for {len(bacteria_models)} bacteria models and {len(phages_models)} phage models...")
    
    # bacteria_encoded has columns: bacterium_id, embedding_MegaDNA, embedding_DNABert, etc.
    bacteria_embed_names = [f"embedding_{model.name()}" for model in bacteria_models]
    bacteria_encoded = pd.DataFrame(columns=["bacterium_id"] + bacteria_embed_names)
    bacteria_encoded["bacterium_id"] = bacteria_df["bacterium_id"]

    # Create all the embeddings for one model and then save them all at once
    for bacteria_model in bacteria_models:
        logger.debug(f"Creating bacteria embeddings for model {bacteria_model.name()}...")
        bacteria_encoded[f"embedding_{bacteria_model.name()}"] = bacteria_df.progress_apply(lambda row: bacteria_model.embed(row["bacterium_sequence"]), axis=1) # type: ignore
        output_manager.save_embeddings_batch(bacteria_encoded["bacterium_id"], bacteria_encoded[f"embedding_{bacteria_model.name()}"], model_name=bacteria_model.name(), overwrite=overwrite) # type: ignore
    
    # phages_encoded has columns: phage_id, embedding_MegaDNA, embedding_DNABert, etc.
    phages_embed_names = [f"embedding_{model.name()}" for model in phages_models]
    phages_encoded = pd.DataFrame(columns=["phage_id"] + phages_embed_names)
    phages_encoded["phage_id"] = phages_df["phage_id"]

    # Create all the embeddings for one model and then save them all at once
    for phages_model in phages_models:
        logger.debug(f"Creating phage embeddings for model {phages_model.name()}...")
        phages_encoded[f"embedding_{phages_model.name()}"] = phages_df.progress_apply(lambda row: phages_model.embed(row["phage_sequence"]), axis=1) # type: ignore
        output_manager.save_embeddings_batch(phages_encoded["phage_id"], phages_encoded[f"embedding_{phages_model.name()}"], model_name=phages_model.name(), overwrite=overwrite) # type: ignore
    
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

def compute_metrics(tn: float, fp: float, fn: float, tp: float) -> Tuple[float, float, float]:
    """Compute Accuracy, Recall and F1Score from the confusion matrix"""
    acc = (tp+tn)/(tp+tn+fp+fn)
    rec = tp/(tp+fn)
    f1 = (2*tp)/(2*tp+fp+fn)

    return acc, rec, f1

def dataframe_to_numpy_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df["interaction_type"]
    X = df[["bacterium_embedding", "phage_embedding"]]

    X = X.apply(lambda x: np.concatenate([x["bacterium_embedding"].cpu().numpy(), x["phage_embedding"].cpu().numpy()], axis=None), axis=1, result_type="expand")
    return X, y

def train_model(train_df: pd.DataFrame, model: nn.Module | SklearnClassifier, training_config: TrainingConfig, device: str, use_multiple_gpu: bool = True, val_df: pd.DataFrame | None = None, verbose: int = 2, progressbar_description: str = "") -> np.ndarray:
    """Allow torch models and scikit-learn models"""
    
    if isinstance(model, nn.Module):
        return train_nn_model(train_df, model, training_config, device, use_multiple_gpu, val_df, verbose, progressbar_description)
    
    else:
        return train_sklearn_model(train_df, model, training_config, device, use_multiple_gpu, val_df, verbose, progressbar_description)

def train_sklearn_model(train_df: pd.DataFrame, model: SklearnClassifier, training_config: TrainingConfig, device: str, use_multiple_gpu: bool = True, val_df: pd.DataFrame | None = None, verbose: int = 2, progressbar_description: str = "") -> np.ndarray:
    if verbose >= 2:
        logger.info(f"Starting training...")

    X_train, y_train = dataframe_to_numpy_X_y(train_df)

    model.fit(X_train, y_train)

    if val_df is not None:
        X_val, y_val = dataframe_to_numpy_X_y(val_df)

        y_pred = model.predict(X_val)

    else:
        y_val = y_train
        y_pred = model.predict(X_train)

    cm = confusion_matrix(y_val, y_pred)

    tn, fp, fn, tp = cm.ravel().tolist()

    acc, rec, f1 = compute_metrics(tn, fp, fn, tp)

    if verbose >= 2:
        logger.info(f"Finished training")
        logger.info(f'Accuracy (train): {acc}')
        logger.info(f'Recall (train): {rec}')
        logger.info(f'F1 score (train): {f1}')
        logger.info(f"Confusion Matrix (train) (TP, FP, FN, TN): {tp, fp, fn, tn}")

    return cm

def train_nn_model(train_df: pd.DataFrame, model: nn.Module, training_config: TrainingConfig, device: str, use_multiple_gpu: bool = True, val_df: pd.DataFrame | None = None, verbose: int = 2, progressbar_description: str = "") -> np.ndarray:
    dataloader = dataframe_to_tf_dataloader(train_df, batch_size=training_config.batch_size, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min" if training_config.monitor_metric_reduce_lr == "loss" else "max", # 'max' for F1/accuracy, 'min' for loss
    factor=training_config.multiplying_factor_reduce_lr,
    patience=training_config.patience_reduce_lr,
)

    # metrics
    accuracy = tm.Accuracy(task="binary").to(device)
    recall = tm.Recall(task="binary").to(device)
    f1 = tm.F1Score(task="binary").to(device)
    cm = tm.ConfusionMatrix(task="binary").to(device)

    if use_multiple_gpu:
        model = nn.DataParallel(model)
    
    model.to(device)

    if verbose >= 2:
        logger.info(f"Starting training for {training_config.epochs} epochs...")

    # Early stopping
    best_metric = -np.inf if training_config.monitor_metric_early_stopping != "loss" else np.inf
    epochs_no_improve = 0
    best_model_state = None
    
    # Typical torch training loop
    with tqdm(range(training_config.epochs), unit="epoch", desc=progressbar_description, disable=verbose<1) as tepochs:
        for epoch in tepochs:
            # Train 1 epoch
            train_loss = 0.0
            for bact_emb, phg_emb, labels in dataloader:   # each [batch, emb_dim]
                optimizer.zero_grad()
                logits = model(bact_emb, phg_emb)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * bact_emb.size(0)

                predictions = logits.argmax(dim=1, keepdim=True).squeeze()
                acc = accuracy(predictions, labels)
                f1s = f1(predictions, labels)
                rec = recall(predictions, labels)
                cm(predictions, labels)

            # Validation results
            if val_df is not None:
                val_cm, val_loss = test_model(val_df, model, training_config.batch_size, device, silent=True)
                # Inside it calls model.eval(), so we need to set it to train mode again
                model.train()

                tn, fp, fn, tp = val_cm.ravel().tolist()
                val_acc, val_rec, val_f1 = compute_metrics(tn, fp, fn, tp)

                # Early stopping
                if training_config.monitor_metric_early_stopping == "f1":
                    current_metric = val_f1
                    improved = current_metric > best_metric
                elif training_config.monitor_metric_early_stopping == "loss":
                    current_metric = val_loss
                    improved = current_metric < best_metric
                else:
                    raise ValueError(f"Unknown metric {training_config.monitor_metric_early_stopping}. monitor_metric_early_stopping must be 'f1' or 'loss'")

                if improved:
                    best_metric = current_metric
                    epochs_no_improve = 0
                    best_model_state = model.state_dict()
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= training_config.patience_early_stopping:
                    if verbose >= 1:
                        logger.debug(f"Early stopping triggered after {epoch+1} epochs")
                    break

                # Learning rate scheduler
                scheduler.step(val_f1)

                tepochs.set_postfix(OrderedDict(lr=optimizer.param_groups[0]['lr'], loss=val_loss, accuracy=100. * val_acc, recall=100. * val_rec, f1=100. * val_f1))
            else:
                tepochs.set_postfix(OrderedDict(loss=loss.item(), accuracy=100. * acc.item(), recall=100. * rec.item(), f1=100. * f1s.item()))
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    cm_mat = cm.compute().cpu().numpy() # (torchmetrics default: TN, FP, FN, TP)
    tn, fp, fn, tp = cm_mat.ravel().tolist()

    if verbose >= 2:
        logger.info(f"Finished training")
        logger.info(f'Accuracy (train): {accuracy.compute()}')
        logger.info(f'Recall (train): {recall.compute()}')
        logger.info(f'F1 score (train): {f1.compute()}')
        logger.info(f"Loss (train): {loss}")
        logger.info(f"Confusion Matrix (train) (TP, FP, FN, TN): {tp, fp, fn, tn}")

    return cm_mat

def test_model(test_df: pd.DataFrame, model: nn.Module | SklearnClassifier, batch_size: int, device: str, silent: bool = False) -> tuple[np.ndarray, float]:
    """Allow torch and scikit-learn models"""

    if isinstance(model, nn.Module):
        return test_nn_model(test_df, model, batch_size, device, silent)
    
    else:
        return test_sklearn_model(test_df, model, batch_size, device, silent)

def test_sklearn_model(test_df: pd.DataFrame, model: SklearnClassifier, batch_size: int, device: str, silent: bool = False) -> tuple[np.ndarray, float]:
    if not silent:
        logger.info(f"Starting testing...")
    
    X_test, y_test = dataframe_to_numpy_X_y(test_df)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel().tolist()
    acc, rec, f1 = compute_metrics(tn, fp, fn, tp)

    if not silent:
        logger.info(f'Accuracy (test): {acc}')
        logger.info(f'Recall (test): {rec}')
        logger.info(f'F1 score (test): {f1}')
        logger.info(f"Confusion Matrix (test) (TP, FP, FN, TN): {tp, fp, fn, tn}")

    return cm, -1

def test_nn_model(test_df: pd.DataFrame, model: nn.Module, batch_size: int, device: str, silent: bool = False) -> tuple[np.ndarray, float]:
    if not silent:
        logger.info(f"Starting testing...")
    dataloader = dataframe_to_tf_dataloader(test_df, batch_size, device)

    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    # metrics
    if not silent:
        accuracy = tm.Accuracy(task="binary").to(device)
        recall = tm.Recall(task="binary").to(device)
        f1 = tm.F1Score(task="binary").to(device)
    cm = tm.ConfusionMatrix(task="binary").to(device)

    model.eval()
    with torch.no_grad():
        for bact_emb, phg_emb, labels in dataloader:
            logits = model(bact_emb, phg_emb)
            loss = criterion(logits,labels)
            test_loss += loss.item() * bact_emb.size(0)

            predictions = logits.argmax(dim=1, keepdim=True).squeeze()

            if not silent:
                accuracy(predictions, labels)
                f1(predictions, labels)
                recall(predictions, labels)
            cm(predictions, labels)

    cm_mat = cm.compute().cpu().numpy() # (torchmetrics default: TN, FP, FN, TP)
    test_loss = test_loss/len(dataloader.dataset) # type: ignore

    tn, fp, fn, tp = cm_mat.ravel().tolist()
    
    if not silent:
        logger.info(f'Accuracy (test): {accuracy.compute()}')
        logger.info(f'Recall (test): {recall.compute()}')
        logger.info(f'F1 score (test): {f1.compute()}')
        logger.info(f'Loss (test): {test_loss}')
        logger.info(f"Confusion Matrix (test) (TP, FP, FN, TN): {tp, fp, fn, tn}")

    return cm_mat, test_loss

def kfold_train(df: pd.DataFrame, model: nn.Module | SklearnClassifier, training_config: TrainingConfig, device: str, use_multiple_gpu: bool = True):
    """
    Train model and test using K-Fold cross validation
    """

    kfold = KFold(n_splits=training_config.k_folds_cv, shuffle=True, random_state=42)
    all_conf_matrices = []

    logger.info(f"Starting {training_config.k_folds_cv}-Fold Cross Validation...")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
        logger.debug(f"Starting fold {fold + 1}...")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        model.reset_model(device)

        train_model(
            train_df=train_df,
            model=model,
            training_config = training_config,
            device=device,
            use_multiple_gpu=use_multiple_gpu,
            val_df=val_df,
            verbose=1,
            progressbar_description=f"Fold {fold + 1}/{training_config.k_folds_cv}"
        )

        # Evaluate model on validation fold
        cm_mat, _ = test_model(
            test_df=val_df,
            model=model,
            batch_size=training_config.batch_size,
            device=device,
            silent=True
        )

        all_conf_matrices.append(cm_mat)

    mean_cm: np.ndarray = sum(all_conf_matrices) / len(all_conf_matrices) # type: ignore
    # tp, fp, fn, tn = mean_cm[0][0], mean_cm[0][1], mean_cm[1][0], mean_cm[1][1]
    tn, fp, fn, tp = mean_cm.ravel().tolist()
    acc, rec, f1 = compute_metrics(tn, fp, fn, tp)

    logger.info(f"Finished Cross Validation training")
    logger.info(f'Accuracy (CV): {acc}')
    logger.info(f'Recall (CV): {rec}')
    logger.info(f'F1 score (CV): {f1}')
    logger.info(f"Confusion Matrix (CV) (TP, FP, FN, TN): ({tp:.2f}, {fp:.2f}, {fn:.2f}, {tn:.2f})")

    return mean_cm

if __name__ == "__main__":
    
    logger.info(f"Running: {' '.join(sys.argv)}")

    # Parse command line args
    parser = argparse.ArgumentParser(description="PBI Pipeline CLI")
    parser.add_argument("--config", "-c", required=False, default=None, type=str, help="Path to YAML config file")
    parser.add_argument("--json-cli", "-j", required=False, default=None, type=str, help="Alternative way to set config. Recieves the entire json as a parameter.")
    cli_args = parser.parse_args()
    assert cli_args.config is not None or cli_args.json_cli is not None, "A configuration is required. Use --help to see the details."

    config = parse_config(cli_args.config, json_cli=cli_args.json_cli)

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
    if config.training_config.do_train:
        # Create train and test datasets
        bacteria_model_names = [x.name() for x in config.bacteria_embedding_models]
        phages_model_names = [x.name() for x in config.phages_embedding_models]
        dataset = make_dataset(couples_df, bacteria_model_names, phages_model_names , output_manager, device)
        train, test = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True)

        stats = Stats(config)

        # Instantiate classifier
        bacterium_embed_size = len(train["bacterium_embedding"].iloc[0])
        phage_embed_size = len(train["phage_embedding"].iloc[0])
        # model = BasicClassifier(bacterium_embed_size, phage_embed_size, hidden_dim=256)
        model = config.classifier(bacterium_embed_size, phage_embed_size, **config.classifier_params)

        stats.update_classifier(model)

        t = time.perf_counter()
        # cm = train_model(train, model, batch_size=config.batch_size, learning_rate=config.learning_rate, epochs=config.epochs, device=device, use_multiple_gpu=False)
        cm = kfold_train(train, model, training_config=config.training_config, device=device, use_multiple_gpu=False)
        train_time = time.perf_counter() - t

        stats.update_train_results(cm, train_time)

        # t = time.perf_counter()
        # cm, _ = test_model(test, model, batch_size=config.batch_size, device=device)
        # test_time = time.perf_counter() - t

        # stats.update_test_results(cm, test_time)

        stats.log(logger.info)






