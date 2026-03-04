from collections import OrderedDict
import json
import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
from pbi_models.classifiers.sklearn_classifier import SklearnClassifier
from pbi_utils.config_parser import TrainingConfig, parse_config
from pbi_utils.data_manager import (
    H5pyEmbeddingsManager,
    PerphectDataInput,
    EmbeddingsManager,
)
from pbi_utils.logging import Logging, INFO, DEBUG
from pbi_models.embedders.abstract_model import AbstractModel
from typing import Literal, Tuple, List
from sklearn.model_selection import (
    KFold,
    train_test_split,
    GroupKFold,
    StratifiedGroupKFold,
)
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchmetrics as tm
import argparse
import time
from pbi_utils.utils import Stats
from pbi_utils.embeddings_merging_strategies import *
from sklearn.metrics import confusion_matrix
from pbi_utils.types import *
import os
from sklearn.decomposition import PCA

tqdm.pandas()  # Initialize tqdm with pandas
Logging.set_logging_level(DEBUG)
logger = Logging()


def create_embeddings_bacteria(
    bacteria_models: List[AbstractModel],
    compute_bacteria_embeddings: List[CACHED_EMBEDDINGS_OPTION],
    bacteria_df: pd.DataFrame,
    output_manager: EmbeddingsManager,
) -> None:
    """
    Create embeddings for all bacteria using the provided models and save them using the output manager. If embeddings already exist and the corresponding compute_bacteria_embeddings is `auto`, they will not be recomputed.

    :param bacteria_models: List of bacteria embedding models (instances of AbstractModel) to use. They can be loaded or not depending on the `use_cached_embeddings` option in the config.
    :type bacteria_models: List[AbstractModel]
    :param compute_bacteria_embeddings: List indicating whether to compute embeddings for each model or use cached ones. If "auto", embeddings will be computed only if they do not already exist. Use the one provided by the config parser.
    :type compute_bacteria_embeddings: List[CACHED_EMBEDDINGS_OPTION]
    :param bacteria_df: DataFrame containing the bacteria data with at least the columns `bacterium_id` and `bacterium_sequence`.
    :type bacteria_df: pd.DataFrame
    :param output_manager: EmbeddingsManager instance to handle saving and loading of embeddings.
    :type output_manager: EmbeddingsManager
    """

    # bacteria_encoded has columns: bacterium_id, embedding_MegaDNA, embedding_DNABert, etc.
    bacteria_embed_names = [
        f"embedding_{model.name()}" for model in bacteria_models if model.is_loaded()
    ]

    logger.info(
        f"Creating embeddings for {len(bacteria_embed_names)} bacteria models..."
    )

    bacteria_encoded = pd.DataFrame(columns=["bacterium_id"] + bacteria_embed_names)
    bacteria_encoded["bacterium_id"] = bacteria_df["bacterium_id"]

    # Create all the embeddings for one model and then save them all at once
    for bacteria_model, compute_embeddings in zip(
        bacteria_models, compute_bacteria_embeddings
    ):
        if not bacteria_model.is_loaded():
            logger.debug(
                f"Skipping bacteria model {bacteria_model.name()} (use_cached_embeddings=True)."
            )
            continue
        device = bacteria_model.device
        logger.debug(
            f"Creating bacteria embeddings for model {bacteria_model.name()}..."
        )

        def _embed_or_load(row):
            if compute_embeddings == "auto" and output_manager.has_key(
                id=row["bacterium_id"], model_name=bacteria_model.name()
            ):
                logger.trace(
                    f"Loading cached embedding for bacterium_id={row['bacterium_id']}"
                )
                return output_manager.load_embedding(
                    id=row["bacterium_id"],
                    model_name=bacteria_model.name(),
                    device=device,
                )
            else:
                logger.trace(
                    f"Computing embedding for bacterium_id={row['bacterium_id']}"
                )
                return bacteria_model.embed(row["bacterium_sequence"])

        bacteria_encoded[f"embedding_{bacteria_model.name()}"] = bacteria_df.progress_apply(_embed_or_load, axis=1)  # type: ignore
        output_manager.save_embeddings_batch(bacteria_encoded["bacterium_id"], bacteria_encoded[f"embedding_{bacteria_model.name()}"], model_name=bacteria_model.name(), overwrite=True)  # type: ignore


def create_embeddings_phages(
    phages_models: List[AbstractModel],
    compute_phages_embeddings: List[CACHED_EMBEDDINGS_OPTION],
    phages_df: pd.DataFrame,
    output_manager: EmbeddingsManager,
) -> None:
    """
    Create embeddings for all phages using the provided models and save them using the output manager. If embeddings already exist and the corresponding compute_phages_embeddings is `auto`, they will not be recomputed.

    :param phages_models: List of phage embedding models (instances of AbstractModel) to use. They can be loaded or not depending on the `use_cached_embeddings` option in the config.
    :type phages_models: List[AbstractModel]
    :param compute_phages_embeddings: List indicating whether to compute embeddings for each model or use cached ones. If "auto", embeddings will be computed only if they do not already exist. Use the one provided by the config parser.
    :type compute_phages_embeddings: List[CACHED_EMBEDDINGS_OPTION]
    :param phages_df: DataFrame containing the phages data with at least the columns `phage_id` and `phage_sequence`.
    :type phages_df: pd.DataFrame
    :param output_manager: EmbeddingsManager instance to handle saving and loading of embeddings.
    :type output_manager: EmbeddingsManager
    """

    # phages_encoded has columns: phage_id, embedding_MegaDNA, embedding_DNABert, etc.
    phages_embed_names = [
        f"embedding_{model.name()}" for model in phages_models if model.is_loaded()
    ]

    logger.info(f"Creating embeddings for {len(phages_embed_names)} phages models...")

    phages_encoded = pd.DataFrame(columns=["phage_id"] + phages_embed_names)
    phages_encoded["phage_id"] = phages_df["phage_id"]

    # Create all the embeddings for one model and then save them all at once
    for phages_model, compute_embeddings in zip(
        phages_models, compute_phages_embeddings
    ):
        if not phages_model.is_loaded():
            logger.debug(
                f"Skipping phage model {phages_model.name()} (use_cached_embeddings=True)."
            )
            continue
        device = phages_model.device
        logger.debug(f"Creating phage embeddings for model {phages_model.name()}...")

        def _embed_or_load(row):
            if compute_embeddings == "auto" and output_manager.has_key(
                id=row["phage_id"], model_name=phages_model.name()
            ):
                logger.trace(f"Loading cached embedding for phage_id={row['phage_id']}")
                return output_manager.load_embedding(
                    id=row["phage_id"], model_name=phages_model.name(), device=device
                )
            else:
                logger.trace(f"Computing embedding for phage_id={row['phage_id']}")
                return phages_model.embed(row["phage_sequence"])

        phages_encoded[f"embedding_{phages_model.name()}"] = phages_df.progress_apply(_embed_or_load, axis=1)  # type: ignore
        output_manager.save_embeddings_batch(phages_encoded["phage_id"], phages_encoded[f"embedding_{phages_model.name()}"], model_name=phages_model.name(), overwrite=True)  # type: ignore


def make_dataset(
    couples_df: pd.DataFrame,
    bacteria_model_names: List[str],
    phages_model_names: List[str],
    output_manager: EmbeddingsManager,
    device: str,
) -> pd.DataFrame:
    """
    Create a dataset with embeddings for bacteria and phages by loading them from the output manager. The final DataFrame will have columns: bacterium_id, phage_id, interaction_type, bacterium_embedding, phage_embedding.

    :param couples_df: DataFrame containing the couples data with columns `bacterium_id`, `phage_id` and `interaction_type`.
    :type couples_df: pd.DataFrame
    :param bacteria_model_names: List of bacteria embedding model names to load embeddings from. Used to load the corresponding embeddings for each bacterium.
    :type bacteria_model_names: List[str]
    :param phages_model_names: List of phage embedding model names to load embeddings from.
    :type phages_model_names: List[str]
    :param output_manager: EmbeddingsManager instance to handle loading of embeddings.
    :type output_manager: EmbeddingsManager
    :param device: Device to load the embeddings onto (e.g., "cpu" or "cuda:0").
    :type device: str
    :return: DataFrame with columns: bacterium_id, phage_id, interaction_type, bacterium_embedding, phage_embedding.
    :rtype: DataFrame
    """

    # Helper function to average tensors of different lengths. Not currently used, but kept for reference.
    def avg_tensors(sequences):
        import torch.nn.functional as F

        num = len(sequences)
        max_len = max([s.size(0) for s in sequences])
        out_dims = (num, max_len)
        out_tensor = sequences[0].data.new(*out_dims).fill_(0)
        mask = sequences[0].data.new(*out_dims).fill_(0)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length] = tensor
            mask[i, :length] = 1

        return torch.mean(out_tensor, dim=0)

    result = couples_df.copy(deep=True)

    logger.info(f"Creating dataset (loading embeddings)...")

    bacteria_embeddings = []
    for bacteria_model in bacteria_model_names:
        bacteria_embeddings.append(
            output_manager.load_embedding_batch(
                result["bacterium_id"].tolist(),
                model_name=bacteria_model,
                device=device,
            )
        )

    # The embeddings are concatenated to form 1 final embedding per bacterium/phage.
    result["bacterium_embedding"] = pd.Series(
        [torch.cat(embeds) for embeds in zip(*bacteria_embeddings)]
    )
    # One of the papers mentions that you can also simply add them, to reduce the final size, but did not obtain better results
    # result["bacterium_embedding"] = pd.Series([avg_tensors(embeds) for embeds in zip(*bacteria_embeddings)]) # Avg of the embeddings instead of concat

    phage_embeddings = []
    for phage_model in phages_model_names:
        phage_embeddings.append(
            output_manager.load_embedding_batch(
                result["phage_id"].tolist(), model_name=phage_model, device=device
            )
        )
    result["phage_embedding"] = pd.Series(
        [torch.cat(embeds) for embeds in zip(*phage_embeddings)]
    )
    # result["phage_embedding"] = pd.Series([avg_tensors(embeds) for embeds in zip(*phage_embeddings)]) # Avg of the embeddings instead of concat

    logger.debug(
        f"Final embedding size (bacteria): {len(result['bacterium_embedding'].iloc[0])}"
    )
    logger.debug(
        f"Final embedding size (phages): {len(result['phage_embedding'].iloc[0])}"
    )

    return result


def reduce_dimensionality(
    dataset: pd.DataFrame,
    technique: DIMENSIONALITY_REDUCTION_TECHNIQUE,
    output_dir: str | None,
    n_components_bact: int | None = None,
    n_components_phag: int | None = None,
) -> pd.DataFrame:
    """
    Reduce the dimensionality of the embeddings in the dataset using the specified technique. Supported techniques are "none" (no reduction), "PCA" (Principal Component Analysis).

    If output_dir is provided and PCA is used, plots of the explained variance will be saved in that directory.

    :param dataset: DataFrame containing the dataset with columns `bacterium_embedding` and `phage_embedding` as tensors.
    :type dataset: pd.DataFrame
    :param technique: Dimensionality reduction technique to apply. Must be one of `DIMENSIONALITY_REDUCTION_TECHNIQUE`.
    :type technique: DIMENSIONALITY_REDUCTION_TECHNIQUE
    :param output_dir: Directory to save output plots (if PCA is used). If None, no plots will be saved.
    :type output_dir: str | None
    :param n_components_bact: Number of components to keep for bacteria embeddings. If None, all components are kept.
    :type n_components_bact: int | None
    :param n_components_phag: Number of components to keep for phage embeddings. If None, all components are kept.
    :type n_components_phag: int | None
    :return: DataFrame with reduced dimensionality embeddings.
    :rtype: DataFrame
    """

    def plot_exp_variance(
        exp_var, output_path: str, title: str = "PCA Explained Variance"
    ):
        cum_sum = np.cumsum(exp_var)

        _, ax = plt.subplots(figsize=(8, 4))
        ax.step(
            range(1, len(cum_sum) + 1),
            cum_sum,
            where="mid",
            label="Cumulative explained variance",
        )

        ax.set_ylabel("Explained variance")
        ax.set_xlabel("nº of components")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_path)

    if technique == "none":
        pass

    elif technique == "PCA":
        pca_bact = PCA(random_state=42, n_components=n_components_bact)
        dataset["bacterium_embedding"] = list(torch.from_numpy(pca_bact.fit_transform(dataset["bacterium_embedding"].apply(lambda x: x.detach().cpu().numpy()).to_list())).float())  # type: ignore

        pca_phag = PCA(random_state=42, n_components=n_components_phag)
        dataset["phage_embedding"] = list(torch.from_numpy(pca_phag.fit_transform(dataset["phage_embedding"].apply(lambda x: x.detach().cpu().numpy()).to_list())).float())  # type: ignore

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

            plot_exp_variance(
                pca_bact.explained_variance_ratio_,
                os.path.join(output_dir, "pca_explained_variance_bacterium.png"),
                "PCA Explained Variance (Bacterium Embedding)",
            )
            plot_exp_variance(
                pca_phag.explained_variance_ratio_,
                os.path.join(output_dir, "pca_explained_variance_phage.png"),
                "PCA Explained Variance (Phage Embedding)",
            )

        logger.debug(
            f"Embedding size (After dimensionality reduction) (bacteria): {len(dataset['bacterium_embedding'].iloc[0])}"
        )
        logger.debug(
            f"Embedding size (After dimensionality reduction) (phages): {len(dataset['phage_embedding'].iloc[0])}"
        )

    else:
        raise ValueError(
            f"{technique} for dimensionality reduction not supported. Allowed values: {DIMENSIONALITY_REDUCTION_TECHNIQUE}"
        )

    return dataset


def dataframe_to_tf_dataloader(df: pd.DataFrame, batch_size: int, device: str):
    """
    Convert a DataFrame with embeddings and interaction types into a PyTorch DataLoader for training/testing.

    :param df: DataFrame containing the dataset with columns `bacterium_embedding`, `phage_embedding`, and `interaction_type`.
    :type df: pd.DataFrame
    :param batch_size: Batch size for the DataLoader.
    :type batch_size: int
    :param device: Device to load the tensors onto (e.g., "cpu" or "cuda:0").
    :type device: str
    """

    dataset = TensorDataset(
        torch.stack(list(df["bacterium_embedding"])).to(device),
        torch.stack(list(df["phage_embedding"])).to(device),
        torch.tensor(df["interaction_type"].values, dtype=torch.long, device=device),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader


def compute_metrics(
    tn: float, fp: float, fn: float, tp: float
) -> Tuple[float, float, float]:
    """
    Compute Accuracy, Recall and F1Score from the confusion matrix.

    :param tn: True Negatives
    :type tn: float
    :param fp: False Positives
    :type fp: float
    :param fn: False Negatives
    :type fn: float
    :param tp: True Positives
    :type tp: float
    :return: Tuple containing Accuracy, Recall and F1Score.
    :rtype: Tuple[float, float, float]

    """

    acc = (tp + tn) / (tp + tn + fp + fn)
    rec = tp / (tp + fn)
    f1 = (2 * tp) / (2 * tp + fp + fn)

    return acc, rec, f1


def dataframe_to_numpy_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convert a DataFrame with embeddings and interaction types into numpy arrays for X and y.

    :param df: DataFrame containing the dataset with columns `bacterium_embedding`, `phage_embedding`, and `interaction_type`.
    :type df: pd.DataFrame
    :return: Tuple containing the features (X) and labels (y).
    :rtype: tuple[DataFrame, Series[Any]]
    """

    y = df["interaction_type"]
    X = df[["bacterium_embedding", "phage_embedding"]]

    X = X.apply(
        lambda x: np.concatenate(
            [
                x["bacterium_embedding"].cpu().numpy(),
                x["phage_embedding"].cpu().numpy(),
            ],
            axis=None,
        ),
        axis=1,
        result_type="expand",
    )
    return X, y


def train_model(
    train_df: pd.DataFrame,
    model: nn.Module | SklearnClassifier,
    training_config: TrainingConfig,
    device: str,
    val_df: pd.DataFrame | None = None,
    verbose: int = 2,
    progressbar_description: str = "",
    pos_weight: float = 1.0,
) -> np.ndarray:
    """
    Train a model (either a PyTorch nn.Module or a SklearnClassifier) using the provided training DataFrame. Optionally, a validation DataFrame can be provided for evaluation during training.

    :param train_df: DataFrame containing the training dataset with columns `bacterium_embedding`, `phage_embedding`, and `interaction_type`.
    :type train_df: pd.DataFrame
    :param model: Model to train, either a PyTorch nn.Module or a SklearnClassifier.
    :type model: nn.Module | SklearnClassifier
    :param training_config: Training configuration parameters.
    :type training_config: TrainingConfig
    :param device: Device to use for training (e.g., "cpu" or "cuda:0").
    :type device: str
    :param val_df: Optional DataFrame containing the validation dataset for evaluation during training.
    :type val_df: pd.DataFrame | None
    :param verbose: Verbosity level for logging.
    :type verbose: int
    :param progressbar_description: Description for the progress bar during training.
    :type progressbar_description: str
    :return: Confusion matrix after training. Format: [[tn, fp], [fn, tp]]
    :rtype: np.ndarray
    """

    if isinstance(model, nn.Module):
        return train_nn_model(
            train_df,
            model,
            training_config,
            device,
            val_df,
            verbose,
            progressbar_description,
            pos_weight,
        )

    else:
        return _train_sklearn_model(
            train_df,
            model,
            training_config,
            device,
            val_df,
            verbose,
            progressbar_description,
        )


def _train_sklearn_model(
    train_df: pd.DataFrame,
    model: SklearnClassifier,
    training_config: TrainingConfig,
    device: str,
    val_df: pd.DataFrame | None = None,
    verbose: int = 2,
    progressbar_description: str = "",
) -> np.ndarray:
    """
    Train a SklearnClassifier model using the provided training DataFrame. Optionally, a validation DataFrame can be provided for evaluation during training.

    See :func:`train_model` for parameter descriptions.
    """

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
        logger.info(f"Accuracy (train): {acc}")
        logger.info(f"Recall (train): {rec}")
        logger.info(f"F1 score (train): {f1}")
        logger.info(f"Confusion Matrix (train) (TP, FP, FN, TN): {tp, fp, fn, tn}")

    return cm


def train_nn_model(
    train_df: pd.DataFrame,
    model: nn.Module,
    training_config: TrainingConfig,
    device: str,
    val_df: pd.DataFrame | None = None,
    verbose: int = 2,
    progressbar_description: str = "",
    pos_weight: float = 1.0,
) -> np.ndarray:
    """
    Train a PyTorch nn.Module model using the provided training DataFrame. Optionally, a validation DataFrame can be provided for evaluation during training.

    See :func:`train_model` for parameter descriptions.
    """

    dataloader = dataframe_to_tf_dataloader(
        train_df, batch_size=training_config.batch_size, device=device
    )

    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    # criterion = nn.CrossEntropyLoss()
    class_weights = torch.tensor([1.0, float(pos_weight)], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=(
            "min" if training_config.monitor_metric_reduce_lr == "loss" else "max"
        ),  # 'max' for F1/accuracy, 'min' for loss
        factor=training_config.multiplying_factor_reduce_lr,
        patience=training_config.patience_reduce_lr,
    )

    # metrics
    accuracy = tm.Accuracy(task="binary").to(device)
    recall = tm.Recall(task="binary").to(device)
    f1 = tm.F1Score(task="binary").to(device)
    cm = tm.ConfusionMatrix(task="binary").to(device)

    if verbose >= 2:
        logger.info(f"Starting training for {training_config.epochs} epochs...")

    # Early stopping
    best_metric = (
        -np.inf if training_config.monitor_metric_early_stopping != "loss" else np.inf
    )
    epochs_no_improve = 0
    best_model_state = None

    # Typical torch training loop
    with tqdm(
        range(training_config.epochs),
        unit="epoch",
        desc=progressbar_description,
        disable=verbose < 1,
    ) as tepochs:
        for epoch in tepochs:
            # Train 1 epoch
            train_loss = 0.0
            for bact_emb, phg_emb, labels in dataloader:  # each [batch, emb_dim]
                optimizer.zero_grad()
                if model.training and training_config.training_noise_std != 0:
                    # Create noise tensors with the same shape as embeddings
                    bact_noise = (
                        torch.randn_like(bact_emb) * training_config.training_noise_std
                    )
                    phg_noise = (
                        torch.randn_like(phg_emb) * training_config.training_noise_std
                    )

                    # Add noise to the inputs
                    bact_emb_noisy = bact_emb + bact_noise
                    phg_emb_noisy = phg_emb + phg_noise
                else:
                    bact_emb_noisy = bact_emb
                    phg_emb_noisy = phg_emb

                logits = model(bact_emb_noisy, phg_emb_noisy)

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
                val_cm, val_loss = test_model(
                    val_df, model, training_config.batch_size, device, silent=True
                )
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
                    raise ValueError(
                        f"Unknown metric {training_config.monitor_metric_early_stopping}. monitor_metric_early_stopping must be 'f1' or 'loss'"
                    )

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

                tepochs.set_postfix(
                    OrderedDict(
                        lr=optimizer.param_groups[0]["lr"],
                        loss=val_loss,
                        accuracy=100.0 * val_acc,
                        recall=100.0 * val_rec,
                        f1=100.0 * val_f1,
                    )
                )
            else:
                tepochs.set_postfix(
                    OrderedDict(
                        loss=loss.item(),
                        accuracy=100.0 * acc.item(),
                        recall=100.0 * rec.item(),
                        f1=100.0 * f1s.item(),
                    )
                )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    cm_mat = cm.compute().cpu().numpy()  # (torchmetrics default: TN, FP, FN, TP)
    tn, fp, fn, tp = cm_mat.ravel().tolist()

    if verbose >= 2:
        logger.info(f"Finished training")
        logger.info(f"Accuracy (train): {accuracy.compute()}")
        logger.info(f"Recall (train): {recall.compute()}")
        logger.info(f"F1 score (train): {f1.compute()}")
        logger.info(f"Loss (train): {loss}")
        logger.info(f"Confusion Matrix (train) (TP, FP, FN, TN): {tp, fp, fn, tn}")

    return cm_mat


def test_model(
    test_df: pd.DataFrame,
    model: nn.Module | SklearnClassifier,
    batch_size: int,
    device: str,
    silent: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Test a model (either a PyTorch nn.Module or a SklearnClassifier) using the provided test DataFrame.

    :param test_df: DataFrame containing the test dataset with columns `bacterium_embedding`, `phage_embedding`, and `interaction_type`.
    :type test_df: pd.DataFrame
    :param model: Model to test, either a PyTorch nn.Module or a SklearnClassifier.
    :type model: nn.Module | SklearnClassifier
    :param batch_size: Batch size for testing.
    :type batch_size: int
    :param device: Device to use for testing (e.g., "cpu" or "cuda:0").
    :type device: str
    :param silent: Whether to suppress logging output during testing.
    :type silent: bool
    :return: Tuple containing the confusion matrix and test loss (if applicable). Format of confusion matrix: [[tn, fp], [fn, tp]]
    :rtype: tuple[np.ndarray, float]
    """

    if isinstance(model, nn.Module):
        return test_nn_model(test_df, model, batch_size, device, silent)

    else:
        return test_sklearn_model(test_df, model, batch_size, device, silent)


def test_sklearn_model(
    test_df: pd.DataFrame,
    model: SklearnClassifier,
    batch_size: int,
    device: str,
    silent: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Test a SklearnClassifier model using the provided test DataFrame.

    See :func:`test_model` for parameter descriptions.
    """

    if not silent:
        logger.info(f"Starting testing...")

    X_test, y_test = dataframe_to_numpy_X_y(test_df)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel().tolist()
    acc, rec, f1 = compute_metrics(tn, fp, fn, tp)

    if not silent:
        logger.info(f"Accuracy (test): {acc}")
        logger.info(f"Recall (test): {rec}")
        logger.info(f"F1 score (test): {f1}")
        logger.info(f"Confusion Matrix (test) (TP, FP, FN, TN): {tp, fp, fn, tn}")

    return cm, -1


def test_nn_model(
    test_df: pd.DataFrame,
    model: nn.Module,
    batch_size: int,
    device: str,
    silent: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Test a PyTorch nn.Module model using the provided test DataFrame.

    See :func:`test_model` for parameter descriptions.
    """

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
            loss = criterion(logits, labels)
            test_loss += loss.item() * bact_emb.size(0)

            predictions = logits.argmax(dim=1, keepdim=True).squeeze()

            if not silent:
                accuracy(predictions, labels)
                f1(predictions, labels)
                recall(predictions, labels)
            cm(predictions, labels)

    cm_mat = cm.compute().cpu().numpy()  # (torchmetrics default: TN, FP, FN, TP)
    test_loss = test_loss / len(dataloader.dataset)  # type: ignore

    tn, fp, fn, tp = cm_mat.ravel().tolist()

    if not silent:
        logger.info(f"Accuracy (test): {accuracy.compute()}")
        logger.info(f"Recall (test): {recall.compute()}")
        logger.info(f"F1 score (test): {f1.compute()}")
        logger.info(f"Loss (test): {test_loss}")
        logger.info(f"Confusion Matrix (test) (TP, FP, FN, TN): {tp, fp, fn, tn}")

    return cm_mat, test_loss


def kfold_train(
    df: pd.DataFrame,
    model: nn.Module | SklearnClassifier,
    training_config: TrainingConfig,
    device: str,
):
    """
    Perform K-Fold Cross Validation training on the provided DataFrame using the specified model and training configuration.

    :param df: DataFrame containing the dataset with columns `bacterium_embedding`, `phage_embedding`, and `interaction_type`.
    :type df: pd.DataFrame
    :param model: Model to train, either a PyTorch nn.Module or a SklearnClassifier.
    :type model: nn.Module | SklearnClassifier
    :param training_config: Training configuration parameters.
    :type training_config: TrainingConfig
    :param device: Device to use for training (e.g., "cpu" or "cuda:0").
    :type device: str
    :param use_multiple_gpu: Whether to use multiple GPUs for training (if available).
    :type use_multiple_gpu: bool
    """

    if training_config.stratify_cv:
        sgkf = StratifiedGroupKFold(
            n_splits=training_config.k_folds_cv
        )  # To make sure that the validation dataset contains only unknown phages
        groups = df["phage_id"].values
        y = df["interaction_type"].values
        splits = sgkf.split(df, y=y, groups=groups)

    else:
        kfold = KFold(
            n_splits=training_config.k_folds_cv, shuffle=True, random_state=42
        )
        splits = kfold.split(df)

    all_conf_matrices = []

    logger.info(f"Starting {training_config.k_folds_cv}-Fold Cross Validation...")

    for fold, (train_idx, val_idx) in enumerate(splits):
        logger.debug(f"Starting fold {fold + 1}...")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        num_neg_train = (train_df["interaction_type"] == 0).sum()
        num_pos_train = (train_df["interaction_type"] == 1).sum()
        # num_neg_val = (val_df["interaction_type"] == 0).sum()
        # num_pos_val = (val_df["interaction_type"] == 1).sum()
        curr_pos_weight = num_neg_train / num_pos_train if num_pos_train > 0 else 1.0

        # logger.info(f"--- Fold {fold + 1} Stats ---")
        # logger.info(f"Train: {num_neg_train} Neg, {num_pos_train} Pos. Weight applied: {curr_pos_weight:.2f}")
        # logger.info(f"Val:   {num_neg_val} Neg, {num_pos_val} Pos.")

        model.reset_model(device)

        train_model(
            train_df=train_df,
            model=model,
            training_config=training_config,
            device=device,
            val_df=val_df,
            verbose=1,
            progressbar_description=f"Fold {fold + 1}/{training_config.k_folds_cv}",
            pos_weight=curr_pos_weight,
        )

        # Evaluate model on validation fold
        cm_mat, _ = test_model(
            test_df=val_df,
            model=model,
            batch_size=training_config.batch_size,
            device=device,
            silent=True,
        )

        all_conf_matrices.append(cm_mat)

    mean_cm: np.ndarray = sum(all_conf_matrices) / len(all_conf_matrices)
    # tp, fp, fn, tn = mean_cm[0][0], mean_cm[0][1], mean_cm[1][0], mean_cm[1][1]
    tn, fp, fn, tp = mean_cm.ravel().tolist()
    acc, rec, f1 = compute_metrics(tn, fp, fn, tp)

    logger.info(f"Finished Cross Validation training")
    logger.info(f"Accuracy (CV): {acc}")
    logger.info(f"Recall (CV): {rec}")
    logger.info(f"F1 score (CV): {f1}")
    logger.info(
        f"Confusion Matrix (CV) (TP, FP, FN, TN): ({tp:.2f}, {fp:.2f}, {fn:.2f}, {tn:.2f})"
    )

    return mean_cm


if __name__ == "__main__":

    logger.info(f"Running: {' '.join(sys.argv)}")

    # Parse command line args
    parser = argparse.ArgumentParser(description="PBI Pipeline CLI")
    parser.add_argument(
        "--config",
        "-c",
        required=False,
        default=None,
        type=str,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--json-cli",
        "-j",
        required=False,
        default=None,
        type=str,
        help="Alternative way to set config. Recieves the entire json as an argument.",
    )
    cli_args = parser.parse_args()
    assert (
        cli_args.config is not None or cli_args.json_cli is not None
    ), "A configuration is required. Use --help to see the details."

    config = parse_config(cli_args.config, json_cli=cli_args.json_cli)

    # Set number of threads for PyTorch
    if config.torch_num_threads > 0:
        logger.debug(f"Setting PyTorch number of threads to {config.torch_num_threads}")
        torch.set_num_threads(config.torch_num_threads)

    # Load input data
    if config.input_perphect is not None:
        bacteria_df, phages_df, couples_df = PerphectDataInput(
            input_paths=config.input_perphect
        ).load()

    device = config.device
    logger.debug(f"Running on device: {device}")

    output_manager = H5pyEmbeddingsManager(config.embeddings_dir)

    create_embeddings_bacteria(
        config.bacteria_embedding_models,
        config.compute_bacteria_embeddings,
        bacteria_df,
        output_manager,
    )
    create_embeddings_phages(
        config.phages_embedding_models,
        config.compute_phages_embeddings,
        phages_df,
        output_manager,
    )

    # Create datasets, train and test classifier
    if config.training_config.do_train:
        # Create train and test datasets
        bacteria_model_names = [x.name() for x in config.bacteria_embedding_models]
        phages_model_names = [x.name() for x in config.phages_embedding_models]

        dataset = make_dataset(
            couples_df, bacteria_model_names, phages_model_names, output_manager, device
        )
        dataset = reduce_dimensionality(
            dataset,
            config.training_config.reduce_dimensionality,
            config.output_dir,
            config.training_config.n_components_bacteria,
            config.training_config.n_components_phages,
        )

        if config.training_config.do_test:
            train, test = train_test_split(
                dataset, test_size=0.2, random_state=42, shuffle=True
            )
        else:
            train = dataset

        # Hardcode the path to the test dataset file
        # TEST_DATASET_PATH = "data/perphect-data/predphi/predphi_test_dataset.csv"
        # logger.info(f"Using hardcoded test set from: {TEST_DATASET_PATH}")

        # test_couples_df = pd.read_csv(TEST_DATASET_PATH)

        # test = make_dataset(
        #     test_couples_df, bacteria_model_names, phages_model_names, output_manager, device
        # )

        # test = reduce_dimensionality(
        #     test,
        #     config.training_config.reduce_dimensionality,
        #     config.output_dir,
        #     config.training_config.n_components_bacteria,
        #     config.training_config.n_components_phages,
        # )

        # train = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        # test = test.sample(frac=1, random_state=42).reset_index(drop=True)

        # Only for testing purposes
        # dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        # Select 250 random phages for testing
        # test_phag_ids = dataset["phage_id"].drop_duplicates().sample(n=250, random_state=25).to_list()
        # train = dataset[~dataset["phage_id"].isin(test_phag_ids)].reset_index(drop=True)
        # test = dataset[dataset["phage_id"].isin(test_phag_ids)].reset_index(drop=True)

        # Select 50 random bacteria for testing
        # test_bact_ids = dataset["bacterium_id"].drop_duplicates().sample(n=25, random_state=25).to_list()
        # train = dataset[~dataset["bacterium_id"].isin(test_bact_ids)].reset_index(drop=True)
        # test = dataset[dataset["bacterium_id"].isin(test_bact_ids)].reset_index(drop=True)

        logger.info(f"Train dataset size: {len(train)}")
        logger.info(
            f"Test dataset size: {len(test) if config.training_config.do_test else 'N/A'}"
        )

        stats = Stats(config)

        # Instantiate classifier
        bacterium_embed_size = len(train["bacterium_embedding"].iloc[0])
        phage_embed_size = len(train["phage_embedding"].iloc[0])
        model = config.classifier(
            bacterium_embed_size, phage_embed_size, **config.classifier_params
        )

        stats.update_classifier(model)

        t = time.perf_counter()

        if config.training_config.k_folds_cv <= 1:
            cm = train_model(
                train,
                model,
                training_config=config.training_config,
                device=device,
                verbose=1,
                progressbar_description="Training...",
            )
        else:
            cm = kfold_train(
                train,
                model,
                training_config=config.training_config,
                device=device,
            )
        train_time = time.perf_counter() - t

        stats.update_train_results(cm, train_time)

        if config.training_config.do_test:
            t = time.perf_counter()
            cm, _ = test_model(
                test, model, batch_size=config.training_config.batch_size, device=device
            )
            test_time = time.perf_counter() - t

            stats.update_test_results(cm, test_time)

        # Save the trained model
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
            model_path = os.path.join(config.output_dir, "trained_model.pth")
            if isinstance(model, nn.Module):
                torch.save(model.state_dict(), model_path)
                logger.info(f"Trained model saved to: {model_path}")
            else:
                # TODO: SklearnClassifier save method
                logger.warning(
                    "Saving sklearn models currently not supported. Not saving the model."
                )

            # Save stats and the config file used for training
            training_config_path = os.path.join(
                config.output_dir, "training_config.yaml"
            )
            with open(training_config_path, "w") as f:
                yaml.dump(config.raw_dict, f)
            logger.info(f"Training config saved to: {training_config_path}")

            stats_path = os.path.join(config.output_dir, "stats.tsv")
            with open(stats_path, "w") as f:
                # Headers
                f.write(
                    "Date\tCommand\tDescription\tBacteria embedder\tPhages embedder\tClassifier\tEpochs\tBS\tLR\tTrain Elapsed time (s)\tTrain True Positive\tTrain False Positive\tTrain False Negative\tTrain True Negative\tTrain Accuracy\tTrain Weighted accuracy\tTrain F1 Score\tTest Elapsed time (s)\tTest True Positive\tTest False Positive\tTest False Negative\tTest True Negative\tTest Accuracy\tTest Weighted accuracy\tTest F1 Score"
                )
                stats.log(lambda msg: f.write(msg + "\n"))
            logger.info(f"Run stats saved to: {stats_path}")
