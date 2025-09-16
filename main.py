import pandas as pd
from tqdm import tqdm
from pbi_utils.data_manager import H5pyEmbeddingsManager, PerphectDataInput, EmbeddingsManager
from pbi_utils.logging import Logging, INFO
from pbi_models.megaDNA import MegaDNA
from typing import Tuple

tqdm.pandas() # Initialize tqdm with pandas
Logging.set_logging_level(INFO)

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

if __name__ == "__main__":
    bacteria_df, phages_df, couples_df = PerphectDataInput(base_path="data/perphect-data/dummy").load()

    output_manager = H5pyEmbeddingsManager("./data/embeddings/megaDNA.h5")

    # create_embeddings(bacteria_df, phages_df, output_manager)
    bacteria_encoded, phages_encoded = load_embeddings(bacteria_df, phages_df, output_manager)

    
