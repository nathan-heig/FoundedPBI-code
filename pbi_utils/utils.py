import argparse
import json
import sys
import numpy as np
import torch
import gc
import torchmetrics as tm
import datetime
import torch.nn as nn


def clean_gpu():
    torch.cuda.empty_cache()
    gc.collect()

class Stats:
    def __init__(self, args) -> None:
        self.args = args
        self.test_cm = None
        self.train_cm = None
        self.classifier = None
        self.train_time = None
        self.test_time = None

    def update_test_results(self, cm: np.ndarray, test_time):
        self.test_cm = cm
        self.test_time = test_time

    def update_train_results(self, cm: np.ndarray, train_time):
        self.train_cm = cm
        self.train_time = train_time
    
    def update_classifier(self, classifier: nn.Module):
        self.classifier = classifier

    def log(self, logger):
        # Print all the stats in the format required by excel

        bacteria_model_names = [str(x) for x in self.args.bacteria_embedding_models]
        phages_model_names = [str(x) for x in self.args.phages_embedding_models]

        data = [
            datetime.datetime.now().strftime("%d/%m/%YT%H:%M:%S"),
            # " ".join(sys.argv),
            f"main.py -j '{json.dumps(self.args.raw_dict)}'",
            "",
            ", ".join(bacteria_model_names),
            ", ".join(phages_model_names),
            self.classifier.name() if self.classifier is not None else "",
            f"{self.args.training_config.epochs}",
            f"{self.args.training_config.batch_size}",
            f"{self.args.training_config.learning_rate:.4e}".replace(".",","),
            f"{self.train_time:.2f}".replace(".",",") if self.train_time is not None else "",
            f"{self.train_cm[0][0]:.2f}".replace(".",",") if self.train_cm is not None else "",
            f"{self.train_cm[0][1]:.2f}".replace(".",",") if self.train_cm is not None else "",
            f"{self.train_cm[1][0]:.2f}".replace(".",",") if self.train_cm is not None else "",
            f"{self.train_cm[1][1]:.2f}".replace(".",",") if self.train_cm is not None else "",
            "",
            "",
            "",
            f"{self.test_time:.2f}".replace(".",",") if self.test_time is not None else "",
            f"{self.test_cm[0][0]:.2f}".replace(".",",") if self.test_cm is not None else "",
            f"{self.test_cm[0][1]:.2f}".replace(".",",") if self.test_cm is not None else "",
            f"{self.test_cm[1][0]:.2f}".replace(".",",") if self.test_cm is not None else "",
            f"{self.test_cm[1][1]:.2f}".replace(".",",") if self.test_cm is not None else "",
        ]
        logger("\n"+"\t".join(data))