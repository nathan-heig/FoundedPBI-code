import torch
import gc

def clean_gpu():
    torch.cuda.empty_cache()
    gc.collect()