# PBI

This repository contains all the code for the new iteration at solving the PBI problem by the CI4CB laboratory.

It includes a framework to test and verify multiple approaches at solving the problem, with a modular architecture where any part can be tuned.

![Framework architecture](doc/images/Architecture.png)

The framework consists in two branches that embed and compress the DNA sequences of the bacterium and phage, and a classifier that makes the prediction. For each of the branches, first, 3 foundation models are used to compute the embeddings of the sequence, by dividing it into multiple subsequences of maximum size (that the model allows) and embedding all the parts. Then, a strategy is used to transform this set of embeddings to one single meta-embedding, to which PCA is applied to reduce even more the dimensionality. Finally, the meta-embeddings from the two branches are given to the classifier and it predicts wether the two organisms have an interaction or not.

Every part of the architecture is configurable, by passing a YAML configuration to the execution. As an example, [model_configs/example.yaml](model_configs/example.yaml) contains all the parameters that can be used with an explanation.

## Repository structure

## Environment

Each of the foundation models requires an specific version of libraries, so it is much recommended to set up a Python package manager such as conda or micromamba. In this guide, micromamba with an alias to `mm` is used. If using conda instead, just replace `mm` with `conda`.

### Base Environment
This environment will be used to run everything that does not require a specific environment, for now, everything except the DNABERT2 model and finetuning the Nucleotide Transformer v2.

Start by creating a new environment and activating it.
```bash
mm create -n pbi
mm activate -n pbi
```
> [!IMPORTANT]
> **Make sure that you have activated correctly the environment. If not, the next commands will also work but install everything in your base environment, which might cause problems later on.**

Next, install Python 3.10.18 (higher versions might also work, but this is the one used to develop the project).
```bash
mm install "python==3.10.18"
```

To continue, this project has been developed with CUDA version 12.4. Again, it might work with higher versions, but it has not been tested. To install CUDA 12.4, run:
```bash
mm install cuda-libraries-dev cuda-nvcc cuda-nvtx cuda-cupti nccl -c nvidia/label/cuda-12.4.0
```

Finally, to install all the pip dependencies, run:
```bash
pip install -r requirements.txt
```

If everything worked correctly, congratulations, you can now start executing things.

### DNABERT2 Environment

To use DNABERT2, first, you need to download their repository.
```bash
git clone https://huggingface.co/zhihan1996/DNABERT-2-117M
```

Next, enter their folder, and in the file `flash_attn_triton.py` replace all the occurrences of `tl.dot(q, k, trans_b=True)` (or similar, all the ones that have `trans_a` or `trans_b` as a parameters) with the updated syntax of `tl.dot(q, tl.trans(k))`.
> [!IMPORTANT]
> Take your time replacing them, as in some of them it is the first parameter the one that needs to be transposed. If you make a mistake it might continue to work (or it might not) but obtain bad results.

Finally, starting from the last step in the base environment (create a new one called `pbi-dnabert` and follow the same steps), install now the latest version of triton, with:
```bash
pip install --upgrade triton
```
Pip will complain, as the `torch` version is too old to use the last version of `triton`. Just ignore it and it will work anyways.

### Finetuning Nucleotide Transformer v2

To finetune the Nucleotide Transformer v2 model, create a new environment called `pbi-finetune` and follow the same steps as in the base environment.

Finally, install the extra dependencies for finetuning:
```bash
pip install -r requirements_nt2_finetuning.txt
```

If you want to use then this finetuned model, you will need to run the main framework in this environment. 

## Execution

Once you have your environment correctly set up, the data prepared, and a YAML configuration file created, for example [model_configs/base.yaml](model_configs/base.yaml), to run the framework, execute:
```bash
python main.py -c model_configs/base.yaml
```

This will compute the embeddings for all the sequences (or use the cached ones if you already did it), train the classifier that you specified, and test it on the dataset test split. The train and test results metrics will be shown in the terminal.

> [!NOTE]
> If you are computing the embeddings from scratch with a merging strategy different than [*TruncateStrategy*, *BottomTruncateStrategy* or *TopBottomTruncateStrategy*], it will take multiple hours to finish.

## Utilities

Some bash scripts are also provided to help with specific needs.
- [run.sh](run.sh): Shows an example run test.
- [run_gridsearch.sh](run_gridsearch.sh): Performs a gridsearch over different parameters that can be customized at the start of the file. The intended use is to use environment variables inside the config file (by using `$<env_var>` as a value), and change them inside the script.
- [run_finetune_nt2.sh](run_finetune_nt2.sh): Shows an example of finetuning the Nucleotide Transformer v2.

Additionally, some jupyter notebooks are also provided, that perform different analysis on the data and models. They are available inside the `analysis/` folder.
- [data_analysis.ipynb](analysis/data_analysis.ipynb): Explores the public and private datasets, and creates one that joins them. It also plots the embeddings from DNABERT2 in a 2D grid to visualize them.
- [embeddings_merging_strategies_analysis.ipynb](analysis/embeddings_merging_strategies_analysis.ipynb): Analyzes the results from the gridsearch over merging strategies, and creates plots to compare them.
- [model_testing.ipynb](analysis/model_testing.ipynb): Explores the results from a trained model, testing it with different datasets and analyzing the predictions.
- [prepare_data_finetuning.ipynb](analysis/prepare_data_finetuning.ipynb): Prepares the data to finetune the Nucleotide Transformer v2 model.