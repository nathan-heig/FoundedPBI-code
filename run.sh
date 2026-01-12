#!/usr/bin/env bash
# To be executed as `. run.sh` (or `source run.sh`)

micromamba activate -n pbi

python main.py -c model_configs/base.yaml
