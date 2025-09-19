# To be executed as `. run.sh` (or `source run.sh`)

micromamba activate -n pbi

python main.py \
    --input-perphect data/perphect-data/dummy \
    --embeddings-dir data/embeddings-dummy \
    --phages-embedding-model MegaDNA data/weights/megaDNA_phage_145M.pt \
    --bacteria-embedding-model MegaDNA data/weights/megaDNA_phage_145M.pt \
