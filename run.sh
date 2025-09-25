# To be executed as `. run.sh` (or `source run.sh`)

micromamba activate -n pbi

# python main.py \
#     --input-perphect data/perphect-data/dummy \
#     --embeddings-dir data/embeddings-dummy \
#     --phages-embedding-model MegaDNA data/weights/megaDNA_phage_145M.pt \
#     --bacteria-embedding-model MegaDNA data/weights/megaDNA_phage_145M.pt \

# python main.py \
#     --input-perphect data/perphect-data/all \
#     --embeddings-dir data/embeddings \
#     --use-cached-embeddings \
#     --phages-embedding-model NT2 \
#     --bacteria-embedding-model NT2 \
#     --phages-embedding-model MegaDNA data/weights/megaDNA_phage_145M.pt \
#     --bacteria-embedding-model MegaDNA data/weights/megaDNA_phage_145M.pt \

echo "==============================================================================================================="
echo "=========================================DNABERT Public dataset================================================"
echo "==============================================================================================================="

python main.py \
    --input-perphect data/perphect-data/public_data_set \
    --embeddings-dir data/embeddings \
    --phages-embedding-model DNABERT2 foundation-models-test/DNABERT-2/DNABERT-2-117M/ \
    --bacteria-embedding-model DNABERT2 foundation-models-test/DNABERT-2/DNABERT-2-117M/ \
    --num-gpu 1 \
    --gpu-id 3

echo "==============================================================================================================="
echo "===========================================DNABERT Private dataset============================================="
echo "==============================================================================================================="

python main.py \
    --input-perphect data/perphect-data/private_data_set \
    --embeddings-dir data/embeddings \
    --phages-embedding-model DNABERT2 foundation-models-test/DNABERT-2/DNABERT-2-117M/ \
    --bacteria-embedding-model DNABERT2 foundation-models-test/DNABERT-2/DNABERT-2-117M/ \
    --num-gpu 1 \
    --gpu-id 3

echo "==============================================================================================================="
echo "=========================================EVO Public dataset================================================"
echo "==============================================================================================================="

python main.py \
    --input-perphect data/perphect-data/public_data_set \
    --embeddings-dir data/embeddings \
    --phages-embedding-model EVO evo-1-131k-base \
    --bacteria-embedding-model EVO evo-1-131k-base \
    --num-gpu 1 \
    --gpu-id 3

echo "==============================================================================================================="
echo "===========================================EVO Private dataset============================================="
echo "==============================================================================================================="

python main.py \
    --input-perphect data/perphect-data/private_data_set \
    --embeddings-dir data/embeddings \
    --phages-embedding-model EVO evo-1-131k-base \
    --bacteria-embedding-model EVO evo-1-131k-base \
    --num-gpu 1 \
    --gpu-id 3

# echo "==============================================================================================================="
# echo "=====================================DNABERT all==============================================================="
# echo "==============================================================================================================="

# python main.py \
#     --input-perphect data/perphect-data/all \
#     --embeddings-dir data/embeddings \
#     --use-cached-embeddings \
#     --phages-embedding-model DNABERT2 foundation-models-test/DNABERT-2/DNABERT-2-117M/ \
#     --bacteria-embedding-model DNABERT2 foundation-models-test/DNABERT-2/DNABERT-2-117M/ \
#     --num-gpu 1 \
#     --gpu-id 3