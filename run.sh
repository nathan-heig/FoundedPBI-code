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
echo "=========================================NT 100 Public dataset================================================="
echo "==============================================================================================================="

python main.py \
    --input-perphect data/perphect-data/public_data_set \
    --embeddings-dir data/embeddings-NT100 \
    --phages-embedding-model NT2 nucleotide-transformer-v2-100m-multi-species \
    --bacteria-embedding-model NT2 nucleotide-transformer-v2-100m-multi-species \
    --num-gpu 1 \
    --gpu-id 0

echo "==============================================================================================================="
echo "============================================NT 100 Private dataset============================================="
echo "==============================================================================================================="

python main.py \
    --input-perphect data/perphect-data/private_data_set \
    --embeddings-dir data/embeddings-NT100 \
    --phages-embedding-model NT2 nucleotide-transformer-v2-100m-multi-species \
    --bacteria-embedding-model NT2 nucleotide-transformer-v2-100m-multi-species \
    --num-gpu 1 \
    --gpu-id 0

echo "==============================================================================================================="
echo "======================================NT 100 all==============================================================="
echo "==============================================================================================================="

python main.py \
    --input-perphect data/perphect-data/all \
    --embeddings-dir data/embeddings-NT100 \
    --use-cached-embeddings \
    --phages-embedding-model NT2 nucleotide-transformer-v2-100m-multi-species \
    --bacteria-embedding-model NT2 nucleotide-transformer-v2-100m-multi-species \
    --num-gpu 1 \
    --gpu-id 0

echo "==============================================================================================================="
echo "=========================================NT 50 Public dataset================================================="
echo "==============================================================================================================="

python main.py \
    --input-perphect data/perphect-data/public_data_set \
    --embeddings-dir data/embeddings-NT50 \
    --phages-embedding-model NT2 nucleotide-transformer-v2-50m-multi-species \
    --bacteria-embedding-model NT2 nucleotide-transformer-v2-50m-multi-species \
    --num-gpu 1 \
    --gpu-id 0

echo "==============================================================================================================="
echo "============================================NT 50 Private dataset============================================="
echo "==============================================================================================================="

python main.py \
    --input-perphect data/perphect-data/private_data_set \
    --embeddings-dir data/embeddings-NT50 \
    --phages-embedding-model NT2 nucleotide-transformer-v2-50m-multi-species \
    --bacteria-embedding-model NT2 nucleotide-transformer-v2-50m-multi-species \
    --num-gpu 1 \
    --gpu-id 0

echo "==============================================================================================================="
echo "======================================NT 50 all==============================================================="
echo "==============================================================================================================="

python main.py \
    --input-perphect data/perphect-data/all \
    --embeddings-dir data/embeddings-NT50 \
    --use-cached-embeddings \
    --phages-embedding-model NT2 nucleotide-transformer-v2-50m-multi-species \
    --bacteria-embedding-model NT2 nucleotide-transformer-v2-50m-multi-species \
    --num-gpu 1 \
    --gpu-id 0

echo "==============================================================================================================="
echo "=========================================NT 250 Public dataset================================================="
echo "==============================================================================================================="

python main.py \
    --input-perphect data/perphect-data/public_data_set \
    --embeddings-dir data/embeddings-NT250 \
    --phages-embedding-model NT2 nucleotide-transformer-v2-250m-multi-species \
    --bacteria-embedding-model NT2 nucleotide-transformer-v2-250m-multi-species \
    --num-gpu 1 \
    --gpu-id 0

echo "==============================================================================================================="
echo "============================================NT 250 Private dataset============================================="
echo "==============================================================================================================="

python main.py \
    --input-perphect data/perphect-data/private_data_set \
    --embeddings-dir data/embeddings-NT250 \
    --phages-embedding-model NT2 nucleotide-transformer-v2-250m-multi-species \
    --bacteria-embedding-model NT2 nucleotide-transformer-v2-250m-multi-species \
    --num-gpu 1 \
    --gpu-id 0

echo "==============================================================================================================="
echo "======================================NT 250 all==============================================================="
echo "==============================================================================================================="

python main.py \
    --input-perphect data/perphect-data/all \
    --embeddings-dir data/embeddings-NT250 \
    --use-cached-embeddings \
    --phages-embedding-model NT2 nucleotide-transformer-v2-250m-multi-species \
    --bacteria-embedding-model NT2 nucleotide-transformer-v2-250m-multi-species \
    --num-gpu 1 \
    --gpu-id 0

