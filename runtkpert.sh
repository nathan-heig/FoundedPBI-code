
micromamba activate -n pbi

echo "==============================================================================================================="
echo "=========================================NT2_sentence_tf4idf overlap==========================================="
echo "==============================================================================================================="


python main.py \
    --input-perphect data/perphect-data/public_data_set \
    --embeddings-dir data/embeddings-NT50-overlap \
    --phages-embedding-model NT2_sentence_tf4idf nucleotide-transformer-v2-50m-multi-species 6 3 \
    --bacteria-embedding-model NT2_sentence_tf4idf nucleotide-transformer-v2-50m-multi-species 6 3 \
    --num-gpu 1 \
    --gpu-id 2

python main.py \
    --input-perphect data/perphect-data/private_data_set \
    --embeddings-dir data/embeddings-NT50-overlap \
    --phages-embedding-model NT2_sentence_tf4idf nucleotide-transformer-v2-50m-multi-species 6 3 \
    --bacteria-embedding-model NT2_sentence_tf4idf nucleotide-transformer-v2-50m-multi-species 6 3 \
    --num-gpu 1 \
    --gpu-id 2

python main.py \
    --input-perphect data/perphect-data/all-private-undersampled \
    --embeddings-dir data/embeddings-NT50-overlap \
    --use-cached-embeddings \
    --phages-embedding-model NT2_sentence_tf4idf nucleotide-transformer-v2-50m-multi-species 6 3 \
    --bacteria-embedding-model NT2_sentence_tf4idf nucleotide-transformer-v2-50m-multi-species 6 3 \
    --num-gpu 1 \
    --gpu-id 2

# echo "==============================================================================================================="
# echo "=========================================NT2_sentence_TKPERT avg==============================================="
# echo "==============================================================================================================="


# python main.py \
#     --input-perphect data/perphect-data/public_data_set \
#     --embeddings-dir data/embeddings-NT50-avg \
#     --phages-embedding-model NT2_sentence_TKPERT nucleotide-transformer-v2-50m-multi-species 16 20 avg \
#     --bacteria-embedding-model NT2_sentence_TKPERT nucleotide-transformer-v2-50m-multi-species 16 20 avg \
#     --num-gpu 1 \
#     --gpu-id 2

# python main.py \
#     --input-perphect data/perphect-data/private_data_set \
#     --embeddings-dir data/embeddings-NT50-avg \
#     --phages-embedding-model NT2_sentence_TKPERT nucleotide-transformer-v2-50m-multi-species 16 20 avg \
#     --bacteria-embedding-model NT2_sentence_TKPERT nucleotide-transformer-v2-50m-multi-species 16 20 avg \
#     --num-gpu 1 \
#     --gpu-id 2

# python main.py \
#     --input-perphect data/perphect-data/all-private-undersampled \
#     --embeddings-dir data/embeddings-NT50-avg \
#     --use-cached-embeddings \
#     --phages-embedding-model NT2_sentence_TKPERT nucleotide-transformer-v2-50m-multi-species 16 20 avg \
#     --bacteria-embedding-model NT2_sentence_TKPERT nucleotide-transformer-v2-50m-multi-species 16 20 avg \
#     --num-gpu 1 \
#     --gpu-id 2


# echo "==============================================================================================================="
# echo "=========================================NT2_sentence_TKPERT concat============================================"
# echo "==============================================================================================================="


# python main.py \
#     --input-perphect data/perphect-data/public_data_set \
#     --embeddings-dir data/embeddings-NT50-concat \
#     --phages-embedding-model NT2_sentence_TKPERT nucleotide-transformer-v2-50m-multi-species 16 20 concat \
#     --bacteria-embedding-model NT2_sentence_TKPERT nucleotide-transformer-v2-50m-multi-species 16 20 concat \
#     --num-gpu 1 \
#     --gpu-id 2

# python main.py \
#     --input-perphect data/perphect-data/private_data_set \
#     --embeddings-dir data/embeddings-NT50-concat \
#     --phages-embedding-model NT2_sentence_TKPERT nucleotide-transformer-v2-50m-multi-species 16 20 concat \
#     --bacteria-embedding-model NT2_sentence_TKPERT nucleotide-transformer-v2-50m-multi-species 16 20 concat \
#     --num-gpu 1 \
#     --gpu-id 2

# python main.py \
#     --input-perphect data/perphect-data/all-private-undersampled \
#     --embeddings-dir data/embeddings-NT50-concat \
#     --use-cached-embeddings \
#     --phages-embedding-model NT2_sentence_TKPERT nucleotide-transformer-v2-50m-multi-species 16 20 concat \
#     --bacteria-embedding-model NT2_sentence_TKPERT nucleotide-transformer-v2-50m-multi-species 16 20 concat \
#     --num-gpu 1 \
#     --gpu-id 2