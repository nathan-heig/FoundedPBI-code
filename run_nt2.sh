# To be executed as `. run.sh` (or `source run.sh`)

micromamba activate -n pbi

strategies=("TruncateStrategy" "AverageStrategy" "MaxStrategy" "TfidfStrategy" "Tf4idfStrategy")

for strategy in "${strategies[@]}"; do
    echo "==============================================================================================================="
    echo "========================================NT2_${strategy}=================================================="
    echo "==============================================================================================================="

    # TruncateStrategy, AverageStrategy, MaxStrategy, TfidfStrategy, Tf4idfStrategy
    MODEL_NAME="nucleotide-transformer-v2-50m-multi-species" STRATEGY=$strategy OVERLAP="0" python main.py -c model_configs/NT2_env.yaml
    MODEL_NAME="nucleotide-transformer-v2-100m-multi-species" STRATEGY=$strategy OVERLAP="0" python main.py -c model_configs/NT2_env.yaml
done

echo "==============================================================================================================="
echo "========================================NT2_tkpert_concat======================================================"
echo "==============================================================================================================="
MODEL_NAME="nucleotide-transformer-v2-50m-multi-species" TKPERT_MERGING_STRATEGY="concat" OVERLAP="0" python main.py -c model_configs/NT2_tkpert.yaml
MODEL_NAME="nucleotide-transformer-v2-100m-multi-species" TKPERT_MERGING_STRATEGY="concat" OVERLAP="0" python main.py -c model_configs/NT2_tkpert.yaml

echo "==============================================================================================================="
echo "========================================NT2_tkpert_avg========================================================="
echo "==============================================================================================================="
MODEL_NAME="nucleotide-transformer-v2-50m-multi-species" TKPERT_MERGING_STRATEGY="avg" OVERLAP="0" python main.py -c model_configs/NT2_tkpert.yaml
MODEL_NAME="nucleotide-transformer-v2-100m-multi-species" TKPERT_MERGING_STRATEGY="avg" OVERLAP="0" python main.py -c model_configs/NT2_tkpert.yaml