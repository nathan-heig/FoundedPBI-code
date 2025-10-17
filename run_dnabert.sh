# To be executed as `. run.sh` (or `source run.sh`)

micromamba activate -n pbi

strategies=("TruncateStrategy" "AverageStrategy" "MaxStrategy" "TfidfStrategy" "Tf4idfStrategy")

for strategy in "${strategies[@]}"; do
    echo "==============================================================================================================="
    echo "========================================DNABERT2_${strategy}============================================="
    echo "==============================================================================================================="

    # TruncateStrategy, AverageStrategy, MaxStrategy, TfidfStrategy, Tf4idfStrategy
    STRATEGY=$strategy OVERLAP="0" python main.py -c model_configs/DNABERT2_env.yaml
done

echo "==============================================================================================================="
echo "========================================DNABERT2_tkpert_concat================================================="
echo "==============================================================================================================="
TKPERT_MERGING_STRATEGY="concat" OVERLAP="0" python main.py -c model_configs/DNABERT2_tkpert.yaml

echo "==============================================================================================================="
echo "========================================DNABERT2_tkpert_avg===================================================="
echo "==============================================================================================================="
TKPERT_MERGING_STRATEGY="avg" OVERLAP="0" python main.py -c model_configs/DNABERT2_tkpert.yaml