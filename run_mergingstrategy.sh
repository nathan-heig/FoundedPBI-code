#!/usr/bin/env bash
# To be executed as `. run.sh` (or `source run.sh`)

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <MergingStrategy> <GPUID> [UseCachedEmbeddings](default: true)"
    echo "Example: $0 BottomTruncateStrategy 0 false"
    return 1 2>/dev/null || exit 1
fi

MERGINGSTRATEGY="$1"
GPUID="$2"
USECACHEDEMBEDDINGS="${3:-true}"

# MERGINGSTRATEGY="BottomTruncateStrategy"
# GPUID="1"
# USECACHEDEMBEDDINGS="true"

LOGS_FOLDER="logs/$MERGINGSTRATEGY"
mkdir -p $LOGS_FOLDER

echo "Exploring strategy $MERGINGSTRATEGY, GPU $GPUID, use cached embeddings: $USECACHEDEMBEDDINGS"

echo "Running Bacteria NT2"
micromamba activate -n pbi
STRATEGY="$MERGINGSTRATEGY" CACHEEMBEDDINGS="$USECACHEDEMBEDDINGS" GPUID="$GPUID" python main.py -c model_configs/tmp/bact_nt2.yaml &> $LOGS_FOLDER/bact_nt2.log
SCORE1=$(tail -n 6 $LOGS_FOLDER/bact_nt2.log | grep -F "F1 score (CV): " | awk '{printf "%.2f", $NF*100}' || echo "NaN")
echo "Running Bacteria DNABERT2"
micromamba activate -n pbi-dnabert
STRATEGY="$MERGINGSTRATEGY" CACHEEMBEDDINGS="$USECACHEDEMBEDDINGS" GPUID="$GPUID" python main.py -c model_configs/tmp/bact_dnabert2.yaml &> $LOGS_FOLDER/bact_dnabert2.log
SCORE2=$(tail -n 6 $LOGS_FOLDER/bact_dnabert2.log | grep -F "F1 score (CV): " | awk '{printf "%.2f", $NF*100}' || echo "NaN")
echo "Running Bacteria MegaDNA"
micromamba activate -n pbi
STRATEGY="$MERGINGSTRATEGY" CACHEEMBEDDINGS="$USECACHEDEMBEDDINGS" GPUID="$GPUID" python main.py -c model_configs/tmp/bact_megadna.yaml &> $LOGS_FOLDER/bact_megadna.log
SCORE3=$(tail -n 6 $LOGS_FOLDER/bact_megadna.log | grep -F "F1 score (CV): " | awk '{printf "%.2f", $NF*100}' || echo "NaN")

echo "Running Phages NT2"
micromamba activate -n pbi
STRATEGY="$MERGINGSTRATEGY" CACHEEMBEDDINGS="$USECACHEDEMBEDDINGS" GPUID="$GPUID" python main.py -c model_configs/tmp/phages_nt2.yaml &> $LOGS_FOLDER/phages_nt2.log
SCORE4=$(tail -n 6 $LOGS_FOLDER/phages_nt2.log | grep -F "F1 score (CV): " | awk '{printf "%.2f", $NF*100}' || echo "NaN")
echo "Running Phages DNABERT2"
micromamba activate -n pbi-dnabert
STRATEGY="$MERGINGSTRATEGY" CACHEEMBEDDINGS="$USECACHEDEMBEDDINGS" GPUID="$GPUID" python main.py -c model_configs/tmp/phages_dnabert2.yaml &> $LOGS_FOLDER/phages_dnabert2.log
SCORE5=$(tail -n 6 $LOGS_FOLDER/phages_dnabert2.log | grep -F "F1 score (CV): " | awk '{printf "%.2f", $NF*100}' || echo "NaN")
echo "Running Phages MegaDNA"
micromamba activate -n pbi
STRATEGY="$MERGINGSTRATEGY" CACHEEMBEDDINGS="$USECACHEDEMBEDDINGS" GPUID="$GPUID" python main.py -c model_configs/tmp/phages_megadna.yaml &> $LOGS_FOLDER/phages_megadna.log
SCORE6=$(tail -n 6 $LOGS_FOLDER/phages_megadna.log | grep -F "F1 score (CV): " | awk '{printf "%.2f", $NF*100}' || echo "NaN")

echo "      BACTERIA                   PHAGES            "
echo "NT2   DNABERT2   MegaDNA | NT2   DNABERT2   MegaDNA"
echo "$SCORE1\\% & $SCORE2\\% & $SCORE3\\% & $SCORE4\\% & $SCORE5\\% & $SCORE6\\%"