#!/usr/bin/env bash
# Run each combination 5 times and record results + average

micromamba activate -n pbi

strategies=("TruncateStrategy" "AverageStrategy" "MaxStrategy" "TfidfStrategy" "Tf4idfStrategy" "TKPertStrategy")

nt2phagestrats=("MaxStrategy" "TfidfStrategy" "TKPertStrategy")
nt2bactstrats=("MaxStrategy" "TfidfStrategy" "TKPertStrategy")

megadnaphagestrats=("TruncateStrategy" "MaxStrategy" "Tf4idfStrategy" "TKPertStrategy")
megadnabactstrats=("TruncateStrategy" "MaxStrategy" "Tf4idfStrategy" "TKPertStrategy")

dnabertphagestrats=("TruncateStrategy" "MaxStrategy" "TKPertStrategy")
dnabertbactstrats=("TruncateStrategy" "MaxStrategy" "TKPertStrategy")

repeats=3

mkdir -p ./tmp/gridsearch_results
csv_file=./gridsearch_results.csv
echo "NT2PhageStrategy,NT2BactStrategy,MegaDNAPhageStrategy,MegaDNABactStrategy,DNABERTPhageStrategy,DNABERTBactStrategy,F1Score1,F1Score2,F1Score3,F1Score4,F1Score5,Average" > "$csv_file"

# Progress tracking
progress_file=.progress
lock_file=.progress.lock
echo 0 > "$progress_file"

run_one() {
    local nt2p="$1"
    local nt2b="$2"
    local megadnap="$3"
    local megadnab="$4"
    local dnabertp="$5"
    local dnabertb="$6"
    local total="$7"

    local scores=()
    
    for ((i=1;i<=repeats;i++)); do
        local LOG="./tmp/gridsearch_results/log_${nt2p}_${nt2b}_${megadnap}_${megadnab}_${dnabertp}_${dnabertb}_run${i}.txt"
        
        NT2PHAGESTRAT=$nt2p MEGADNAPHAGESTRAT=$megadnap DNABERTPHAGESTRAT=$dnabertp NT2BACTSTRAT=$nt2b MEGADNABACTSTRAT=$megadnab DNABERTBACTSTRAT=$dnabertb python main.py -c model_configs/all_env.yaml &>"$LOG" #2>/dev/null
        
        local SCORE
        SCORE=$(tail -n 6 "$LOG" | grep -F "F1 score (CV): " | awk '{print $NF}')
        scores+=("$SCORE")
    done

    # Compute average
    local sum=0
    for s in "${scores[@]}"; do sum=$(echo "$sum + $s" | bc); done
    local avg=$(echo "scale=4; $sum / $repeats" | bc)

    # File-lock-safe write
    {
        flock -x 200
        echo "${nt2p},${nt2b},${megadnap},${megadnab},${dnabertp},${dnabertb},${scores[0]},${scores[1]},${scores[2]},${scores[3]},${scores[4]},${avg}" >> "$csv_file"
    } 200>>"$csv_file"

    # Update progress
    {
        flock -x 300
        local count
        count=$(($(<"$progress_file") + 1))
        echo "$count" > "$progress_file"
        local percent=$((100 * count / total))
        printf "\rProgress: %d/%d (%d%%)" "$count" "$total" "$percent" >&2
    } 300>"$lock_file"
}

export -f run_one
export csv_file progress_file lock_file repeats

# Count total combinations
total=$(
for nt2p in "${nt2phagestrats[@]}"; do
  for nt2b in "${nt2bactstrats[@]}"; do
    for megadnap in "${megadnaphagestrats[@]}"; do
      for megadnab in "${megadnabactstrats[@]}"; do
        for dnabertp in "${dnabertphagestrats[@]}"; do
          for dnabertb in "${dnabertbactstrats[@]}"; do
            echo 1
          done
        done
      done
    done
  done
done | wc -l
)
echo "Total combinations: $total"

# Generate jobs and run in parallel
for nt2p in "${nt2phagestrats[@]}"; do
  for nt2b in "${nt2bactstrats[@]}"; do
    for megadnap in "${megadnaphagestrats[@]}"; do
      for megadnab in "${megadnabactstrats[@]}"; do
        for dnabertp in "${dnabertphagestrats[@]}"; do
          for dnabertb in "${dnabertbactstrats[@]}"; do
            echo "$nt2p $nt2b $megadnap $megadnab $dnabertp $dnabertb $total"
          done
        done
      done
    done
  done
done | stdbuf -oL xargs -n 7 -P 80 bash -c 'run_one "$@"' _
