#!/usr/bin/env bash

micromamba activate -n pbi

# ===============================================
# Settings
# ===============================================
repeats=1
max_jobs=55
csv_file=./gridsearch_merging_strategy_final.csv
# Only -1 is working for now...
random_samples=-1 # -1 For gridsearch
config_file=./model_configs/all_env.yaml


# ===============================================
# Setup
# ===============================================
mkdir -p ./tmp/gridsearch_results
progress_file=.progress
lock_file=.progress.lock
echo 0 > "$progress_file"

# ===============================================
# Parameter Grid
# ===============================================
# declare -A param_grid=(
#   [NT2BACTSTRAT]="TruncateStrategy BottomTruncateStrategy TopBottomTruncateStrategy MaxStrategy TKPertStrategy"
#   [MEGADNABACTSTRAT]="TruncateStrategy BottomTruncateStrategy TopBottomTruncateStrategy MaxStrategy TKPertStrategy"
#   [DNABERTBACTSTRAT]="TruncateStrategy BottomTruncateStrategy TopBottomTruncateStrategy MaxStrategy TKPertStrategy"
#   [NT2PHAGESTRAT]="TruncateStrategy BottomTruncateStrategy TopBottomTruncateStrategy MaxStrategy TKPertStrategy"
#   [MEGADNAPHAGESTRAT]="TruncateStrategy BottomTruncateStrategy TopBottomTruncateStrategy MaxStrategy TKPertStrategy"
#   [DNABERTPHAGESTRAT]="TruncateStrategy BottomTruncateStrategy TopBottomTruncateStrategy MaxStrategy TKPertStrategy"
# )

#   [CLASSIFIER]='
# {"name":"SklearnClassifier","params":{"sklearn_model_name":"LGBMClassifier","sklearn_model_params":{"n_estimators":350,"num_leaves":63,"n_jobs":1}}}
# '
#   [CLASSIFIER]='
# {"name":"SklearnClassifier","params":{"sklearn_model_name":"RandomForestClassifier","sklearn_model_params":{"n_estimators":350,"n_jobs":1}}}
# {"name":"MLPClassifier","params":{"bacterium_mlp_sizes":[128,64],"phage_mlp_sizes":[128,64],"dense_dim":64,"dropout":0.2}}
# {"name":"SklearnClassifier","params":{"sklearn_model_name":"XGBClassifier","sklearn_model_params":{"n_estimators":200,"tree_method":"hist","n_jobs":1}}}
# '

declare -A param_grid=(
    [LR]="1e-2 1e-3 1e-4"
    [WD]="0 1e-5 1e-4 1e-3"
    [DROPOUT]="0 0.1 0.2 0.3 0.4"
)

# ===============================================
# Generate combinations
# ===============================================

# Echo but in stderr
errcho() { echo "$@" 1>&2; }

generate_combinations() {
    local keys=("$@")
    local n=${#keys[@]}
    # errcho "keys: ${keys[@]}"
    _generate_recursive() {
        local index=$1
        local currcombo=${@:2}
        # errcho "Iteration $index"
        # errcho "Combo: ${currcombo[@]}"
        if (( index == n )); then
            # errcho "Finishing and printing ${currcombo[@]}"
            printf "%s\n" "${currcombo[@]}"
            return
        fi
        local key="${keys[index]}"
        while IFS= read -r val; do
          [[ -z "$val" ]] && continue
          currcombo[index]="$key=$val"
          # errcho "Creating new combo and calling next: ${currcombo[@]}"
          _generate_recursive $((index + 1)) ${currcombo[@]}
        done <<< "$(echo "${param_grid[$key]}" | tr ' ' '\n')"
    }
    _generate_recursive 0 ""
}

generate_random_combinations() {
    local keys=("$@")
    for ((r=0; r<random_samples; r++)); do
        local combo=()
        for key in "${keys[@]}"; do
            # Convert values to array
            IFS=' ' read -r -a vals <<< "${param_grid[$key]}"
            # Pick one random value
            local val="${vals[$RANDOM % ${#vals[@]}]}"
            combo+=("$key=$val")
        done
        printf "%s\n" "${combo[@]}"
    done
}

# ===============================================
# Helpers
# ===============================================
make_label() {
    # echo "$1" | tr '\n' '_' | tr -d '[:space:]' | tr -cd '[:alnum:]_'

    local vars="$@"
    echo "$vars" | tr ' ' '_' | tr -d '[:space:]' | tr -cd '[:alnum:]_'

}

is_done() {
    local vars="$@"

    # # Check logs
    # local all_logs=true
    # for ((i=1; i<=repeats; i++)); do
    #     [[ -f ./tmp/gridsearch_results/log${label}_run${i}.txt ]] || { all_logs=false; break; }
    # done
    # $all_logs && return 0

    # Check CSV (already logged results)
    if [[ -f "$csv_file" ]]; then
        local vars_csv
        vars_csv=$(echo "${vars[*]}" | tr ' ' ',')
        # errcho "Vars csv: $vars_csv"
        if grep -qF "$vars_csv" "$csv_file"; then
            return 0
        fi
    fi
    return 1
}

# ===============================================
# Main
# ===============================================
run_one() {
    local total="$1"; shift
    local decoded=$@

    local vars=()

    # Export env vars
    for env in $decoded; do
      # errcho "Env var: $env"
      IFS='=' read -r k v <<< $env
    #   errcho "exporting $k = $v"
      export "$k"="$v"
      vars+=($v)
    done

    # errcho "${vars[@]}"

    # while IFS='=' read -r envVar; do
    # errcho "envVar: $envVar"
    # IFS='=' read -r k v <<< $envVar
    # errcho "exporting $k = $v"
    #     export "$k"="$v"
    # done <<< $decoded

    local label
    label=$(make_label ${vars[@]})

    # Skip if done
    if is_done ${vars[@]}; then
        return 0
    fi

    local scores=()
    for ((i=1; i<=repeats; i++)); do
        local LOG="./tmp/gridsearch_results/log${label}_run${i}.txt"
        echo "[$(date '+%H:%M:%S')] Running ${label} (repeat $i)..."
        # python main.py -c model_configs/all_env.yaml &>"$LOG"
        python main.py -c "$config_file" &>"$LOG"
        echo "[$(date '+%H:%M:%S')] Finished running ${label} (repeat $i)"
        local SCORE
        SCORE=$(tail -n 6 "$LOG" | grep -F "F1 score (CV): " | awk '{print $NF}' || echo "NaN")
        scores+=("$SCORE")
    done

    # Compute average
    local valid_scores=()
    local sum=0
    for s in "${scores[@]}"; do
        [[ "$s" =~ ^[0-9]+(\.[0-9]+)?$ ]] || continue
        valid_scores+=("$s")
        sum=$(echo "$sum + $s" | bc)
    done
    local avg
    if (( ${#valid_scores[@]} > 0 )); then
        avg=$(echo "scale=4; $sum / ${#valid_scores[@]}" | bc)
    else
        avg="NaN"
    fi

    # Safe CSV write
    {
        flock -x 200
        local vars_csv
        vars_csv=$(echo "${vars[@]}" | tr ' ' ',')
        echo "${vars_csv},$(IFS=,; echo "${scores[*]}"),${avg}" >> "$csv_file"
    } 200>>"$csv_file"

    # Progress update
    {
        flock -x 300
        local count=$(( $(<"$progress_file") + 1 ))
        echo "$count" > "$progress_file"
        local percent=$((100 * count / total))
        printf "\rProgress: %d/%d (%d%%)" "$count" "$total" "$percent" >&2
    } 300>"$lock_file"
}

export -f run_one make_label is_done
export csv_file progress_file lock_file repeats

# ===============================================
# Generate header and combinations
# ===============================================
keys=("${!param_grid[@]}")

# generate_combinations "${keys[@]}"

# If random_samples is -1, do full gridsearch
if (( random_samples == -1 )); then
    echo "Generating full gridsearch combinations..."
    mapfile -t combos < <(generate_combinations "${keys[@]}")
else
    echo "Generating $random_samples random combinations..."
    mapfile -t combos < <(generate_random_combinations "${keys[@]}")
fi

# mapfile -t combos < <(generate_combinations "${keys[@]}")
# mapfile -t combos < <(generate_random_combinations "${keys[@]}")

total=${#combos[@]}
echo "Total combinations: $total"

# Add header if CSV missing
if [[ ! -s "$csv_file" ]]; then
    {
      printf "%s," "${keys[@]}"
      for ((i=1; i<=repeats; i++)); do printf "F1Score%d," "$i"; done
      echo "Average"
    } > "$csv_file"
fi

# ===============================================
# Run combinations in parallel
# ===============================================

job_count=0
# echo "Combos: ${combos[@]}"
for combo in "${combos[@]}"; do
  # echo Combination: $combo
  run_one "$total" $combo &
  ((job_count++))
  if (( job_count >= max_jobs )); then
    wait -n
    ((job_count--))
  fi
done
wait
echo
echo "Done! Results in $csv_file"