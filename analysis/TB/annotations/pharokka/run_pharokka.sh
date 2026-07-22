#!/usr/bin/env bash
# Annotation fonctionnelle des phages predphi avec pharokka, sur la FENETRE vue par NT2.
# Reproduit tout le pipeline : FASTA tronque -> pharokka -> matrice de labels.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PBI=/home/nathan/anaconda3/envs/pbi/bin/python
DB="$HERE/pharokka_db"
source /home/nathan/anaconda3/etc/profile.d/conda.sh

# --- Installation (une seule fois ; decommenter si l'env/DB n'existent pas) -------------
# conda create -y -n pharokka -c conda-forge -c bioconda pharokka
# conda run -n pharokka install_databases.py -o "$DB"     # ~1.8 Go

# --- 1) FASTA des phages tronque a la fenetre NT2 (12276 bp) ----------------------------
"$PBI" "$HERE/build_phage_fasta.py"

# --- 2) Annotation pharokka, mode meta (1 phage = 1 contig), 24 threads -----------------
#   prodigal-gv (gene calling) + MMseqs2 sur PHROG / VFDB / CARD + tRNAscan (defaut meta).
conda run -n pharokka pharokka.py \
    -i "$HERE/phages_nt2trunc.fasta" \
    -o "$HERE/out_nt2trunc" \
    -d "$DB" -t 24 -m -f

# --- 3) Matrice de labels biologiques [phage x feature] pour le SAE ---------------------
"$PBI" "$HERE/parse_pharokka.py"

echo "Termine -> $HERE/phage_function_matrix.csv"
