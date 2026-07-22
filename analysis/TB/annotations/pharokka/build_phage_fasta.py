#!/usr/bin/env python
"""Construit le multi-FASTA des phages predphi TRONQUE a la fenetre vue par NT2
(NT2_TRUNC = 12276 bp). On annote ce que le modele voit reellement, pas le genome entier."""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../../.."))
NT2_TRUNC = 12276
SRC = f"{ROOT}/data/perphect-data/predphi/bacteriophages_sequences.csv"
OUT = f"{HERE}/phages_nt2trunc.fasta"

d = pd.read_csv(SRC)
with open(OUT, "w") as f:
    for pid, seq in zip(d["phage_id"].astype(str), d["phage_sequence"].astype(str)):
        s = seq[:NT2_TRUNC]
        f.write(f">{pid}\n")
        for i in range(0, len(s), 80):
            f.write(s[i:i + 80] + "\n")

L = d["phage_sequence"].astype(str).str.len()
print(f"{OUT} : {len(d)} phages tronques a {NT2_TRUNC} bp "
      f"({(L <= NT2_TRUNC).mean()*100:.0f}% vus entiers, mediane genome {int(L.median())} bp)")
