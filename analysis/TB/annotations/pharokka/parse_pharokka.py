#!/usr/bin/env python
"""Transforme la sortie pharokka en matrice de LABELS BIOLOGIQUES [phage x feature],
prete a corraler aux latents d'un sparse autoencoder entraine sur les embeddings NT2.

Entree  : out_nt2trunc/  (sortie pharokka, mode meta)
Sortie  : phage_function_matrix.csv  (1 ligne par phage)
"""
import os, re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = f"{HERE}/out_nt2trunc"

# Un label fonctionnel n'est garde que s'il apparait dans >= MIN_PHAGES phages.
# Seuil bas volontairement : la caracterisation des latents se fait en F1 avec seuil
# choisi sur le train et mesure sur le test (protocole Decode-gLM), ce qui elimine
# tout seul les annotations trop rares (elles ne se repliquent pas sur le test).
MIN_PHAGES = 3


def sanit(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")

# Categories fonctionnelles PHROG (le coeur du contenu genique).
# NB : "other" et "unknown function" sont volontairement exclus -> ce sont des
# fourre-tout ("gene classe nulle part ailleurs" / "fonction inconnue"), pas des
# concepts biologiques : un latent qui les matche n'apprend rien d'interpretable.
PHROG_CATS = ["head and packaging", "connector", "tail", "lysis",
              "integration and excision", "transcription regulation",
              "DNA, RNA and nucleotide metabolism",
              "moron, auxiliary metabolic gene and host takeover"]


def main():
    # 1) Comptes par categorie PHROG (format long -> matrice large)
    cf = pd.read_csv(f"{OUT}/pharokka_cds_functions.tsv", sep="\t")
    mat = cf.pivot_table(index="contig", columns="Description", values="Count",
                         fill_value=0).rename_axis(None, axis=1)
    keep = [c for c in (["CDS"] + PHROG_CATS +
                        ["tRNAs", "CRISPRs", "VFDB_Virulence_Factors", "CARD_AMR_Genes"])
            if c in mat.columns]
    out = mat[keep].copy()

    # 2) Labels FINS : presence (0/1) de chaque fonction nommee et de chaque famille PHROG
    #    presentes dans >= MIN_PHAGES phages -> vocabulaire riche pour nommer les latents SAE.
    gg = pd.read_csv(f"{OUT}/pharokka_cds_final_merged_output.tsv", sep="\t",
                     usecols=["contig", "phrog", "annot"])

    ann = gg.assign(annot=gg["annot"].fillna("").str.lower())
    ann = ann[~ann["annot"].isin(["", "hypothetical protein"])]
    keep = ann.groupby("annot")["contig"].nunique()
    keep = keep[keep >= MIN_PHAGES].index
    fmat = (ann[ann["annot"].isin(keep)].assign(v=1)
            .pivot_table(index="contig", columns="annot", values="v", aggfunc="max", fill_value=0))
    fmat.columns = [f"func_{sanit(c)}" for c in fmat.columns]

    ph = gg.assign(phrog=gg["phrog"].astype(str))
    ph = ph[~ph["phrog"].isin(["No_PHROG", "nan", ""])]
    keep_p = ph.groupby("phrog")["contig"].nunique()
    keep_p = keep_p[keep_p >= MIN_PHAGES].index
    pmat = (ph[ph["phrog"].isin(keep_p)].assign(v=1)
            .pivot_table(index="contig", columns="phrog", values="v", aggfunc="max", fill_value=0))
    pmat.columns = [f"phrog_{c}" for c in pmat.columns]

    n_coarse = out.shape[1]
    out = out.join(fmat, how="left").join(pmat, how="left").fillna(0)
    out = out.astype(int)          # tout en entier : comptes ou presence 0/1, rien de continu

    out.index.name = "phage_id"
    out.to_csv(f"{HERE}/phage_function_matrix.csv")

    n_func = len([c for c in out.columns if c.startswith("func_")])
    n_phrog = len([c for c in out.columns if c.startswith("phrog_")])
    print(f"{len(out)} phages x {out.shape[1]} labels -> phage_function_matrix.csv")
    print(f"  dont {n_coarse} labels grossiers + {n_func} fonctions nommees + "
          f"{n_phrog} familles PHROG (>= {MIN_PHAGES} phages)\n")
    print("Couverture (phages avec >=1 gene de la categorie) :")
    for c in PHROG_CATS:
        if c in out.columns:
            print(f"  {c:55s} : {(out[c] > 0).mean()*100:5.1f}%")
    cats = [c for c in PHROG_CATS if c in out.columns]
    print(f"  phages sans aucun gene de categorie nommee : "
          f"{(out[cats].sum(axis=1) == 0).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
