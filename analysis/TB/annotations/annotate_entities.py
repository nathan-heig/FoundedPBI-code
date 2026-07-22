#!/usr/bin/env python
"""Annote bacteries/phages de PredPhi via la taxonomie NCBI et flagge les contaminants.

Une vraie bacterie doit avoir une division taxonomique 'Bacteria' ; un vrai phage
'Phages' (ou 'Viruses'). Tout le reste (mitochondries, genes isoles...) est flagge.
Les IDs predphi sont des accessions NCBI -> interrogeables directement.

La partie biologie de ce script depasse ce que je connais en informatique.
Je me suis aide d'une IA (Claude, Anthropic) pour l'ecrire.
"""
import json, time, urllib.request, urllib.parse, os
import xml.etree.ElementTree as ET
import pandas as pd

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.abspath(os.path.join(HERE, "../../.."))
PRED   = os.path.join(ROOT, "data/perphect-data/predphi")
OUTDIR = HERE
BATCH  = 150
SLEEP  = 0.34  # < 3 req/s sans cle API


def esummary(db, ids):
    """esummary JSON par lots ; renvoie {uid: record}."""
    out = {}
    for i in range(0, len(ids), BATCH):
        chunk = [str(x) for x in ids[i:i + BATCH]]
        q = urllib.parse.urlencode({"db": db, "id": ",".join(chunk), "retmode": "json"})
        with urllib.request.urlopen(EUTILS + "esummary.fcgi?" + q, timeout=60) as r:
            res = json.load(r)["result"]
        for uid in res.get("uids", []):
            out[uid] = res[uid]
        time.sleep(SLEEP)
        print(f"    {db}: {min(i + BATCH, len(ids))}/{len(ids)}", end="\r")
    print()
    return out


def efetch_taxonomy(taxids):
    """taxid -> superkingdom (Bacteria/Archaea/Eukaryota/Viruses) + organisme, via la lignee."""
    out = {}
    ids = sorted(taxids)
    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i + BATCH]
        q = urllib.parse.urlencode({"db": "taxonomy", "id": ",".join(chunk), "retmode": "xml"})
        with urllib.request.urlopen(EUTILS + "efetch.fcgi?" + q, timeout=60) as r:
            root = ET.parse(r).getroot()
        for tx in root.findall("Taxon"):
            tid = tx.findtext("TaxId")
            lin = tx.findtext("Lineage") or ""
            sk = next((k for k in ("Viruses", "Bacteria", "Archaea", "Eukaryota") if k in lin), "?")
            out[tid] = {"superkingdom": sk, "organism": tx.findtext("ScientificName") or "?"}
        time.sleep(SLEEP)
        print(f"    taxonomy: {min(i + BATCH, len(ids))}/{len(ids)}", end="\r")
    print()
    return out


def annotate(accessions):
    """accession -> dict(taxid, superkingdom, organism, title)."""
    nuc = esummary("nuccore", accessions)                      # accession -> taxid + title
    acc2, taxids = {}, set()
    for rec in nuc.values():
        acc = rec.get("caption", "")
        tx  = str(rec.get("taxid", ""))
        acc2[acc] = {"taxid": tx, "title": rec.get("title", "")}
        if tx:
            taxids.add(tx)
    tx2 = efetch_taxonomy(taxids)                              # taxid -> superkingdom + nom
    rows = []
    for acc, d in acc2.items():
        t = tx2.get(d["taxid"], {})
        rows.append({"id": acc, "taxid": d["taxid"],
                     "superkingdom": t.get("superkingdom", "?"),
                     "organism": t.get("organism", "?"),
                     "title": d["title"]})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    bact  = pd.read_csv(f"{PRED}/bacteria_sequences.csv", usecols=["bacterium_id"])
    phage = pd.read_csv(f"{PRED}/bacteriophages_sequences.csv", usecols=["phage_id"])
    tr = pd.read_csv(f"{PRED}/predphi_train_dataset.csv")
    te = pd.read_csv(f"{PRED}/predphi_test_dataset.csv")

    for side, ids, valid_sk, idcol in [
        ("bacteria", bact["bacterium_id"].astype(str).unique(), {"Bacteria", "Archaea"}, "bacterium_id"),
        ("phages",   phage["phage_id"].astype(str).unique(),    {"Viruses"}, "phage_id"),
    ]:
        print(f"\n=== {side} : {len(ids)} entites ===")
        df = annotate(list(ids))
        df["valid"] = df["superkingdom"].isin(valid_sk)
        df.to_csv(f"{OUTDIR}/{side}_annotations.csv", index=False)

        bad = df[~df["valid"]]
        print(f"  contaminants (superkingdom hors {valid_sk}) : {len(bad)}/{len(df)}")
        print(df["superkingdom"].value_counts().to_string())
        if len(bad):
            print("\n  --- entites flaggees ---")
            print(bad.sort_values("superkingdom")[["id", "superkingdom", "organism"]]
                  .to_string(index=False))
            # couples impactes
            badset = set(bad["id"])
            n_tr = tr[idcol].astype(str).isin(badset).sum()
            n_te = te[idcol].astype(str).isin(badset).sum()
            print(f"\n  couples impactes : train={n_tr}  test={n_te}")


if __name__ == "__main__":
    main()
