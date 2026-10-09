#!/usr/bin/env python3
"""
discovery_facts.py — recompute, from the files fetch_sources.py placed, every
number DISCOVERY.md cites about the weight vectors, gene lists and gene
overlaps. Reads gene NAMES of the four audit datasets and no cell label.

    python3 experiments/w5_new_rows/discovery_facts.py

Writes experiments/w5_new_rows/discovery/discovery_facts.json.
"""
from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "own_baseline" / "paths.py").is_file())
sys.path.insert(0, str(REPO))
from own_baseline.paths import data_root  # noqa: E402

SRC = data_root() / "w5_new_rows" / "src"
OUT = HERE / "discovery" / "discovery_facts.json"

R_DUMP = r"""
bgw <- readRDS(file.path(src, "FitDevo", "BGW.rds"))
load(file.path(src, "TCGAbiolinks", "data", "SC_PCBC_stemSig.rda"))
w <- SC_PCBC_stemSig
e <- new.env()
load(file.path(src, "stemfinder", "data", "s_genes_human.rda"), envir = e)
load(file.path(src, "stemfinder", "data", "g2m_genes_human.rda"), envir = e)
objs <- ls(e)
cat(jsonlite_like <- paste0(
  "{\"bgw_n\":", length(bgw), ",\"bgw_ones\":", sum(bgw == 1),
  ",\"bgw_zeros\":", sum(bgw == 0),
  ",\"w_n\":", length(w), ",\"w_pos\":", sum(w > 0), ",\"w_neg\":", sum(w < 0),
  ",\"w_zero\":", sum(w == 0), ",\"w_dup\":", sum(duplicated(names(w))),
  ",\"cc_objects\":[", paste0("\"", objs, "\"", collapse = ","), "]",
  ",\"cc_lengths\":[", paste(sapply(objs, function(o) length(get(o, envir = e))), collapse = ","), "]",
  "}\n"))
writeLines(names(bgw), file.path(tmp, "bgw_genes.txt"))
writeLines(names(w), file.path(tmp, "w_genes.txt"))
writeLines(unique(unlist(mget(objs, envir = e))), file.path(tmp, "cc_genes.txt"))
"""


def gene_names():
    """Gene symbols of each audit dataset, as the score code paths see them."""
    out = {}
    out["GSE106474"] = (data_root() / "track2/data/prepared/genes.txt").read_text().split("\n")
    c1 = data_root() / "w4/data/c1_gse117498"
    if c1.is_dir():
        genes = set()
        for f in sorted(c1.glob("GSM*.raw_counts.tsv.gz")):
            with gzip.open(f, "rt") as fh:
                fh.readline()
                for line in fh:
                    g = line.split("\t", 1)[0]
                    if g != "Library":
                        genes.add(g)
        out["GSE117498"] = sorted(genes)
    c2 = data_root() / "w4/data/c2_gse125970/GSE125970_raw_UMIcounts.txt.gz"
    if c2.is_file():
        with gzip.open(c2, "rt") as fh:
            fh.readline()
            out["GSE125970"] = [line.split("\t", 1)[0] for line in fh]
    br = data_root() / "fix3/data/prepared/genes.txt"
    if br.is_file():
        out["GSE113074"] = br.read_text().split("\n")
    return {k: [g for g in v if g] for k, v in out.items()}


def main():
    tmp = data_root() / "w5_new_rows" / "facts_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["Rscript", "--vanilla", "-e",
                        f'src <- "{SRC}"; tmp <- "{tmp}";' + R_DUMP],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(r.stderr)
    facts = json.loads(r.stdout.strip().splitlines()[-1])
    bgw = (tmp / "bgw_genes.txt").read_text().split()
    w = (tmp / "w_genes.txt").read_text().split()
    cc = (tmp / "cc_genes.txt").read_text().split()
    facts["cc_unique_genes"] = len(cc)

    overlap = {}
    for ds, genes in gene_names().items():
        upper = {g.upper() for g in genes}
        exact = set(genes)
        overlap[ds] = {
            "n_genes": len(genes),
            "FitDevo_BGW_uppercase_match": len(set(bgw) & upper),
            "mRNAsi_weights_exact_match": len(set(w) & exact),
            "stemFinder_human_cc_exact_match": len(set(cc) & exact),
            "stemFinder_human_cc_uppercase_match": len({c.upper() for c in cc} & upper),
        }
    facts["overlap"] = overlap
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(facts, indent=2) + "\n")
    print(json.dumps(facts, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
