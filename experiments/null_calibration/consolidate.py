"""
Union the null-calibration grid runs into one shipped floors table.

Several cells were run more than once, in separate batches with different seed
streams. Their p97.5 estimates differ by Monte Carlo noise alone, in one case by
20 percent of the floor, and picking whichever file was read first would be
arbitrary. Where the raw per-seed values exist for every replicate of a cell, the
seeds are POOLED and the percentile is recomputed on the pooled sample. Where they
do not, the LARGEST replicate p97.5 is kept, because the floor is a decision
threshold and the smaller of two independent estimates biases toward CLEARS.
Both the number of runs and the spread across them are recorded per row.
"""
import csv, gzip, os, sys
import numpy as np

R = "experiments/null_calibration/results"
OUT = "own_baseline/data/null_floors.csv.gz"

FIELDS = ["null", "n", "levels", "rho_target", "kernel", "k_covariates",
          "align", "nullB_strength", "n_seeds", "n_runs", "mean", "sd",
          "p2.5", "p97.5", "p97.5_spread", "pooled", "source"]

GRIDS = ["null_grid.csv", "null_grid_n3000.csv", "null_grid_n39505.csv",
         "null_grid_n500_n10000.csv", "null_grid_n127607_tail.csv"]

cells = {}                      # key -> list of dicts, one per replicate run

def key_of(r):
    return (r["null"], int(r["n"]), int(r["levels"]), float(r["rho_target"]),
            r["kernel"], int(r["k_covariates"]), int(r["align"]),
            str(r["nullB_strength"]))

def add(r, src, raw):
    r["source"] = src
    r["_raw"] = raw
    cells.setdefault(key_of(r), []).append(r)

def load_raw(csv_name):
    npz = os.path.join(R, "null_grid_raw.npz" if csv_name == "null_grid.csv" else csv_name.replace(".csv", ".npz"))
    if not os.path.exists(npz):
        return None
    return np.load(npz, allow_pickle=True)

for f in GRIDS:
    p = os.path.join(R, f)
    if not os.path.exists(p):
        print(f"  missing {f}", file=sys.stderr); continue
    z = load_raw(f)
    n_rows = 0
    for r in csv.DictReader(open(p)):
        out = {k: r.get(k, "") for k in FIELDS if k in r}
        out.update({k: r.get(k, "") for k in
                    ("null", "n", "levels", "rho_target", "kernel", "align",
                     "nullB_strength", "n_seeds", "mean", "sd", "p2.5", "p97.5")})
        out["k_covariates"] = "1"          # every grid row is single-primitive
        raw = None
        cid = r.get("cell_id")
        if z is not None and cid:
            name = f"{cid}_{r['kernel']}"
            if name in z:
                raw = np.asarray(z[name], dtype=float)
        add(out, f, raw)
        n_rows += 1
    print(f"  {f:30s} {n_rows:4d} rows, raw={'yes' if z is not None else 'no'}")

# the covariate sweep: Null B, strength 0.55, 12 levels, align on. No raw file.
n_rows = 0
for r in csv.DictReader(open(os.path.join(R, "null_covariates.csv"))):
    out = {k: r.get(k, "") for k in
           ("n", "k_covariates", "rho_target", "kernel", "n_seeds", "mean",
            "sd", "p2.5", "p97.5")}
    out.update({"null": "B", "levels": "12", "align": "1", "nullB_strength": "0.55"})
    add(out, "null_covariates.csv", None)
    n_rows += 1
print(f"  {'null_covariates.csv':30s} {n_rows:4d} rows, raw=no")

rows, n_pooled, n_maxed, worst = [], 0, 0, (0.0, None)
for k, reps in sorted(cells.items()):
    nl, n, lv, rho, ker, kcov, al, st = k
    p97 = [float(r["p97.5"]) for r in reps]
    spread = max(p97) - min(p97)
    if spread > worst[0]:
        worst = (spread, k)
    base = reps[0]
    if len(reps) > 1 and all(r["_raw"] is not None for r in reps):
        v = np.concatenate([r["_raw"] for r in reps])
        v = v[np.isfinite(v)]
        chosen = float(np.percentile(v, 97.5))
        row = dict(base, **{"n_seeds": str(len(v)), "mean": repr(float(v.mean())),
                            "sd": repr(float(v.std(ddof=1))),
                            "p2.5": repr(float(np.percentile(v, 2.5))),
                            "p97.5": repr(chosen)})
        pooled = 1
        n_pooled += 1
    else:
        row = dict(reps[int(np.argmax(p97))])
        pooled = 0
        if len(reps) > 1:
            n_maxed += 1
    row.update({"null": nl, "n": n, "levels": lv, "rho_target": rho,
                "kernel": ker, "k_covariates": kcov, "align": al,
                "nullB_strength": st, "n_runs": len(reps),
                "p97.5_spread": repr(spread) if len(reps) > 1 else "0",
                "pooled": pooled,
                "source": ";".join(sorted({r["source"] for r in reps}))})
    rows.append({f: row.get(f, "") for f in FIELDS})

with gzip.open(OUT, "wt", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)

dupes = sum(1 for r in rows if int(r["n_runs"]) > 1)
print(f"\nwrote {OUT}: {len(rows)} cells, {os.path.getsize(OUT)} bytes")
print(f"  run once            {len(rows) - dupes}")
print(f"  run more than once  {dupes}  ->  pooled {n_pooled}, max-of-replicates {n_maxed}")
print(f"  largest replicate disagreement in p97.5: {worst[0]:.4f} at {worst[1]}")
print("  n     :", sorted({int(r['n']) for r in rows}))
print("  levels:", sorted({int(r['levels']) for r in rows}))
print("  rho   :", sorted({float(r['rho_target']) for r in rows}))
print("  k_cov :", sorted({int(r['k_covariates']) for r in rows}))
