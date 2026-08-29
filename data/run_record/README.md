# Frozen run record

The run artefacts the paper's figures are drawn from, committed verbatim so a
clone can regenerate every figure without re-running a score.

Nothing here is computed by this directory. Each file is a byte-for-byte copy of
the JSON its experiment script wrote to `$OWNBASELINE_DATA_ROOT/<exp>/results/`
on the run date below. Re-running the experiment writes to the data root, not
here; this copy is the record, and `figures/make_paper_a_figures.py` names the
file and the field behind every number it draws.

| file | written by | run | what it holds |
|---|---|---|---|
| `track2/gate2_conditional_skill.json` | `experiments/track2/code/gate2_run.py` | 2026-07-20 | the fine-ordinal decider: conditional skill of each score against its declared primitive on GSE106474, 39,505 cells, 12-stage Kimmel ordinal, seed 42, both kernels, with bootstrap CIs and the score-primitive Spearman correlations |
| `track4/fix1_mce_ncg_skill.json` | `experiments/track4/code/fix1_ncg_skill.py` | 2026-07-21 | NCG against `PCC(x,degree)` on the shared seed-42 3,000-cell subsample, and MCE's `INCONCLUSIVE-infra` status |
| `w4/c1_gse117498_results.json` | `experiments/w4/scripts/w4_gate2_run.py` | 2026-08-29 | sorted human haematopoiesis (GSE117498): per-population median gene count, per-score marginal gap and conditional skill, AUROC of score and primitive, Kendall tau_b |
| `w4/c2_gse125970_results.json` | `experiments/w4/scripts/w4_gate2_run.py` | 2026-08-29 | human intestine (GSE125970), same fields |
| `stemsc/summary.json` | `experiments/stemsc/` | 2026-07-18 | StemSC on the two sorted atlases. Read only to establish that StemSC *was* measured, somewhere other than the staged ordinal, so Figure 2 can give it a row and no bar. |

sha256:

    1ad8f5eb3523939128b27bd06f03f8e762fb39089a0858ca34b64e2e2af3e852  track2/gate2_conditional_skill.json
    885bbc365204f2e5a09a14ca9014310962798138f904d284307dbbb9518896ff  track4/fix1_mce_ncg_skill.json
    1fc503e5ab8fc45ed93b4e63d4b3bf06efa841fb5dabe7cdba9ba8a333f6d6ec  w4/c1_gse117498_results.json
    d7f03f7200799e5a15ba4920590e146b2f60a794c1895fe66e76f6de14961f5f  w4/c2_gse125970_results.json
    568fa45ed0384e3ca5cc590e125b6960a17318693791e9d409e0adf5138e67c8  stemsc/summary.json

The two `w4` files were recomputed under Kendall tau_b at commit `702ed1a`; the
weighted-tau versions they replaced are not used by any figure, because every
figure value in this repository is tau_b.

`OWNBASELINE_RUN_RECORD` overrides this directory. `make_paper_a_figures.py
--crosscheck` additionally re-reads the same fields from
`$OWNBASELINE_DATA_ROOT` and prints the largest disagreement, so a local re-run
can be checked against the record instead of silently replacing it.
