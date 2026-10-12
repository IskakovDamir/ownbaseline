# W6 direction check: what it adds to the registered plan

Written 2026-10-12, after every registered value was known. Nothing here changes a
registered verdict. The format follows Willroth and Atherton (2024, Advances in
Methods and Practices in Psychological Science 7(1), doi 10.1177/25152459231213802):
one row per change, with what was registered, what was added, when, why, and what
it does to the reader's conclusions.

## Why

The registered estimator (`own_baseline/conditional_skill.py`, `align`) takes each
score's sign from the sign of its own Spearman correlation with the ordinal, then
residualizes and takes Kendall tau_b against the ordinal. Every published benchmark
of these scores that we checked fixes the sign in advance from the method's
definition and reports a reversed ordering as a failure: Gulati et al. 2020
(CytoTRACE), Kang et al. 2025 (CytoTRACE 2), Teschendorff and Enver 2017 (SCENT),
Teschendorff et al. 2021 (CCAT), Shi et al. 2020 (MCE), Noller and Cahan 2024
(StemFinder), Zhang et al. 2022 (FitDevo). Choosing the sign from the labels the
skill is then scored against is a selection on those labels, the circularity of
Kriegeskorte et al. 2009 (Nature Neuroscience 12:535). The two readings disagree
exactly where a score's marginal ordering runs against its authors' polarity.

## Changes

| item | registered | added | type | timing | effect on conclusions |
|---|---|---|---|---|---|
| sign of a score | sign of its Spearman correlation with the stage ordinal (`align`) | a second reading with the sign fixed by the authors' polarity, reported beside the first for every placed row | added sensitivity analysis; registered result stays primary | after all registered results were known | four of nine verdicts differ: signalling entropy and ORIGINS fail, CCAT and MCE clear; six of nine clear under either reading; four clear under both (CytoTRACE, NCG, dpath, StemFinder) |
| floor for the second reading | Null B, alignment on | the same grid cell with alignment off, since a sign fixed in advance is not chosen from the data | added | same | floors move by at most 0.0026; no verdict depends on the change |
| lower tail of the null | not used (one-sided rule) | the null's 2.5th percentile at each registered cell, reported for description | added, descriptive only | same | CCAT, SLICE and MCE lie below it; no verdict is drawn from it |

Not done, deliberately: re-reading the registered one-sided rule as two-sided. That
is a change of sidedness after the results were known, which Willroth and Atherton
list as a deviation to avoid.

## Polarity sources

In `direction_check.py`, `POLARITY`. NCG's authors state no direction; it is
inferred from the construction (an expression-weighted centrality sum with no sign
flip) and is marked as inferred wherever it is used.

## Output

`results/direction.json`, frozen as `data/run_record/w6/direction.json`. The script
recomputes every registered value from the score vectors and asserts equality with
`figures/fig2_values.json` and with the frozen joint values before writing.
