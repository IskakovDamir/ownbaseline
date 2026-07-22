# own-baseline: does a single-cell potency score add anything beyond a low-order statistic?

Unsupervised single-cell potency/stemness scores are usually validated (or dismissed)
by their pairwise correlation with a primitive or a ground truth. That is the wrong
test. A score can correlate 0.96 with node degree and still add real ordering skill;
another can be a 0.998 identity that adds nothing. This repo is the control that
tells them apart, plus the code for the field audit in the paper.

## The test

For a score `s`, a declared low-order primitive `p` (one of: gene count `|supp(x)|`;
`PCC(x, degree)` on a PPI; Shannon entropy `H(x/Σx)`), and a **score-independent**
potency ordinal `g` (e.g. developmental stage assigned by microscopy before
dissociation):

    conditional skill = weighted-tau( rank(s) residualized on rank(p) , g )

If `s` is a monotone function of `p`, this is exactly 0. If it is > 0 (both weighting
kernels, bootstrap CI excluding 0), `s` orders cells by potency beyond `p` — it is not
that primitive. A high pairwise `corr(s, p)` does not bound this either way.

## Install

    pip install -r requirements.txt        # numpy, scipy, scikit-learn for the core
    # then use own_baseline/ as a package (it is import-clean on its own)

## Usage

    import anndata as ad
    from own_baseline import run_own_baseline

    adata = ad.read_h5ad("cells.h5ad")                 # .X = raw counts
    res = run_own_baseline(adata, score="cytotrace_v1", # or "scent_sr"/"ccat"/your callable
                           gt_ordinal="stage")          # a score-independent potency ordinal
    print(res["delta"], res["takeaway"])                # delta ~ 0 => reduces to gene counts

`own_baseline.py` is also runnable as a CLI / self-test (`python own_baseline/own_baseline.py`).
The full both-kernel, bootstrapped conditional skill used for the paper figures is in
`own_baseline/conditional_skill.py`.

## What the audit found

On a clean 12-stage zebrafish developmental ordinal (GSE106474, 39,505 cells), running
each score's real implementation: pairwise correlation and conditional skill are
decoupled, and the field is a spectrum, not uniformly trivial.

- **Reduce to their primitive** (conditional skill ~ 0): CCAT (= `PCC(x, degree)` by
  definition, ρ=0.998), SLICE (≈ raw Shannon, ρ=0.93), StemID, cmEntropy.
- **Add beyond it** (both kernels, surviving a library-size control): full CytoTRACE
  (its diffusion step), SCENT signalling entropy (degree-correlation plus a
  transcriptome-breadth term), ORIGINS, dpath. A joint residualization on all four
  low-order statistics at once still leaves a residual for these four.
- MCE and NCG rest only on their authors' pairwise-R² admissions; NCG's own-baseline
  puts it in the "adds beyond" group, so an admission does not establish triviality.

The strong claim that every unsupervised potency score is a reparametrization of a
low-order statistic is refuted by this test — it holds for the identity/entropy
reductions but not for the entropy-rate or diffusion-smoothed scores at single-cell
resolution. Isolating what the surviving residual actually is remains open.

## Layout

    own_baseline/        the released diagnostic (import-clean)
      own_baseline.py    tool + CLI (Test-A error-model and potency-score Delta)
      conditional_skill.py  sign-agnostic conditional skill, both kernels, bootstrap CI
      potency_metrics.py    primitives (gene count, PCC(x,degree), Shannon) + SR/CCAT/CytoTRACE reimpls
      atlas_run/wdm_tau.py   Kang wdm weighted-Kendall kernel (== R wdm::wdm)
      synthetic.py           synthetic-data generators for the self-tests
    scores/              score implementations used in the audit
      cytotrace_full.py  R-faithful CytoTRACE v1 port (MVG -> GCS -> NNLS -> Markov diffusion)
      run_scent_validate.R / run_origins_validate.R / run_slice.R / run_dpath.R / run_ncg.R
                         wrappers calling the real R packages (see below)
    experiments/         every run script, verbatim, organized as it was run:
      track2/ (zebrafish decider) track4/ (field audit + joint-primitive)
      fix3/   (Xenopus replicate) atlas_run/ w4/ w1_gate2/ ct2_probe/ track1/ stemsc/
    figures/make_paper_a_figures.py
    data/README.md       GEO accessions + STRING v12 + how to fetch (no data stored here)

## Scores (R packages)

The audit ran published implementations, not reimplementations, wherever one exists:
SCENT (`aet21/SCENT`, SR + CCAT), SLICE (`xu-lab/SLICE`), dpath (`gongx030/dpath`),
ORIGINS (`danielasenraoka/ORIGINS`), NCG (`Xinzhe-Ni/NCG`). CytoTRACE v1 is a port of
the Gulati 2020 R algorithm (`scores/cytotrace_full.py`); SR/CCAT/ORIGINS Python
implementations were checked against the R packages to machine precision on a
subsample before use at scale. Install the R packages from their GitHub repos.

## Reproduce

Each dataset's pipeline lives under `experiments/`; the scripts carry hardcoded paths
to the research vault they were run in (set the `VAULT`/`PREP` constant at the top to
your own working directory). The zebrafish decider is `experiments/track2/code/`
(`prep_data.py -> run_cytotrace.py / run_scent_full.py / run_origins_full.py ->
conditional_skill.py`); the field audit and joint-primitive test are in
`experiments/track4/code/` (`fix2_joint_primitive.py`, `fix1_ncg_skill.py`); the Xenopus
replicate is `experiments/fix3/code/`. See `data/README.md` for the inputs.

## License

MIT (`LICENSE`).
