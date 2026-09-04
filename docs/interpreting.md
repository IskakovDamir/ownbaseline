# Reading an own-baseline result

What a passing verdict means, what it does not, and the four ways a run can be
wrong that the tool cannot detect for you.

---

## What CLEARS means

The score orders cells in a way that survives removing the low-order statistic
you residualized on, and the surviving order beats the 97.5th percentile of the
estimator's own null at that value's sample size, kernel, rank correlation and
covariate count.

That is the whole claim. It is a statement about one experiment on one ordinal
with one set of covariates.

## What CLEARS does not mean

**It does not say what the residual is.** A score can order cells beyond gene
count because it tracks developmental position, or manifold structure, or a
fifth statistic nobody has named. The test separates "beyond this statistic"
from "not beyond this statistic". It says nothing about what lies beyond.

**It does not license a conservation claim.** A score anchored to a fixed
scaffold preserves whatever the scaffold preserves, across every dataset, by
construction. Clearing a floor on one atlas is not evidence that the same
structure holds on another one, and running the tool twice does not make it so.

**It does not transfer to a coarser ordinal.** The floors here were measured on
ordinals with 2, 4, 12 and 50 levels. A result on twelve levels does not carry
to a binary fate label, and the tool refuses fewer than three levels for that
reason: on binary labels the two kernels disagree on the same data.

**It is not a p-value and there is no correction applied.** The floor is a fixed
threshold read from a measured null, one row at a time. If you run seven scores
and one clears, you have run seven tests. Decide what to do about that yourself
and say what you decided.

## What DOES NOT CLEAR means, and the trap inside it

A value below its floor means the residual did not beat the estimator's own
noise at that cell. A *negative* value means something else: the residual runs
opposite to the ordinal. A negative number does not exceed a positive threshold,
so the row does not clear, but that is where the value sits rather than a test
the score failed. The tool prints the distinction and you should keep it.

## Four ways a run can be wrong that the tool cannot see

### 1. Your ordinal is downstream of your score

This is the one that ruins results, and it is why `--ordinal-source` has no
override. If the ordinal is a pseudotime, a trajectory position, a cluster
ranking or anything else computed from the same expression matrix, then the
primitive predicts the ordinal too, and a positive result measures the shared
dependence. A valid ordinal is fixed before the cells were dissociated: a staged
embryo, a sorted population, a collection timepoint, a treatment arm.

The tool takes your word for this. It cannot check it.

### 2. Your primitive is not the one your score reduces to

The test controls one realization of one statistic. If a score approximates
something you did not residualize on, it will clear, and it should not. Use the
statistic the score's own authors name, and when there is more than one
candidate, residualize on all of them at once and read the JOINT row.

The four the CLI computes are gene count, Pearson correlation between the
transcriptome and interaction-network node degree, Shannon entropy of the
expression vector, and log library size. There is no reason to believe that list
is complete.

### 3. Your scaffold is not the construct

`PCC(x, degree)` on STRING v12 at confidence 700 is one interactome at one
threshold. Part of any surviving residual may be scaffold error rather than
biology. If the degree leg carries your conclusion, vary the threshold and the
interactome and report what changed.

### 4. Your score and your primitive point in different directions

The direction check prints above every margin for this reason. When the score
and a primitive order cells in opposite directions with respect to the ordinal,
a raw margin between them is mostly the sign difference, and no gap shows this.
Conditional skill is orientation-invariant and does not have the problem, which
is why the tool reports conditional skill and not a gap.

Watch also for the case where the score itself runs against the ordinal. On a
staged timeline that is what a working potency score looks like, because later
stages are less potent. On a sorted hierarchy it means the score does not
recover the hierarchy, and every number below it is computed on the reversed
ordering.

## The floor is a property of the estimator, not of the biology

It is positive, and close to flat in n, while a bootstrap interval narrows as one
over the square root of n. At atlas scale an interval excluding zero is therefore
not evidence of anything: at n = 39,505 the tau_b floor sits three to four
interval half-widths above zero. Read the floor, not the interval.

The floors shipped here were measured on synthetic nulls under two constructions.
They model the geometry the arbiter faces; they are not that geometry. A score
can clear its own floor and still be tracking something the nulls do not
represent.

Outside the measured grid, `ownbaseline floors` refuses and names the cell that
would have to be run. Do not interpolate past it by hand.

Inside the grid but away from a design point, two things are worth reading off
the output. The floor falls monotonically with n, from 0.0741 at n = 500 to
0.0228 at n = 127,607 for tau_b at rho = 0.5, so a grid point below your n gives
a threshold that is too high and clearing it is safe, while a grid point above
your n gives one that is too low and clearing it proves less than it looks. The
tool labels which case you are in and warns on the second. The floor is not
monotone in the covariate count, so an unmeasured count takes the larger of the
two measured values bracketing it, and it is not monotone in rho either, so a
large rho offset can run in either direction and the tool says how large it is.

## Near rank identity, read the residual and not the value

When your score is a rank-preserving function of its primitive, the least-squares
fit is exact and what comes back is the rounding error of the subtraction, at
about 1e-12. That debris is monotone in the primitive rank, so where the
primitive predicts the ordinal it inherits that association and both kernels
score it: on a twelve-level staged ordinal, Kendall tau_b returns 0.71 to 0.87 in
magnitude and the weighted kernel 0.62 to 0.93, with a sign that changes with the
seed.

`ownbaseline check` refuses to print a value in that regime. What separates a real
residual from debris is magnitude: `own_baseline.cli.residual_scale` returns
`max|resid| / (eps * n)`, real residuals sit at 1e14 to 1e15 on that ratio and
debris sits at 1. If you are calling the library directly rather than the CLI,
check that ratio yourself before reading any number near rho = 1.

## The scaffold-randomization control

`--scaffold-null` permutes which gene carries which degree, which keeps the
degree distribution and breaks the correspondence between expression and
connectivity. It is worth looking at and it is not a verdict.

On the systems in the paper the gap over that null was 0.120 for signalling
entropy against 0.057 for CCAT, while both sat at a conditional skill of about
zero. A rule reading a larger gap as more genuine would have called signalling
entropy the more genuine of the two and been backwards. The entropy null itself
came in below chance at 0.437, which leaves its gap uninterpretable as a measure
of how much biology a score carries. Two pre-registered predictions about this
control were falsified on the way to that conclusion.

Read it as a description of the scaffold, and read the own baseline for the
verdict.

## Kernels

Kendall tau_b is the default and the one to report. The weighted kernel is
available and it should be read with its own floor, never against zero: its null
mean is not zero and it rises with sample size, from 0.188 at n = 500 to 0.309
at n = 127,607, and it returned a positive verdict on 720 of 720 null replicates
in our calibration. A kernel pinned at one answer cannot disagree with anything,
so a rule requiring the two kernels to agree does not do the work it looks like
it does.

## What to report

The score's conditional skill, its own floor with the cell it came from, the
direction check, the sample size, the kernel, and the rank correlation between
the score and the primitive. `ownbaseline check --json` writes all of it plus a
receipt carrying versions, seed, bootstrap count and a sha256 of every input
vector, and `ownbaseline verify --rerun` recomputes from those inputs and prints
the delta per row.

Report the own-baseline next to the score, and report the floor. The comparison
costs one line.
