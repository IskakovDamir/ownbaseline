# Errata to PREREG.md, written after it was locked

`PREREG.md` is locked at commit `833acba` and is not edited. An independent judge of the results, on 2026-11-22, found three places where it states something inaccurately. None changes a registered quantity.

1. **The calibration.** The plan says the synthetic calibration "was started before this file was committed and its tables are in `results/calibration_*.json`". An earlier run of it had been started and stopped before the lock; the run whose tables are reported started after the analysis finished and wrote its files between 08:42 and 08:48 on 2026-11-22. It reads the design of GSE106474 and no score, so the order does not affect it, but the sentence is wrong as written.
2. **Batches.** The code reads the sequencing batch from the first three characters after the stage in the sample name, so the runs named DS2b, DS4b and DS5b are counted in batches DS2, DS4 and DS5. The plan says "DS2 to DS5" and does not state this. The 28 units are the 28 sample names of the accession; whether a run marked b is a separate pool of embryos or a second run of the same pool was not established.
3. **The adversarial review.** The plan describes changes made after a review before the lock. The repository holds one version of the plan, so the earlier draft and the reviewer's report cannot be checked from it. They exist only in the transcript of the session that produced them.

Also noted by the judge: the first results commit (`92dd551`) says REPORT.md lists T1 and T2 as failed. At that commit REPORT.md covered the first plan only; T1 and T2 were marked failed in `results/depth_2.json`. `report.py` now appends the predictions of the later plans and the calibration tables.
