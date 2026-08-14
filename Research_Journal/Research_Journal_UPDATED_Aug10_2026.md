# Research Journal (Living Laboratory Notebook)

> Canonical chronological record of research progress.

# Journal Rules

Each entry should contain:

- Date
- Objective
- Methods
- Observations
- Conclusions
- Next Steps

Record:
- Meetings
- YAML changes
- H5 versions
- Training runs
- Hyperparameters
- Comet observations
- Debugging
- Research ideas

---

# Entry 001

**Date:** Monday, July 6, 2026

Objective: Initial end-to-end SALT training.

Configuration:
`/home/jjohnstu/summer_2026/run3mltoolkit/configs/GN2_bbtautau_v08_lephad_baseline.yaml`

Machine: Shahzad's CERN machine.

Outcome: Training completed successfully. Metrics and plots logged to Comet.

---

# Entry 002

**Date:** July 8, 2026

Major conceptual progress:

- Understood ID vs Anti-ID.
- Understood why fake taus cannot be identified individually.
- Understood true tau subtraction.
- Learned that the baseline YAML is not yet the fake-factor network.
- Began planning a Variable Encyclopedia.

---

# Entry 003

Current tasks from Shahzad:

- Await fake-factor H5 files.
- Read SALT evaluation documentation.
- Select checkpoint with minimum validation loss.
- Use that checkpoint's weights.
- Inspect learned p(x).
- Compare different loss choices.
- Study how different losses affect outputs.

---

# Instructions for Future ChatGPT

Maintain this as a chronological lab notebook.

Do NOT rewrite history.

Add new entries instead.

When decisions change, explain why.

Always keep links between Journal and Handbook.

The Handbook explains concepts. The Journal records what actually happened.

At major milestones:
- export PDF releases;
- keep these Markdown files as the canonical sources.

---

# Entry 004

**Date:** July 8, 2026

**Objective:** Inspect the checkpoints produced by a completed SALT training run.

**Methods:** Explored the `logs/.../ckpts/` directory after training.

**Observations:**
- One `.ckpt` file was produced per epoch.
- Each filename stores the epoch number and validation loss.
- The minimum observed validation loss was `epoch=021-val_loss=0.06494.ckpt`.

**Conclusions:**
- The preferred checkpoint for evaluation is the checkpoint with the lowest validation loss, not necessarily the final epoch.
- A checkpoint is a complete snapshot of the network and its training state.

**Next Steps:**
- Inspect the internal contents of a checkpoint.
- Trace how SALT loads checkpoints during evaluation.
- Determine where `p(x)` is saved and where `FF(x)=p(x)/(1-p(x))` is computed.

---

# Entry 005

**Date:** July 14, 2026

**Objective:** Generate ROC curves using the SALT evaluation framework and understand the evaluation pipeline.

**Methods:**
- Ran `python eval.py eval_config.yaml --out_dir ./results --eval_fold eval_all`.
- Encountered a `KeyError` indicating that the expected `..._pS` column was missing.
- Inspected `epoch=021-val_loss=0.06494__test_inclusive.h5` to list stored prediction columns.

**Observations:**
- The network outputs `pB`, `pSGGF`, `pSVBF`, and `pSGGFBSM`.
- No `pS` column exists because the model is multiclass.
- Temporarily modified `eval_config.yaml` to use `pSGGF` as the signal score to generate ROC curves and learn the evaluation framework.

**Results:**
- Successfully generated ROC curves.
- BDT ggF High: AUC = 0.971.
- Temporary GNN score: AUC = 0.965.

**Lessons Learned:**
- Evaluation configurations must match prediction names written by `salt test`.
- Inspecting the HDF5 output is an effective first step when debugging evaluation failures.
- ROC curves evaluate classifier performance across all thresholds, not just one operating point.

**Next Steps:**
- Define the proper inclusive signal score.
- Trace how `eval.py` constructs ROC curves internally.

---

# Entry 006

**Date:** July 21–24, 2026

**Objective:** Run the dedicated fake-factor GNN workflow for all folds and understand the train/calibration/application split.

**Methods:**
- Used the fake-factor configurations for model folds 0, 1, and 2.
- Ran the end-to-end `run_fold.sh` workflow.
- Inspected the produced checkpoint and HDF5 prediction files.
- Used `fakefactor_closure.py` and `fakefactor_benchmark.py` to understand the downstream fake-factor calculation.

**Fold scheme established:**
For model fold `k`:
- training fold = `k`
- calibration fold = `(k+1) % 3`
- application/test fold = `(k+2) % 3`

Thus, for model fold 0:
- train = fold 0
- calibrate = fold 1
- apply/test = fold 2

The application fold is not used to choose the global calibration scale.

**Important output interpretation:**
- Shahzad's preprocessed `fakeCR.h5` files contain the event data and event weights.
- SALT prediction H5 files contain event information plus the network outputs such as `..._pID` and `..._pantiID`.
- `pID` and `pantiID` are the network scores/probabilities associated with the ID and anti-ID classes.
- The event-level GNN fake factor is formed from the odds:
  `FF_GNN(x) = pID / (1 - pID)` after normalization of the two class outputs when necessary.

**Next Steps:**
- Compare closure for all folds.
- Compare the learned GNN fake-factor shape between independently trained folds.

---

# Entry 007

**Date:** July 27–August 5, 2026

**Objective:** Compare the GNN fake-factor curves across model folds and investigate an unexpectedly flat dependence on tau pT.

**Methods:**
- Developed `compare_gnn_folds.py`.
- Binned events by Barrel/Endcap, 1-/3-prong, and tau-pT bins:
  `[20, 30, 40, 50, 60, 80, 100, 400] GeV`.
- For anti-ID events in each bin, initially computed:
  - denominator = `sum(w)`
  - numerator = `sum(w * FF_GNN)`
  - plotted value = numerator / denominator
- Added debugging output for event counts, numerator, denominator, pID statistics, and per-event fake-factor statistics.

**Initial observation:**
The GNN fake factor appeared nearly constant at approximately 0.075 across essentially every pT bin and category. This was suspicious because the traditional binned fake factor has a strong pT dependence.

**Debugging checks:**
- Printed pID min/mean/std/max for application and calibration samples.
- Printed the number of events contributing to each bin.
- Printed weighted numerator and denominator for every category/pT bin.
- Confirmed that the numerator and denominator changed strongly with pT while their ratio remained nearly constant.
- Inspected per-event fake-factor distributions and confirmed there was real event-to-event spread, even though the bin averages appeared flat.

**Representative old closure values:**
- Model fold 0 global scale ≈ 0.820760, application closure ≈ 0.9932.
- Model fold 1 global scale ≈ 0.957553, application closure ≈ 0.9760.
- Model fold 2 global scale ≈ 0.976668, application closure ≈ 1.0286.

These good inclusive closure numbers did not prove that the differential pT shape was correct.

**Conclusion at this stage:**
The flatness could not be explained by a simple lack of statistics. A deeper check of event/prediction alignment was required.

---

# Entry 008

**Date:** August 2026

**Objective:** Determine whether the preprocessed H5 and SALT prediction H5 were aligned row-by-row.

**Methods:**
Compared:
`/home/shahzad/ML_Fakes_LepHad/inputData/preprocessed/fold_2/fakeCR.h5`

against the fold-0 SALT application prediction:
`logs/GN2_fakeFactor_lephad_fold0_20260721-T212751/ckpts/epoch=006-val_loss=0.02421__test_fakeCR.h5`

Both files contained exactly 2,320,993 event rows.

A direct row-by-row comparison was then performed using identifying and physics fields.

**Results:**
Despite equal row counts, the rows were not aligned:
- `eventNumber`: 0 / 2,320,993 rows matched.
- `tau_eta`: 0 / 2,320,993 matched.
- `tau_pt`: only 4 / 2,320,993 matched.
- Other fields showed only partial accidental agreement.

A subsequent event-ID search sampled 100 events from the preprocessed file and searched for them anywhere in the SALT prediction file.

**Critical result:**
100/100 sampled events were found somewhere in the prediction file.

**Conclusion:**
The two H5 files contain the same event population, but SALT has reordered the rows. Equal row counts do **not** imply row-by-row correspondence.

This explained the flat fake-factor curves: the analysis was combining the tau pT and event weights from one event with the GNN prediction from a different event. This effectively destroyed the correlation between the GNN prediction and event kinematics.

---

# Entry 009

**Date:** August 2026

**Objective:** Fix the row-order bug in `fakefactor_closure.py`.

**Fix:**
Added a `load_prediction_events()` approach that loads both:
1. the event record, and
2. the matching `pID`/`pantiID` prediction

from the **same SALT prediction H5 file**.

This guarantees that each prediction remains attached to the event on which the prediction was made.

The same principle was applied independently to the calibration prediction file.

**Updated logic:**
- Application event variables and probabilities come from `--pred`.
- Calibration event variables and probabilities come from `--calib-pred`.
- The preprocessed file is no longer used to supply row-by-row event variables when the prediction file already contains those fields.
- The existing global calibration procedure remains conceptually unchanged.

**Validation:**
After the fix, the closure plot changed dramatically:
- GNN fake factors now decrease strongly with tau pT.
- The GNN shape tracks the traditional binned fake-factor shape.
- The result agrees with the output produced by Shahzad's established shell workflow.

**Conclusion:**
The previously flat GNN fake-factor result was an analysis artifact caused by row-order mismatch, not evidence that the GNN had learned a constant fake factor.

---

# Entry 010

**Date:** August 10, 2026

**Objective:** Propagate the event-alignment fix to `compare_gnn_folds.py` and compare the three independently trained fake-factor models.

**Methods:**
- Modified `compare_gnn_folds.py` to load event variables and probabilities from each model's own SALT prediction H5.
- Preserved the calibration procedure for each model fold.
- Re-ran the fold comparison in the four categories:
  - Barrel 1-prong
  - Barrel 3-prong
  - Endcap 1-prong
  - Endcap 3-prong

**Result:**
The corrected comparison is physically sensible:
- All three model folds show a strong falling fake factor with increasing tau pT.
- The three folds learn very similar shapes.
- Differences remain between folds, particularly in normalization and some low-pT bins, but they are small relative to the overall pT dependence.
- The old approximately-flat ~0.075 behavior disappeared.

**Major conclusion:**
The GNN itself was not producing a flat fake factor. The apparent flatness came from breaking event/prediction correspondence when combining the original preprocessed H5 ordering with the reordered SALT prediction H5.

**Research significance:**
This debugging established an important analysis rule: when a prediction H5 contains both event variables and model outputs, downstream calculations should use those aligned fields together unless event identity is explicitly matched between files.

**Next Steps:**
- Quantify fold-to-fold differences and uncertainties.
- Overlay/compare the corrected GNN curves with the traditional binned fake factors.
- Continue benchmarking closure in variables not used by the traditional binned method.
- Preserve event-ID matching checks as a standard validation whenever multiple H5 files are combined.
