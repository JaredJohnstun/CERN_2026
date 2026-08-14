# Di-Higgs Handbook (Living Document)

> **Status:** Living source document (canonical copy)
>
> **Purpose:** This is the master reference for Jared Johnstun's ATLAS Di-Higgs fake-factor GNN research. It is intended to grow throughout the bachelor's senior project and remain useful into graduate school.

# Project Charter

## Goal

Build a textbook-style handbook that explains the physics, machine learning, software, and analysis decisions behind the project.

For every topic answer:

1. What is it?
2. Why does it exist?
3. How does it work?
4. Why does it matter for *this* Di-Higgs project?

Always write for **future Jared**.

## Style Guide

- Prefer intuition before formalism.
- Connect every concept back to the research.
- Include:
  - Common Misconceptions
  - Research Insight
  - Connections to SALT
  - Connections to ATLAS physics
- Cross-reference related sections whenever possible.

# Planned Structure

1. Physics Foundations
2. Di-Higgs Analysis
3. Fake-Factor Method
4. Machine Learning Foundations
5. SALT Framework
6. Project-Specific Network
7. Comet & Training Interpretation
8. Evaluation & Inference
9. Variable Encyclopedia
10. Glossary

# Current Notes

## Fake Factor

- Traditional fake factors use binned histograms.
- Project goal: replace histogram with a learned function.

`FF(x)=p(x)/(1-p(x))`

where `p(x)` is learned by a GNN.

## ID vs Anti-ID

- ID = pass tau-ID.
- Anti-ID = fail tau-ID while passing all other selections.
- Passing tau-ID does NOT imply a genuine tau.

## True Tau Subtraction

Monte Carlo estimates the genuine tau contamination, which is statistically subtracted before measuring fake factors.

## Current SALT YAML

Important finding:

The uploaded `GN2_bbtautau_v08_lephad_baseline.yaml` is **not** yet the fake-factor network.

It is the baseline HH signal classifier. The future project will likely add a new ClassificationTask for ID vs anti-ID while keeping the encoder and pooling architecture unchanged.

## Open Questions

- What variables are most important for p(x)?
- Which features dominate fake migration?
- How does changing the loss affect p(x)?
- Which checkpoint should be used for inference?

## Variable Encyclopedia (TODO)

Every YAML variable should eventually receive:
- Definition
- Units
- Physical meaning
- Computation
- Why useful for the GNN
- Related variables

# Instructions for Future ChatGPT

Treat this file as the authoritative handbook.

Never overwrite explanations without improving them.

Preserve structure and numbering.

When new concepts arise:
- integrate them into the appropriate chapter;
- update cross references;
- add glossary entries if needed.

The handbook should eventually read like a polished textbook suitable for a bachelor's senior project.

## Checkpoint Selection

During training, SALT saves a checkpoint (`.ckpt`) after every epoch. Each checkpoint is a complete snapshot of the model at that moment, containing the learned network weights, optimizer state, learning-rate scheduler state, epoch number, and validation metrics.

Checkpoint filenames include the epoch and validation loss, for example:

`epoch=021-val_loss=0.06494.ckpt`

### Which checkpoint should be used?

For inference and evaluation, choose the checkpoint with the **lowest validation loss**, not necessarily the final epoch. This model is expected to generalize best to unseen data.

### Connection to the Fake-Factor Network

The network learns the probability function `p(x)`. The fake factor is computed afterward using

`FF(x) = p(x)/(1-p(x))`.

Evaluation loads the selected checkpoint and performs forward passes only; no additional learning occurs.

---

# Evaluation Pipeline

## SALT Test Output

After training, the `salt test` stage performs inference using a selected checkpoint and writes the model predictions into a new HDF5 file. These prediction columns are appended to the existing event variables.

Example prediction columns observed during the baseline project:
- `..._pB`
- `..._pSGGF`
- `..._pSVBF`
- `..._pSGGFBSM`
- `..._pggFHighMassB`
- `..._pggFHighMassS`

## Binary vs. Multiclass Outputs

Many evaluation scripts assume a binary classifier with outputs `pS` and `pB`, where `pS + pB = 1`.

The baseline Di-Higgs network is multiclass, predicting separate probabilities for ggF, VBF, and BSM Higgs-pair production in addition to background. Consequently there is no `pS` column written to that HDF5 file.

## ROC Curves

A ROC (Receiver Operating Characteristic) curve is created by sweeping the classifier threshold from high to low.

- **True Positive Rate (TPR):** fraction of signal correctly accepted.
- **False Positive Rate (FPR):** fraction of background incorrectly accepted.

The closer the curve approaches the upper-left corner, the better the classifier. The diagonal corresponds to random guessing.

The Area Under the Curve (AUC) summarizes overall discrimination power:
- AUC = 1.0: perfect classifier.
- AUC = 0.5: random classifier.

## Evaluation Debugging Workflow

When evaluation fails:
1. Read the traceback.
2. Identify the missing column or dataset.
3. Inspect the output HDF5 file.
4. Compare the HDF5 column names with `eval_config.yaml`.
5. Update the evaluation configuration to match the stored prediction names.

---

# Fake-Factor GNN Workflow

## Three-Fold Train / Calibration / Application Scheme

The fake-factor workflow uses three folds with distinct jobs. For model fold `k`:

- **Training:** fold `k`
- **Calibration:** fold `(k+1) % 3`
- **Application/test:** fold `(k+2) % 3`

For example, model fold 0 is trained on fold 0, calibrated on fold 1, and applied to fold 2.

The calibration fold is separate from the application fold so that the final application sample is not used to tune the global normalization.

## What `pID` and `pantiID` Mean

For the binary fake-factor classifier, SALT writes scores for the two target classes:
- `pID`: model score/probability for the event belonging to the ID class.
- `pantiID`: model score/probability for the event belonging to the anti-ID class.

When needed, the two outputs are normalized using

`p = pID / (pID + pantiID)`.

The event-level GNN fake factor is then

`FF_GNN(x) = p / (1 - p)`.

This is an **event-level** quantity: different anti-ID events can receive different fake factors according to their features `x`.

## Traditional Binned Fake Factor

For a category and pT bin, the traditional fake factor is

`FF_binned = sum(w)_ID / sum(w)_antiID`

with `w = fakefactor_weight`, representing the data-minus-true-MC weighted yield.

The categories used here are:
- Barrel 1-prong
- Barrel 3-prong
- Endcap 1-prong
- Endcap 3-prong

with tau-pT edges:

`[20, 30, 40, 50, 60, 80, 100, 400] GeV`.

## Averaging the Event-Level GNN Fake Factor

To compare the smooth GNN output with a traditional binned FF, restrict to anti-ID events in a category/pT bin and compute the weighted average:

`<FF_GNN> = sum(w_i * FF_i) / sum(w_i)`.

Here:
- the denominator is the effective weighted anti-ID yield in the bin;
- the numerator is the GNN-predicted weighted ID yield from those anti-ID events;
- the ratio is the weighted mean GNN fake factor in that bin.

This distinction is important: the GNN produces one FF per event, while the plotted point summarizes all selected event-level FFs in a bin.

## Global Calibration

A global scale factor can be derived on the calibration fold:

`scale = actual_ID_yield / predicted_ID_yield`.

The predicted yield is obtained by applying the GNN fake factors to anti-ID events. The same single scale factor is then applied to the GNN fake factors on the application fold.

This changes the overall normalization but **does not create or alter the differential shape** as a function of pT or other variables.

A good inclusive closure after calibration therefore does not, by itself, guarantee correct differential closure.

---

# Critical HDF5 Event-Ordering Lesson

## The Problem Discovered

A major debugging issue arose when comparing a preprocessed `fakeCR.h5` file with the corresponding SALT prediction H5.

The files had the same number of rows, which initially suggested that row `i` in one file represented row `i` in the other. This assumption was false.

For the fold-2 application sample, both files had 2,320,993 rows, but a row-by-row comparison found:
- 0 matching `eventNumber` values at the same row index;
- 0 matching `tau_eta` values at the same row index;
- only 4 matching `tau_pt` values at the same row index.

A search by event identifiers then found 100/100 sampled original events somewhere in the SALT prediction file.

## Interpretation

The event population was not missing or replaced. **SALT had reordered the events.**

Therefore this is unsafe:

`event variables from original H5 row i + prediction from SALT H5 row i`

unless the two files have first been explicitly aligned by a unique event key.

Doing this produced a severe but deceptive analysis bug: tau pT from one event was paired with the GNN prediction of another event.

## Why the Bug Produced a Flat Curve

The GNN prediction genuinely depended on event kinematics. However, randomly pairing predictions with unrelated pT values destroys that correlation.

When many such mismatched events are averaged within pT bins, each bin samples approximately the same overall prediction distribution. The result is an artificially flat mean fake factor.

This is why the earlier plots showed approximately 0.075 in nearly every pT bin despite substantial event-level variation.

## Correct Solution

When possible, load both the event variables and prediction fields from the **same SALT prediction H5**.

The corrected `load_prediction_events()` strategy:
1. opens the SALT prediction file;
2. locates the structured dataset containing `pID` and `pantiID`;
3. reads that dataset once;
4. extracts the event records and their matching predictions from the same row ordering;
5. normalizes `pID` and `pantiID` if necessary;
6. returns the aligned event array and `pID`.

The calibration prediction file is handled independently in the same way.

If separate files absolutely must be combined, they should be joined using a reliable event identifier rather than assuming identical row order.

### General Rule

**Equal HDF5 lengths do not prove event alignment.**

Whenever event quantities from one file are combined with model outputs from another:
- compare event identifiers;
- verify ordering;
- or explicitly join/match the events.

This should be treated as a standard validation step in future SALT analyses.

---

# Corrected Closure Result

After preserving event/prediction alignment, the fake-factor closure plots changed from nearly flat GNN curves to the expected falling pT dependence.

The corrected GNN:
- has its largest fake factors at low tau pT;
- decreases rapidly as tau pT increases;
- broadly follows the traditional binned fake-factor shape;
- reproduces the behavior seen in Shahzad's established workflow.

This demonstrates that the earlier flat behavior was **not a failure of the trained GNN**. It was a downstream bookkeeping/alignment error.

---

# Comparing the Three GNN Model Folds

After propagating the alignment fix to `compare_gnn_folds.py`, the three independently trained models were compared in the four eta/prong categories.

## Corrected Observation

All three folds now show:
- strong pT dependence;
- the same overall falling shape;
- relatively small fold-to-fold differences compared with the full dynamic range of the fake factor.

This is encouraging because independently trained folds recovering similar shapes suggests the learned dependence is stable rather than an artifact of a single training sample.

Some normalization and low-pT differences remain and should be quantified rather than ignored.

## Research Interpretation

The corrected comparison changes the scientific question.

Before the fix, the central question was:
> Why did the GNN apparently learn an almost constant fake factor?

After the fix, that premise is no longer supported. The relevant questions are now:
- How consistent are the three independently trained folds?
- How closely does each GNN reproduce the traditional binned method?
- Does the GNN improve closure in variables that the traditional pT/eta/prong binning does not explicitly model?
- What systematic uncertainty should be associated with fold-to-fold variation?

---

# Fake-Factor Debugging Checklist

When a fake-factor result looks suspicious:

1. Check the prediction file contains the expected `pID` and `pantiID` fields.
2. Check the number of prediction and event rows.
3. Do **not** stop at equal row counts.
4. Compare `runNumber`/`eventNumber` or another reliable event key.
5. Inspect pID min, max, mean, and standard deviation.
6. Inspect event-level `FF = p/(1-p)` statistics.
7. Print per-bin event counts.
8. Print the weighted numerator `sum(w*FF)` and denominator `sum(w)`.
9. Check inclusive closure separately from differential closure.
10. Confirm calibration uses a separate calibration fold.
11. Compare the final result with an independent known-good workflow when available.

# Updated Open Questions

- Quantify the fold-to-fold spread of the corrected GNN fake-factor curves.
- Compare corrected GNN closure against the binned method in variables not used by the binned parametrization.
- Determine the appropriate uncertainty treatment for the learned fake-factor method.
- Continue studying which input features drive the event-level fake factor.
- Preserve explicit event-alignment validation in future evaluation scripts.
