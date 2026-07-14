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

FF(x)=p(x)/(1-p(x))

where p(x) is learned by a GNN.

## ID vs Anti-ID

- ID = pass tau-ID.
- Anti-ID = fail tau-ID while passing all other selections.
- Passing tau-ID does NOT imply a genuine tau.

## True Tau Subtraction

Monte Carlo estimates the genuine tau contamination, which is statistically subtracted before measuring fake factors.

## Current SALT YAML

Important finding:

The uploaded GN2_bbtautau_v08_lephad_baseline.yaml is **not** yet the fake-factor network.

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
