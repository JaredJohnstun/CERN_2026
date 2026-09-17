# ATLAS HH → bbττ Fake-Background Estimation with Machine Learning

Undergraduate research project investigating machine-learning methods for estimating
fake-background contributions in the ATLAS HH → bbττ analysis.

This work was conducted during Summer 2026 through California State University,
Sacramento and involved analysis using CERN computing resources and the ATLAS
software/research environment.

## Project Overview

The HH → bbττ analysis searches for Higgs boson pair production in events where
one Higgs boson decays to a pair of bottom quarks and the other decays to a pair
of tau leptons.

A significant challenge in this analysis is estimating backgrounds containing
objects that are misidentified as hadronically decaying tau leptons. Traditionally,
these backgrounds can be estimated using a binned fake-factor method.

The goal of this project was to investigate whether machine-learning methods could
provide an event-level fake-factor estimate that captures correlations between
multiple event variables.

## Machine-Learning Approach

The project used graph neural networks based on the GN2 architecture and trained using the SALT machine-learning framework.

The network predicts the probability that an event belongs to the identification
(ID) region rather than the anti-ID region. This probability can be converted into
an event-level fake factor:

$$
FF(x) = \frac{p(x)}{1-p(x)}
$$

where $p(x)$ is the model-predicted probability for an event with features $x$.

The analysis used a three-fold procedure in which separate data folds were used
for training, calibration, and application. Model predictions were then compared
with the conventional binned fake-factor method.

## Negative Monte Carlo Weights

Part of the project investigated the treatment of negative Monte Carlo event
weights.

In addition to conventional signed-weight subtraction, a soft-mask approach was
studied as a method of performing background subtraction while avoiding negative
weights during machine-learning training.

The resulting distributions and fake-factor predictions were validated against
the conventional signed-weight method.

## Validation and Closure Studies

Model performance was evaluated using several validation procedures, including:

- Three-fold training, calibration, and application
- Comparison with conventional binned fake factors
- Fake-factor closure tests
- Per-bin validation of important kinematic variables
- Comparison of signed-weight and soft-mask background subtraction
- Diagnostic plots and statistical checks

## Selected Results

The conventional fake-factor calculation was first reproduced as a validation
of the analysis pipeline. The inclusive fake factor obtained from the validation
was 0.0753, compared with the reference value of 0.0754.

Three independent machine-learning folds were then trained and evaluated using
separate training, calibration, and application datasets. Closure tests were
used to compare the fake-background yield predicted by the machine-learning
fake factors with the corresponding target yield.

The soft-mask background-subtraction method was also validated against the
conventional signed-weight approach. Across the three folds, the inclusive
soft-mask fake-factor results remained consistent with the conventional
fake-factor calculation, while allowing model training to avoid the direct
use of negative event weights.

Selected diagnostic plots and validation outputs produced during these studies
are available in the `plots/` and `eval_outputs/` directories in this repository.

## Tools and Technologies

The project primarily used:

- Python
- SALT
- PyTorch-based machine-learning tools
- NumPy
- h5py
- HDF5 datasets
- Linux / Bash
- Git
- CERN remote computing resources

## Repository Contents

This repository contains code, documentation, configuration files, plots, and
analysis material produced during the project.

Notable contents include:

- `Handbook.md` — technical documentation and explanations developed during the project
- `Research_Journal.md` — chronological record of the research process, debugging, and results
- `evaluation/` — model evaluation and fake-factor validation tools
- `plots/` — selected analysis and diagnostic plots
- YAML configuration files used for SALT model training
- Python scripts used for data inspection, validation, and comparison studies

## Research Context

This repository documents my undergraduate research and the code and analysis
work I performed during Summer 2026.

It is intended as a record and portfolio of my individual research work and
should not be interpreted as an official ATLAS Collaboration software release
or official ATLAS physics result.
