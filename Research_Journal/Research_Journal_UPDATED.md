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

Objective:
Initial end-to-end SALT training.

Configuration:
`/home/jjohnstu/summer_2026/run3mltoolkit/configs/GN2_bbtautau_v08_lephad_baseline.yaml`

Machine:
Shahzad's CERN machine.

Outcome:
Training completed successfully.
Metrics and plots logged to Comet.

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

The Handbook explains concepts.
The Journal records what actually happened.

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
- The minimum observed validation loss was:

`epoch=021-val_loss=0.06494.ckpt`

**Conclusions:**
- The preferred checkpoint for evaluation is the checkpoint with the lowest validation loss, not necessarily the final epoch.
- A checkpoint is a complete snapshot of the network and its training state.

**Next Steps:**
- Inspect the internal contents of a checkpoint.
- Trace how SALT loads checkpoints during evaluation.
- Determine where `p(x)` is saved and where `FF(x)=p(x)/(1-p(x))` is computed.
