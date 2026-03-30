# CLAUDE.md — AI Assistant Context

This file gives Claude (and other AI assistants) the persistent context needed to work effectively in this repository.

---

## Project Summary

**Goal:** Compare TabDDPM against SMOTE and CTGAN for fixing class imbalance in a hospital readmission prediction task. The research question is whether diffusion-model-generated synthetic patients improve an XGBoost classifier's F1 score on the rare (<30-day readmission) class.

**Dataset:** `data/diabetic_data.csv` — 101,766 rows × 50 columns. Target column is `readmitted` with values `NO`, `<30`, `>30`. The minority class is `<30` (~11%). Missing values are encoded as `"?"`.

**Stack:** Python 3.9+, PyTorch (TabDDPM), scikit-learn, XGBoost, imbalanced-learn (SMOTE), CTGAN/SDV, pandas, matplotlib/seaborn.

---

## Workflow (in order)

1. **Preprocessing** — clean `?`, encode categoricals, scale numerics, stratified 80/20 split
2. **Baseline** — XGBoost on raw imbalanced data
3. **Synthetic generation** — SMOTE, CTGAN, TabDDPM (class-conditioned, T=1000 diffusion steps)
4. **Augmented training** — XGBoost retrained at 50%, 100%, 200% minority augmentation levels
5. **Evaluation** — F1 (<30 class), AUC-ROC, Jensen-Shannon divergence, t-SNE plots, synth-train/real-test stress test
6. **Reporting** — save figures to `results/figures/`, metrics to `results/metrics/`

---

## Key Conventions

- **Never modify** `data/diabetic_data.csv` in place. All preprocessing produces new DataFrames in memory.
- `models/` and `results/` are gitignored — they are generated artifacts, not source files.
- All experiments use **5 random seeds**; always report mean ± std.
- Primary metric is **F1 on the `<30` class**, not overall accuracy.
- TabDDPM is **implemented in PyTorch from scratch** (not via an external `tab-ddpm` package) per the proposal.
- Categorical columns use **multinomial diffusion**; numerical columns use **Gaussian diffusion**.
- The model is **class-conditioned** so we can generate specifically `<30` patients.

---

## File Roles

| Path | Role |
|---|---|
| `data/diabetic_data.csv` | Raw input, read-only |
| `notebook/main.ipynb` | Primary development notebook |
| `models/` | Serialized PyTorch checkpoints and sklearn pipelines |
| `results/figures/` | PNG/SVG plots |
| `results/metrics/` | CSV / JSON evaluation tables |
| `requirements.txt` | Pinned dependencies |

---

## What NOT to Do

- Do not add privacy/anonymisation mechanisms (out of scope per proposal).
- Do not connect to any external medical system or API.
- Do not test on datasets other than `diabetic_data.csv`.
- Do not use pre-built TabDDPM packages — the model is implemented from scratch in PyTorch.
- Do not change the 80/20 split ratio or remove stratification.

---

## Evaluation Checklist (before submitting results)

- [ ] F1 and AUC-ROC reported for all 4 methods (no-fix, SMOTE, CTGAN, TabDDPM)
- [ ] Results at 50%, 100%, 200% augmentation levels for generative methods
- [ ] 5-seed mean ± std reported for every number
- [ ] Column-distribution plots saved to `results/figures/`
- [ ] Jensen-Shannon divergence table saved to `results/metrics/`
- [ ] t-SNE overlay plot saved to `results/figures/`
- [ ] Synth-train / real-test stress test recorded
