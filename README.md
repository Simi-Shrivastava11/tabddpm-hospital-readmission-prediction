# TabDDPM: Using Diffusion Models to Generate Synthetic Patient Data for Hospital Readmission Prediction

**Group 5 — DATA 612 Final Project**
Taneir Arani · Jiten Bhalavat · Rohith Mandla · Anum Sagheer · Simi Shrivastava

---

## Overview

Hospital readmission within 30 days is both costly and clinically significant. In the Diabetes 130-US Hospitals dataset, only **11% of patients** were readmitted within 30 days — a severe class imbalance that causes standard classifiers to ignore the exact patients that matter most.

This project investigates whether **TabDDPM** (Tabular Denoising Diffusion Probabilistic Model, Kotelnikov et al. ICML 2023) can generate synthetic minority-class patients realistic enough to improve downstream classifier performance. We compare TabDDPM against three baselines:

| Method | Description |
|---|---|
| **No fix** | Train XGBoost on the raw imbalanced data |
| **SMOTE** | Interpolate between real minority-class patients |
| **CTGAN** | GAN-based conditional tabular data generator |
| **TabDDPM** | Diffusion-model-based conditional tabular generator *(ours)* |

---

## Dataset

**Diabetes 130-US Hospitals** (UCI / Kaggle)
- 101,766 patient records from 130 US hospitals (1999–2008)
- 50 features: demographics, diagnoses (ICD codes), medications, lab results
- Target: `readmitted` — `NO`, `<30` (within 30 days), `>30` (after 30 days)
- Class split: ~11% `<30`, ~89% other

The raw CSV lives at `data/diabetic_data.csv` and is never modified in place.

---

## Project Structure

```
tabddpm-hospital-readmission-prediction/
├── data/
│   └── diabetic_data.csv          # Raw dataset (do not modify)
├── models/                        # Saved model checkpoints (gitignored)
├── results/
│   ├── figures/                   # Generated plots (gitignored)
│   └── metrics/                   # Evaluation CSVs / JSON (gitignored)
├── notebook/
│   └── main.ipynb                 # End-to-end notebook
├── requirements.txt
├── CLAUDE.md                      # AI assistant context file
└── README.md
```

---

## ML Workflow

```
Raw Data
   │
   ▼
1. Preprocessing
   • Drop identifiers (encounter_id, patient_nbr)
   • Replace "?" with NaN; impute or drop
   • Encode categoricals (label / one-hot)
   • Scale numerics (StandardScaler)
   • Stratified 80/20 train-test split
   │
   ▼
2. Baseline Classifier (no augmentation)
   • XGBoost on imbalanced training set
   • Record F1 (<30 class) and AUC-ROC
   │
   ▼
3. Synthetic Data Generation
   ├── SMOTE       (imbalanced-learn)
   ├── CTGAN       (ctgan / sdv)
   └── TabDDPM     (PyTorch, implemented from scratch)
         • Forward: add Gaussian noise over T=1000 steps (numerical cols)
         • Forward: multinomial diffusion (categorical cols)
         • Reverse: U-Net-style MLP predicts added noise (class-conditioned)
         • Generate at 50%, 100%, 200% minority-class augmentation levels
   │
   ▼
4. Augmented Classifiers
   • Retrain XGBoost on real + synthetic training data for each method × level
   • 5 random seeds per experiment; report mean ± std
   │
   ▼
5. Evaluation
   • Primary:   F1 score on the <30-day readmission class
   • Secondary: AUC-ROC
   • Fidelity:  Column-wise distribution plots (real vs. synthetic)
                Jensen-Shannon divergence per column
                t-SNE / UMAP 2-D embedding overlay
   • Stress:    Train entirely on synthetic, test on real patients
```

---

## Setup

### 1. Clone and enter the repo

```bash
git clone <repo-url>
cd tabddpm-hospital-readmission-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU note:** if you have a CUDA-capable GPU, install the matching `torch` wheel from pytorch.org before running `pip install -r requirements.txt`.

### 4. Launch Jupyter

```bash
jupyter lab
```

Open `notebook/main.ipynb` and run cells top to bottom.

---

## Evaluation Metrics

| Metric | Purpose |
|---|---|
| F1 (`<30` class) | Primary — how well the model catches high-risk patients |
| AUC-ROC | Overall ranking quality |
| Jensen-Shannon divergence | Fidelity of generated distributions per column |
| t-SNE overlay | Visual fidelity check |
| Synth-train / real-test F1 | Stress test of generative quality |

---

## References

1. Kotelnikov et al. (2023). *TabDDPM: Modelling Tabular Data with Diffusion Models.* ICML 2023. https://arxiv.org/abs/2209.15421
2. Ho et al. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS 2020. https://arxiv.org/abs/2006.11239
3. Xu et al. (2019). *Modeling Tabular Data using Conditional GAN.* NeurIPS 2019. https://arxiv.org/abs/1907.00503
4. Chawla et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR. https://arxiv.org/abs/1106.1813
