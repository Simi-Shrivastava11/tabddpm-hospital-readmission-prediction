# TabDDPM for Hospital Readmission Prediction

Group project exploring whether **diffusion-based synthetic tabular data generation** can improve prediction of **30-day hospital readmission** in an imbalanced clinical dataset.

This repository investigates whether **TabDDPM (Tabular Denoising Diffusion Probabilistic Model)** can generate realistic synthetic minority-class patient records that improve downstream classifier performance compared with standard approaches such as **no augmentation**, **SMOTE**, and **CTGAN**.

> **Project status:** In progress  
> This repository documents an active group research project and will continue to evolve as implementation, experiments, and evaluation are completed.

## Overview

Hospital readmission within 30 days is both clinically important and expensive. In the Diabetes 130-US Hospitals dataset, only about **11%** of patients are readmitted within 30 days, while the remaining **89%** are not. This severe class imbalance makes it difficult for standard classifiers to learn the patterns associated with the patients who matter most.

This project studies whether **TabDDPM**, a diffusion model designed for tabular data, can generate realistic synthetic high-risk patient records and improve performance on the rare readmission class. We compare four settings:

- **No fix** — train directly on the imbalanced data
- **SMOTE** — oversample the minority class using interpolation
- **CTGAN** — generate synthetic rows with a GAN-based model
- **TabDDPM** — generate synthetic rows with a diffusion model

## Research Question

Can a diffusion model learn what high-risk diabetic patients look like well enough to generate realistic synthetic readmission cases, and does augmenting the training data with those cases improve classifier performance on real patients?

## My Contribution

My primary responsibility in this project is **class conditioning and synthetic data generation**.

Specifically, I am responsible for:
- implementing the class-conditioning component so the model can distinguish between readmitted and non-readmitted patients
- injecting the readmission label into the model during training
- writing the generation pipeline that starts from noise and produces synthetic minority-class patient rows
- helping create visual comparisons of real vs. generated patients for the final analysis

## Dataset

This project uses the **Diabetes 130-US Hospitals** dataset, available through Kaggle.

Dataset characteristics:
- **101,766** patient records
- data collected from **130 U.S. hospitals**
- approximately **50 features**
- includes both **numerical** and **categorical** variables
- target variable indicates whether the patient was readmitted within 30 days

The mixed data types make this a strong use case for tabular generative modeling, since the method must handle both continuous and categorical features.

## Project Goals

The project has three main goals:

1. **Train a TabDDPM model** on structured hospital readmission data
2. **Generate synthetic minority-class patients** at different augmentation levels
3. **Evaluate whether augmentation improves classifier performance** relative to standard imbalance-handling baselines

## Methodology

### 1. Data Preparation
The dataset is cleaned, categorical variables are encoded, numerical variables are scaled, and the data is split into stratified training and test sets.

### 2. TabDDPM Training
The diffusion model is trained in two stages:
- a forward process that gradually corrupts patient rows with noise
- a reverse process that learns to recover the original structure from noisy inputs

For categorical columns, the project uses multinomial diffusion rather than Gaussian noise.

### 3. Conditional Generation
The model is conditioned on the class label so it can explicitly generate synthetic **readmitted** patients rather than uncontrolled random rows. My part of the project focuses on this conditioning and generation step.

### 4. Baselines
We compare TabDDPM against:
- no class-imbalance correction
- SMOTE
- CTGAN

### 5. Downstream Evaluation
Each augmentation strategy is used to retrain an **XGBoost** classifier, which is then evaluated on held-out real patient data.

## Evaluation Plan

The primary evaluation metric is:

- **F1 score on the readmitted (<30 day) class**

Secondary and supporting metrics include:
- **AUC-ROC**
- **Jensen-Shannon divergence** between real and synthetic feature distributions
- **t-SNE-based visual comparisons**
- a **synthetic-train / real-test** stress test to assess whether the generated data captures useful real-world structure

Experiments are planned across multiple augmentation levels:
- **50%**
- **100%**
- **200%** of the minority-class size

## Tech Stack

- **Python**
- **PyTorch**
- **XGBoost**
- **scikit-learn**
- **Jupyter Notebook**
- **imbalanced-learn**
- **CTGAN / SDV**

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd tabddpm-hospital-readmission-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter

```bash
jupyter lab
```

Then open:

```text
notebook/main.ipynb
```

## Current Status

This project is still in the **implementation and experimentation phase**. At this stage, the repository is intended to document:

- the research question
- planned methodology
- team responsibilities
- evolving implementation for a graduate machine learning project

Results, visualizations, and final evaluation findings will be added as the project progresses.

## Why This Project Matters

This project sits at the intersection of:
- healthcare machine learning
- class-imbalance learning
- synthetic tabular data generation
- diffusion models for structured data

## Future Additions

As the project progresses, this repository will be expanded with:
- model architecture details
- training curves
- synthetic vs. real distribution plots
- final evaluation metrics
- ablation comparisons across augmentation levels
- results summary and conclusions

## Team

Group 5 — DATA 612 Final Project

- Taneir Arani
- Jiten Bhalavat
- Rohith Mandla
- Anum Sagheer
- Simi Shrivastava

## References

1. Kotelnikov et al. (2023). *TabDDPM: Modelling Tabular Data with Diffusion Models*
2. Ho et al. (2020). *Denoising Diffusion Probabilistic Models*
3. Xu et al. (2019). *Modeling Tabular Data using Conditional GAN*
4. Chawla et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*
