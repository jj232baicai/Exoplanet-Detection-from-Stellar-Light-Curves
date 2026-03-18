# Exoplanet Detection from Stellar Light Curves

**Team:** The Overthinkers  
**Course:** Machine Learning — Final Project

---

## Project Overview

This project tackles **exoplanet detection as a rare-event binary classification problem**. Using brightness measurements from NASA's Kepler Space Telescope, we analyze stellar light curves to determine whether a distant star has a planet orbiting it.

When a planet passes in front of its host star, it blocks a tiny fraction of the star's light — typically 0.01% to 1% — creating a characteristic periodic dip in brightness. The challenge is detecting these subtle signals buried in noisy, high-dimensional time-series data under extreme class imbalance.

---

## Repository Structure

```
.
├── final_project_final_vers.ipynb                        # Full modeling pipeline (EDA → ML → DL)
└── Exoplanet_Detection_from_Stellar_Light_Curves.pptx   # Final presentation slides
```

> **Data files required (not included):** `exoTrain.csv` and `exoTest.csv`  
> Download from: https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data

---

## Dataset

**Source:** NASA Kepler Space Telescope Mission (via Kaggle)

| Split | Total Stars | Planet Stars | Non-Planet Stars | Imbalance Ratio |
|---|---|---|---|---|
| Train | 5,087 | 37 (0.73%) | 5,050 (99.27%) | ~137:1 |
| Test | 570 | 5 (0.9%) | 565 (99.1%) | ~113:1 |

Each star is represented as a **time series of 3,197 brightness (flux) measurements** recorded at regular intervals.

**Labels:** `1` = Non-planet star, `2` = Confirmed exoplanet host

> ⚠️ **Class Imbalance Warning:** A naive model that always predicts "no planet" achieves 99.1% accuracy while detecting zero exoplanets. Standard accuracy is therefore a misleading metric — this project focuses on **Precision, Recall, F1-Score, and PR-AUC** for the minority (planet) class.

---

## Modeling Pipeline

```
Raw Light Curve (3,197 time steps)
        ↓
  Feature Engineering
  (FFT + Summary Stats + Dip Detection
   + Autocorrelation + Rolling Window + Derivative)
        ↓
    ML / DL Models
  (LR, RF, XGBoost, LSTM, CNN)
        ↓
  CV-Optimized Threshold Tuning
        ↓
  Planet / Non-Planet Prediction
```

---

## Notebook Structure — `final_project_final_vers.ipynb`

| Section | Description |
|---|---|
| 1. Load Data | Read `exoTrain.csv` / `exoTest.csv`, inspect class distributions |
| 2. EDA | Light curve plots (planet vs. non-planet), ACF/PACF analysis, PCA scree plot |
| 3. Feature Engineering | Six feature extraction modules (see below) |
| 4. Tabular ML Helpers | Shared evaluation: OOF threshold tuning, confusion matrix, ROC/PR curves |
| 5. Logistic Regression | L2-regularized LR with 5-fold stratified CV |
| 6. Random Forest | Randomized search hyperparameter tuning |
| 7. XGBoost | Gradient boosting with `scale_pos_weight` for imbalance handling |
| 8. Deep Learning Helpers | Augmentation, per-sample normalization, dataset/dataloader utilities |
| 9. LSTM | Bidirectional LSTM on downsampled raw signal (3,197 → ~400 time steps) |
| 10. 1D CNN (Raw) | CNN on full 3,197-step raw light curve |
| 11. 1D CNN (Engineered Features) | CNN on 1,659 engineered features reshaped as a 1D sequence |
| 12. Model Comparison | Summary table of ROC-AUC and PR-AUC across all models |

---

## Feature Engineering

All features are extracted from the raw 3,197-step light curve. The final engineered feature set totals **1,659 features**:

| Module | Count | Description |
|---|---|---|
| FFT | 1,598 | Fast Fourier Transform magnitude spectrum (half-spectrum, normalized) — captures periodic transit patterns and orbital frequency signatures |
| Summary Stats | 15 | Mean, median, std, min, max, range, Q1, Q3, IQR, skewness, kurtosis, and more |
| Dip Detection | ~10 | Peak count, dip count, mean/max dip depth, dip duration — directly targets the transit brightness-drop signature |
| Autocorrelation | 6 | Autocorrelation at lags 10, 25, 50, 100, 200, 500 — detects the periodicity of orbital cycles |
| Rolling Window | ~12 | Rolling mean and std at windows 50, 100, 200 — captures local flux variations over time |
| Derivative | 8 | First/second-order differences (mean abs, max, std, min, max) — highlights sharp brightness transitions |
| **Total** | **1,659** | Combined feature set used for CNN (engineered version); 1,613 (FFT + Stats) used for classical ML baselines |

**Class imbalance handling:** SMOTE is applied to the **training set only**, balancing planet/non-planet to 1:1 ratio (10,100 training samples after oversampling). The test set is kept at its natural imbalance to reflect real-world conditions.

---

## Models & Results

### Performance Summary

| Model | Input | Threshold | Precision | Recall | F1 | Notes |
|---|---|---|---|---|---|---|
| Stats Only | 15 features | 0.90 | 0.75 | 0.60 | 0.67 | Weakest — misses 2 out of 5 planets |
| FFT + Stats | 1,613 features | 0.70 | 1.00 | 0.80 | 0.89 | Zero false positives |
| XGBoost | 1,613 features | 0.65 | 1.00 | 0.80 | 0.89 | Best classical model |
| **CNN (Eng. Features)** ⭐ | 1,659 features | 0.90 | ~0.83 | 1.00 | ~0.91 | Best overall — perfect recall |

All six models (Logistic Regression, Random Forest, XGBoost, LSTM, 1D CNN raw, 1D CNN engineered) are benchmarked using ROC-AUC and PR-AUC.

### Threshold Tuning
Default threshold of 0.5 is suboptimal under extreme imbalance. All models use **5-fold stratified cross-validation** to sweep thresholds (0.05–0.95) and select the value that maximizes F1-score on out-of-fold predictions.

| Model | Optimal Threshold |
|---|---|
| Stats Only | 0.90 |
| FFT + Stats | 0.70 |
| XGBoost | 0.65 |
| CNN (Eng. Features) | 0.90 |

### Key Findings
- **FFT features are critical:** Frequency-domain features improved recall from 60% to 80% and achieved perfect precision over stats-only.
- **CNN achieves perfect recall (1.00):** Catches all 5 exoplanets in the test set — the most important property for a scientific screening tool.
- **CNN precision-recall tradeoff:** CNN trades some precision (0.83 vs. 1.00) for perfect recall. In rare-event screening, missing a true signal is costlier than a false alarm.
- **Threshold tuning matters:** CV-optimized thresholds consistently outperform the default 0.5 across all models.

---

## Installation & Dependencies

```bash
pip install numpy pandas matplotlib scipy scikit-learn xgboost torch imbalanced-learn
```

**Hardware:** GPU recommended for LSTM and CNN training. PyTorch uses CUDA automatically if available; falls back to CPU.

---

## Limitations & Future Work

- **Small positive support:** Only 5 planet examples in the test set — strong results, but statistical conclusions should be treated with caution.
- **No external validation:** Results have not been tested on independent star populations or telescope datasets.
- **Future directions:** External dataset validation, more positive training samples, probability calibration, and addition of astrophysical domain features (stellar type, magnitude, etc.).

> **Recommendation:** Use the CNN model on engineered light-curve features as a **candidate-screening tool** to rank stars for follow-up review — not as a standalone scientific confirmation system.
