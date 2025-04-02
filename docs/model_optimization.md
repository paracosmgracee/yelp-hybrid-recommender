# Model Optimization Overview for Yelp Hybrid Recommendation System

This document outlines the evolution and optimization of the **Yelp Hybrid Recommender System**, integrating content-based features, clustering, deep learning, and gradient boosting to generate robust predictions. The system is modular, efficient, and designed to run on Apple Silicon (M-series) machines.

---

## ✨ Overview

The hybrid system combines feature engineering with XGBoost and Deep Learning, plus post-processing enhancements. Collaborative filtering was considered but ultimately replaced by feature-driven hybridization.

---

## 🏗️ Architecture Evolution

| Stage           | Description |
|----------------|-------------|
| Baseline       | Simple content-based model using XGBoost, minimal features |
| Improved       | Added DL model and weighted hybrid predictions |
| Final          | Modular pipeline with scalable preprocessing, feature enrichment, model ensemble, and smoothing post-processing |

---

## 🔄 Data Processing & Feature Engineering

### 👤 User Features
- **Initial**: `average_stars`, `review_count`
- **Enhanced**: Added `fans`, `useful`, `funny`, `cool`, rating deviation, and activity rate

### 🏪 Business Features
- **Initial**: `stars`, `review_count`
- **Enhanced**: Added `price_range`, `is_open`, and one-hot encoding of top 20 categories
- **Interaction Terms**: e.g., `stars × price_range`, `stars × log(review_count)`

### 📝 Review Features
- **Initial**: `useful`, `funny`
- **Enhanced**: Added `text_length`, `cool`, and grouped aggregation features per user-business

### 📊 Clustering Features
- KMeans applied to both users and businesses using selected vectors
- Added cluster indices as categorical input

---

## 🔬 Model Engineering

### 1. XGBoost Regressor
- Objective: `reg:squarederror`
- Parameters:
  - `n_estimators=200`
  - `max_depth=6`
  - `learning_rate=0.05`
  - `reg_alpha=0.1`, `reg_lambda=1.0`
- Scaled input with `StandardScaler`

### 2. Deep Learning Model (Keras)
- Architecture: [128 → 64 → 32 → 1], ReLU activations with dropout
- Loss: `mean_squared_error`
- Optimizer: `adam`
- Early stopping enabled

---

## 🔁 Hybrid Prediction Strategy

During prediction:
- Both **XGBoost** and **Deep Learning** models generate predictions
- Final output is the weighted average of both:
  
  ```
  final_pred = 0.6 * xgb_pred + 0.4 * dl_pred
  ```
- Prediction smoothing rules:
  - Add weighted average of user & business historical ratings (if available)
  - Clip results to [1, 5]

---

## 🧼 Post-Processing Enhancements

| Strategy | Description |
|----------|-------------|
| Weighted smoothing | Combine model output with user/business average |
| Extreme value control | Soften overly high/low predictions |
| Final clipping | Ensure output remains in rating range [1, 5] |

---

## ✅ Evaluation Summary

| Metric | Value |
|--------|-------|
| **RMSE** | ~0.95 |
| **MAE**  | ~0.80 |
| Execution Time | ~550 seconds (on M3 Pro) |

### 🔍 Error Distribution
```
>=0 and <1:    103869
>=1 and <2:     32062
>=2 and <3:      5642
>=3 and <4:       471
>=4:               0
```

---

## 📈 Summary Table

| Component        | Status |
|------------------|--------|
| Feature Engineering | ✅ Enhanced with logs, clustering, interactions |
| ML Models            | ✅ XGBoost + Deep Learning ensemble |
| Pipeline             | ✅ Modular and reproducible |
| Evaluation           | ✅ RMSE and error distribution tracking |

---

## 🧠 Key Takeaways

- A hybrid model (XGBoost + DL) with proper feature engineering can outperform standalone models
- Clustering and interactions boost personalization
- Post-processing plays a critical role in final prediction refinement

---

This document acts as a historical log and technical blueprint for improving rating prediction performance on the Yelp dataset.