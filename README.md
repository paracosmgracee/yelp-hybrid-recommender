# 🌟 Yelp Hybrid Recommender System

This is a hybrid recommendation system for Yelp review data, combining **content-based features**, **collaborative filtering (Spark ALS)**, and **deep learning & gradient boosting models** to generate robust rating predictions.

## 📁 Project Structure

```
yelp-hybrid-recommender/
├── data/                        # Raw input data (JSON/CSV)
│   ├── full_data/              # Full Yelp dataset with training, val, json
│   └── sample_data/            # Sample data for fast testing
├── model/                      # Saved models (XGBoost, DL, Scaler)
├── output/                     # Prediction results
│   └── prediction_full.csv
├── docs/
│   └── model_optimization.md   # Model tuning & experiment logs
├── notebook/
│   └── analysis.ipynb          # Analysis and visualization
├── src/                        # Source code files
│   ├── clustering.py           # Business category clustering (KMeans)
│   ├── data_loader.py          # JSON parser and cluster feature generator
│   ├── evaluate_model.py       # RMSE, MAE, error analysis
│   ├── feature_building.py     # Vector builder for ML model
│   ├── feature_engineering.py  # Same as loader but modularized
│   ├── hybrid_predictor.py     # Runs prediction using trained models
│   ├── model_training_old.py   # Old version of training script
│   ├── model_training.py       # Final training logic for XGB + DL
│   ├── test_spark.py           # Spark test file
│   └── utils_preprocessing.py  # Utilities
├── requirements.txt            # Python dependencies
├── README.md                   # ← You are here
```

---

## 🚀 Pipeline Overview

### 1️⃣ Data Preprocessing

Choose either modular or combined feature extraction:

#### Option A: Full pipeline
```bash
python src/data_loader.py \
  data/full_data/user.json \
  data/full_data/business.json \
  data/full_data/review_train.json \
  data/full_data/
```

#### Option B: Modular
```bash
python src/feature_engineering.py \
  data/full_data/user.json \
  data/full_data/business.json \
  data/full_data/review_train.json \
  data/full_data/
```

### 2️⃣ Clustering
```bash
python src/clustering.py data/full_data/
```
Outputs:
- `user_clusters.json`
- `business_clusters.json`
- `categories_dict.json`

### 3️⃣ Model Training
```bash
python src/model_training.py \
  --train data/full_data/yelp_train.csv \
  --user_features data/full_data/user_features.json \
  --business_features data/full_data/business_features.json \
  --review_features data/full_data/review_features.json \
  --user_clusters data/full_data/user_clusters.json \
  --business_clusters data/full_data/business_clusters.json \
  --categories_dict data/full_data/categories_dict.json \
  --output_dir model/
```

### 4️⃣ Prediction
```bash
python src/hybrid_predictor.py \
  --val data/full_data/yelp_val.csv \
  --user_features data/full_data/user_features.json \
  --business_features data/full_data/business_features.json \
  --review_features data/full_data/review_features.json \
  --user_clusters data/full_data/user_clusters.json \
  --business_clusters data/full_data/business_clusters.json \
  --categories_dict data/full_data/categories_dict.json \
  --model_dir model/ \
  --output output/prediction_full.csv
```

### 5️⃣ Evaluation
```bash
python src/evaluate_model.py \
  output/prediction_full.csv \
  data/full_data/yelp_val.csv
```

---

## 📊 Model Summary

| Component         | Method                          |
|------------------|----------------------------------|
| Content-based     | JSON metadata + interactions    |
| Clustering        | KMeans for users/businesses     |
| ML Model (1)      | XGBoost (reg:squarederror)      |
| ML Model (2)      | Deep Learning (Keras MLP)       |
| Feature Scaling   | StandardScaler                  |
| Prediction Merge  | Weighted average + smoothing    |

---

## ✅ Features Used

- **User:** `average_stars`, `review_count`, `fans`, `cool`, `funny`, etc.
- **Business:** `stars`, `review_count`, `price_range`, `is_open`, category one-hot
- **Review:** `useful`, `funny`, `cool`, text length
- **Clustering:** user & business clusters
- **Feature Interactions**: e.g. star diff, review count × rating

---

## 🔧 Requirements

### `requirements.txt`
```text
numpy
pandas
scikit-learn==1.1.3
xgboost==1.6.2
joblib
textblob
matplotlib
seaborn
pyspark==3.1.2

tensorflow-macos==2.12.0
tensorflow-metal==0.8.0
```

### 💻 Environment Notes
This setup is tested on **Apple Silicon (M3 Pro)** with hardware-accelerated TensorFlow.
For non-Mac users, replace `tensorflow-macos` and `tensorflow-metal` with:
```text
tensorflow==2.12.0
```

---

## 💡 Tips

- Use `sample_data/` for fast debugging.
- Run everything step by step for clarity.
- Check `notebook/analysis.ipynb` to analyze errors.
- You can extend the predictor to include pure collaborative filtering or cold-start handling.

---

## 📬 Contact

Maintained by **Yuxuan Liu**.
For feedback or collaboration ideas, feel free to reach out anytime! ☕️