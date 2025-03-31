# 🛒 Hybrid Recommendation System for Yelp Data

A hybrid recommender system built using PySpark RDDs and XGBoost regression to predict Yelp ratings with improved accuracy and personalized recommendations.

---

## 🔧 Features

- ✔️ Implemented with **PySpark RDD-only** (no DataFrame) to comply with project constraints.
- 🎯 **XGBoost regression model** trained on user/business-level features and combined with collaborative filtering.
- 🧠 Cold-start users handled with **content-based filtering** and **K-Means clustering**.
- ⏱️ Fast prediction using **FastAPI + AWS Lambda** (offline prototype).
- 📉 Final model achieved **0.95 RMSE**, a 15% improvement over baseline.

---

## 📘 Model Optimization Summary

This recommender system has gone through multiple iterations of feature engineering, algorithm tuning, and post-processing enhancements.

🔍 For full optimization breakdown, see:  
[docs/model_optimization.md](docs/model_optimization.md)

---

## 🧱 Project Structure

```
├── data/                     # Sample dataset files (truncated for GitHub)
│   ├── sample_user.json
│   ├── sample_business.json
│   ├── sample_review.json
│   ├── sample_train.csv
│   ├── sample_val.csv
├── src/                      # Core source code
│   ├── model_training.py
├── docs/                     # Optimization details
│   └── model_optimization.md
├── notebook/                 # (Optional) Visualizations and analysis
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| RMSE (Validation) | **0.9514** |
| Cold-start CTR ↑ | +18% |
| Ensemble Boost | +15% RMSE improvement |

---

## 🚀 Run Instructions

```bash
# Install requirements
pip install -r requirements.txt

# Run model training
python src/model_training.py
```

Output format (CSV):
```csv
user_id,business_id,prediction
U1,B1,4.21
U2,B3,3.88
```

---

## 🧠 Lessons Learned

- Feature engineering > model complexity
- Handling cold-start users meaningfully improves recommendation value
- Spark RDD is powerful but lower-level → debugging and memory control more important

---

## 🛠️ Tech Stack

- PySpark (RDD API only)
- XGBoost
- Python 3.6
- Jupyter Notebook