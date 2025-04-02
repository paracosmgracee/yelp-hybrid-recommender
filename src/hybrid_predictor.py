import os
import json
import csv
import argparse
import numpy as np
from joblib import load
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from feature_building import build_feature_vector

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_val_data(val_path):
    user_ids, business_ids = [], []
    with open(val_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            user_ids.append(row[0])
            business_ids.append(row[1])
    return user_ids, business_ids

def build_features(user_ids, business_ids, user_feat, business_feat, review_feat, user_clust, business_clust, categories_dict):
    features = []
    for uid, bid in tqdm(zip(user_ids, business_ids), total=len(user_ids), desc="Building prediction features"):
        user_vec = user_feat.get(uid, {})
        business_vec = business_feat.get(bid, {})
        review_vec = review_feat.get(f"{uid}_{bid}", {})
        feature_vector = build_feature_vector(uid, bid, user_vec, business_vec, review_vec, user_clust, business_clust, categories_dict)
        features.append(feature_vector)
    return np.array(features)

def predict_and_save(xgb_model, dl_model, scaler, X, user_ids, business_ids, output_path):
    X_scaled = scaler.transform(X)
    xgb_preds = xgb_model.predict(X_scaled)
    dl_preds = dl_model.predict(X_scaled).reshape(-1)
    hybrid_preds = (xgb_preds + dl_preds) / 2

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "business_id", "prediction"])
        for uid, bid, pred in zip(user_ids, business_ids, hybrid_preds):
            writer.writerow([uid, bid, pred])
    print(f"✅ Saved predictions to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hybrid predictor on validation data")
    parser.add_argument("--val", required=True, help="Path to validation CSV file")
    parser.add_argument("--user_features", required=True)
    parser.add_argument("--business_features", required=True)
    parser.add_argument("--review_features", required=True)
    parser.add_argument("--user_clusters", required=True)
    parser.add_argument("--business_clusters", required=True)
    parser.add_argument("--categories_dict", required=True)
    parser.add_argument("--model_dir", required=True, help="Directory containing trained models")
    parser.add_argument("--output", required=True, help="Output CSV file for predictions")
    args = parser.parse_args()

    user_ids, business_ids = load_val_data(args.val)
    user_features = load_json(args.user_features)
    business_features = load_json(args.business_features)
    review_features = load_json(args.review_features)
    user_clusters = load_json(args.user_clusters)
    business_clusters = load_json(args.business_clusters)
    categories_dict = load_json(args.categories_dict)

    X = build_features(user_ids, business_ids, user_features, business_features, review_features, user_clusters, business_clusters, categories_dict)

    xgb_model = load(os.path.join(args.model_dir, "xgb_model.bin"))
    scaler = load(os.path.join(args.model_dir, "scaler.bin"))
    dl_model = load_model(os.path.join(args.model_dir, "dl_model.h5"))

    predict_and_save(xgb_model, dl_model, scaler, X, user_ids, business_ids, args.output)


'''
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
'''