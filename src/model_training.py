from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col
import sys
import json
import csv
import os
import argparse
import numpy as np
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
from feature_building import build_feature_vector


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_csv(path, has_label=True):
    user_ids, business_ids, ratings = [], [], []
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            uid, bid = row[0], row[1]
            rating = float(row[2]) if has_label else None
            user_ids.append(uid)
            business_ids.append(bid)
            ratings.append(rating)
    return user_ids, business_ids, ratings


def collaborative_filtering(user_ids, business_ids, ratings, spark):
    # Create a Spark DataFrame
    data = [(user_ids[i], business_ids[i], ratings[i]) for i in range(len(user_ids))]
    df = spark.createDataFrame(data, ["user", "business", "rating"])

    # StringIndexer for user and business columns
    indexer_user = StringIndexer(inputCol="user", outputCol="user_index")
    indexer_business = StringIndexer(inputCol="business", outputCol="business_index")

    df = indexer_user.fit(df).transform(df)
    df = indexer_business.fit(df).transform(df)

    # Collaborative filtering using ALS
    als = ALS(userCol="user_index", itemCol="business_index", ratingCol="rating", coldStartStrategy="drop", nonnegative=True)
    model = als.fit(df)
    predictions = model.transform(df)

    predictions.show()
    return predictions


def build_features(user_ids, business_ids, user_feat, business_feat, review_feat, user_clust, business_clust, categories_dict):
    features = []
    for uid, bid in tqdm(zip(user_ids, business_ids), total=len(user_ids), desc="Building feature vectors"):
        user_vec = user_feat.get(uid, {})
        business_vec = business_feat.get(bid, {})
        review_vec = review_feat.get(f"{uid}_{bid}", {})
        feature_vector = build_feature_vector(uid, bid, user_vec, business_vec, review_vec, user_clust, business_clust, categories_dict)
        features.append(feature_vector)
    return np.array(features)


def build_dl_model(input_dim):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


def train_dl_model(X_train, y_train):
    model = build_dl_model(X_train.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2, callbacks=[early_stop], verbose=1)
    return model


def train_xgb_model(X_train, y_train):
    model = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=2
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, scaler, output_dir, model_name):
    dump(model, os.path.join(output_dir, f"{model_name}.bin"))
    dump(scaler, os.path.join(output_dir, "scaler.bin"))
    print(f"✅ Saved {model_name} and scaler to {output_dir}")


if __name__ == "__main__":
    # Initialize Spark session
    spark = SparkSession.builder.appName("HybridRecommenderApp").getOrCreate()

    parser = argparse.ArgumentParser(description="Train Hybrid Recommender Models")

    parser.add_argument("--train", required=True, help="Path to training CSV file")
    parser.add_argument("--user_features", required=True, help="Path to user_features.json")
    parser.add_argument("--business_features", required=True, help="Path to business_features.json")
    parser.add_argument("--review_features", required=True, help="Path to review_features.json")
    parser.add_argument("--user_clusters", required=True, help="Path to user_clusters.json")
    parser.add_argument("--business_clusters", required=True, help="Path to business_clusters.json")
    parser.add_argument("--categories_dict", required=True, help="Path to categories_dict.json")
    parser.add_argument("--output_dir", required=True, help="Directory to save models")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    user_ids, business_ids, ratings = load_csv(args.train)
    user_features = load_json(args.user_features)
    business_features = load_json(args.business_features)
    review_features = load_json(args.review_features)
    user_clusters = load_json(args.user_clusters)
    business_clusters = load_json(args.business_clusters)
    categories_dict = load_json(args.categories_dict)

    X = build_features(user_ids, business_ids, user_features, business_features, review_features, user_clusters, business_clusters, categories_dict)
    y = np.array(ratings)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Collaborative filtering predictions
    collaborative_pred = collaborative_filtering(user_ids, business_ids, ratings, spark)

    # Combine collaborative and content-based predictions for hybrid model
    xgb_model = train_xgb_model(X_train, y_train)
    save_model(xgb_model, scaler, args.output_dir, "xgb_model")

    dl_model = train_dl_model(X_train, y_train)
    dl_model.save(os.path.join(args.output_dir, "dl_model.h5"))
    print("✅ Deep Learning model saved.")

    # Stop Spark session
    spark.stop()



'''
python src/model_training.py \
  --train data/full_data/yelp_train.csv \
  --user_features data/full_data/user_features.json \
  --business_features data/full_data/business_features.json \
  --review_features data/full_data/review_features.json \
  --user_clusters data/full_data/user_clusters.json \
  --business_clusters data/full_data/business_clusters.json \
  --categories_dict data/full_data/categories_dict.json \
  --output_dir model/
'''