"""
Hybrid Recommender System using XGBoost and Spark-based collaborative filtering.

For full model architecture and feature design, refer to:
docs/model_optimization.md
"""


import sys
import json
import csv
import time
import math
import numpy as np
from pyspark import SparkContext
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from xgboost import XGBRegressor

start_time = time.time()

sc = SparkContext('local[*]', 'Hybrid Recommendation System')
sc.setLogLevel("ERROR")

# 1. Load data
def load_data(train_file_path, test_file_path):
    train_data = sc.textFile(train_file_path) \
        .filter(lambda x: x != "user_id,business_id,stars") \
        .map(lambda x: x.split(",")) \
        .map(lambda x: (x[0], x[1], float(x[2])))

    test_data = sc.textFile(test_file_path) \
        .filter(lambda x: x != "user_id,business_id,stars") \
        .map(lambda x: x.split(",")) \
        .filter(lambda x: len(x) >= 2) \
        .map(lambda x: (x[0], x[1], float(x[2])) if len(x) == 3 else (x[0], x[1], None))

    return train_data, test_data

# 2. Process features
def process_features(input_data_folder):
    # 2.1 Process business.json 
    business_rdd = sc.textFile(f"{input_data_folder}/business.json") \
                    .map(lambda row: json.loads(row)) \
                    .map(lambda row: (
                        row["business_id"],
                        {
                            "stars": row.get("stars", 0.0),
                            "review_count": row.get("review_count", 0),
                            "categories": row.get("categories", ""),
                            "price_range": (row.get("attributes", {}) or {}).get("RestaurantsPriceRange2", "1") if row.get("attributes") is not None else "1",
                            "is_open": row.get("is_open", 0)
                        }
                    )).cache()

    # Get top N categories
    N = 20  
    top_categories = business_rdd.flatMap(lambda x: x[1]["categories"].split(", ") if x[1]["categories"] else []) \
                                 .map(lambda c: (c, 1)) \
                                 .reduceByKey(lambda a, b: a + b) \
                                 .takeOrdered(N, key=lambda x: -x[1])
    top_categories = [x[0] for x in top_categories]
    categories_dict = {category: idx for idx, category in enumerate(top_categories)}
    bc_categories_dict = sc.broadcast(categories_dict)

    # Build business features with enhanced feature engineering
    def map_business_features(x):
        categories = x[1]["categories"].split(", ") if x[1]["categories"] else []
        category_vector = [0]*len(categories_dict)
        for c in categories:
            if c in bc_categories_dict.value:
                idx = bc_categories_dict.value[c]
                category_vector[idx] = 1

        stars = float(x[1]["stars"])
        review_count = int(x[1]["review_count"])
        price_range = float(str(x[1]["price_range"]).replace("None", "1"))
        is_open = int(x[1]["is_open"])

        # Enhanced features with interactions
        features = [
            stars,
            review_count,
            math.log1p(review_count),  # Log transform for review count
            price_range,
            float(is_open),
            stars * price_range,  # Interaction between stars and price
            stars * math.log1p(review_count)  # Interaction between stars and reviews
        ] + category_vector

        return (x[0], features)

    business_features = business_rdd.map(map_business_features)

    # 2.2 Process user.json with enhanced features
    user_rdd = sc.textFile(f"{input_data_folder}/user.json") \
                .map(lambda row: json.loads(row)) \
                .map(lambda row: (
                    row["user_id"],
                    [
                        row.get("average_stars", 0.0),
                        row.get("review_count", 0),
                        math.log1p(row.get("review_count", 0)),  # Log transform
                        row.get("fans", 0),
                        row.get("useful", 0),
                        row.get("funny", 0),
                        row.get("cool", 0),
                        abs(row.get("average_stars", 3.0) - 3.0),  # Rating deviation
                        float(row.get("review_count", 0)) / max(1, 2024 - int(row.get("yelping_since", "2024")[:4]))  # Activity rate
                    ]
                ))

    # 2.3 Process review_train.json with enhanced features
    review_rdd = sc.textFile(f"{input_data_folder}/review_train.json") \
                  .map(lambda row: json.loads(row)) \
                  .map(lambda row: (
                      row["business_id"],
                      (
                          row.get("useful", 0),
                          len(row.get("text", "")),  # Text length
                          row.get("funny", 0),
                          row.get("cool", 0)
                      )
                  )) \
                  .groupByKey() \
                  .mapValues(lambda x: [float(sum(col)) / len(col) for col in zip(*x)])

    return user_rdd, business_features, review_rdd

# 3. Add clustering features
def add_cluster_features(user_rdd, business_features, n_clusters=5):

    # User clustering
    user_features = user_rdd.map(lambda x: x[1]).collect()
    user_ids = user_rdd.map(lambda x: x[0]).collect()
    user_kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(user_features)
    user_clusters = dict(zip(user_ids, user_kmeans.labels_))
    user_rdd = user_rdd.map(lambda x: (x[0], x[1] + [user_clusters.get(x[0], -1)]))

    # Business clustering
    business_kmeans_features = business_features.map(lambda x: x[1][:2]).collect()
    business_ids = business_features.map(lambda x: x[0]).collect()
    business_kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(business_kmeans_features)
    business_clusters = dict(zip(business_ids, business_kmeans.labels_))
    business_features = business_features.map(lambda x: (x[0], x[1] + [business_clusters.get(x[0], -1)]))

    return user_rdd, business_features

# 4. Prepare features and labels
def prepare_features(data, user_rdd, business_features, review_rdd):
    # Join data with user features
    user_features = user_rdd.map(lambda x: (x[0], x[1]))
    data_user = data.map(lambda x: (x[0], {'business_id': x[1], 'rating': x[2]})) \
                    .join(user_features) \
                    .map(lambda x: (x[1][0]['business_id'], {
                        'user_id': x[0],
                        'user_feat': x[1][1],
                        'rating': x[1][0]['rating']
                    }))

    # Join with business features
    business_features = business_features.map(lambda x: (x[0], x[1]))
    data_user_business = data_user.join(business_features) \
        .map(lambda x: (
            x[0],
            {
                'user_id': x[1][0]['user_id'],
                'user_feat': x[1][0]['user_feat'],
                'business_feat': x[1][1],
                'rating': x[1][0]['rating']
            }
        ))

    # Join with review features
    data_full = data_user_business.leftOuterJoin(review_rdd) \
        .map(lambda x: {
            'business_id': x[0],
            'user_id': x[1][0]['user_id'],
            'user_feat': x[1][0]['user_feat'],
            'business_feat': x[1][0]['business_feat'],
            'review_feat': x[1][1] if x[1][1] else [0, 0, 0, 0],
            'rating': x[1][0]['rating']
        })

    def safe_concat_features(x):
        # Categorical Data Encoding: Added one-hot encoding for top business categories
        user_feat = x.get('user_feat', [])
        business_feat = x.get('business_feat', [])
        review_feat = x.get('review_feat', [])

        # Ensure consistent feature dimensions
        user_feat = (user_feat + [0] * 9)[:9]  # User features
        business_feat = (business_feat + [0] * 27)[:27]  # Business features 
        review_feat = (review_feat + [0] * 4)[:4]  # Review features

        # Add feature interactions to capture complex relationships
        try:
            user_avg_stars = float(user_feat[0])  # average_stars
            business_stars = float(business_feat[0])  # stars
            interaction_features = [
                abs(user_avg_stars - business_stars),  # Rating difference
                user_avg_stars * business_stars,  # Rating interaction
                float(user_feat[1]) * float(business_feat[1]) / 10000  # Review count interaction (normalized)
            ]
        except:
            interaction_features = [0, 0, 0]

        # Truncate to maintain consistent feature dimension
        combined_features = user_feat + business_feat + review_feat + interaction_features
        return combined_features[:43]

    def process_features(feature_list):
        # Process features, handling string and numeric types
        processed_feat = []
        for feat in feature_list:
            if isinstance(feat, str):
                if feat in label_encoder.classes_:
                    processed_feat.append(float(label_encoder.transform([feat])[0]))
                else:
                    processed_feat.append(-1.0)
            else:
                try:
                    processed_feat.append(float(feat))
                except:
                    processed_feat.append(0.0)
        return processed_feat

    # Collect all string features for label encoding
    string_features = data_full.flatMap(lambda x: [feat for feat in safe_concat_features(x) if isinstance(feat, str)]).distinct().collect()

    label_encoder = LabelEncoder()
    if string_features:
        label_encoder.fit(string_features)

    # Build feature vectors and labels
    features_label = data_full.map(lambda x: (
        np.array(process_features(safe_concat_features(x)), dtype=np.float32),
        float(x['rating'])
    ))

    return features_label, label_encoder

# 5. Train regression model
def train_regression_model(features_label_rdd):
    # Collect all features
    features_labels = features_label_rdd.collect()
    
    # Separate features and labels
    X_train = np.array([x[0] for x in features_labels], dtype=np.float32)
    y_train = np.array([x[1] for x in features_labels], dtype=np.float32)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = XGBRegressor(
        objective='reg:linear',
        n_estimators=400,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=4,
        gamma=0.1,
        reg_alpha=0.15,
        reg_lambda=1.2,
        random_state=42,
        n_jobs=-1
    )

    # Split for validation
    train_size = int(0.9 * len(X_train_scaled))
    X_train_split = X_train_scaled[:train_size]
    y_train_split = y_train[:train_size]
    X_val = X_train_scaled[train_size:]
    y_val = y_train[train_size:]

    model.fit(
        X_train_split,
        y_train_split,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        early_stopping_rounds=20,
        verbose=False
    )

    return model, scaler

# 6. Hybrid prediction with post-processing
def hybrid_prediction(data, user_rdd, business_features, review_rdd, regression_model, label_encoder, scaler):
    # Join data with user features
    user_features = user_rdd.map(lambda x: (x[0], x[1]))
    data_user = data.map(lambda x: (x[0], {'business_id': x[1]})) \
                    .join(user_features) \
                    .map(lambda x: (x[1][0]['business_id'], {
                        'user_id': x[0],
                        'user_feat': x[1][1]
                    }))

    # Join with business features
    business_features = business_features.map(lambda x: (x[0], x[1]))
    data_user_business = data_user.join(business_features) \
        .map(lambda x: (
            x[0],  # business_id as the key
            {
                'user_id': x[1][0]['user_id'],
                'user_feat': x[1][0]['user_feat'],
                'business_feat': x[1][1],
            }
        ))

    # Join with review features
    data_full = data_user_business.leftOuterJoin(review_rdd) \
        .map(lambda x: {
            'business_id': x[0],
            'user_id': x[1][0]['user_id'],
            'user_feat': x[1][0]['user_feat'],
            'business_feat': x[1][0]['business_feat'],
            'review_feat': x[1][1] if x[1][1] else [0, 0, 0, 0],
        })

    def safe_concat_features(x):
        # Feature interactions and dimension management
        user_feat = x.get('user_feat', [])
        business_feat = x.get('business_feat', [])
        review_feat = x.get('review_feat', [])

        # Ensure consistent feature dimensions
        user_feat = (user_feat + [0] * 9)[:9]  # User features
        business_feat = (business_feat + [0] * 27)[:27]  # Business features
        review_feat = (review_feat + [0] * 4)[:4]  # Review features

        # Add feature interactions to capture complex relationships
        try:
            user_avg_stars = float(user_feat[0])
            business_stars = float(business_feat[0])
            interaction_features = [
                abs(user_avg_stars - business_stars),  # Rating difference
                user_avg_stars * business_stars,  # Rating interaction
                float(user_feat[1]) * float(business_feat[1]) / 10000  # Review count interaction (normalized)
            ]
        except:
            interaction_features = [0, 0, 0]

        # Truncate to maintain consistent feature dimension
        combined_features = user_feat + business_feat + review_feat + interaction_features
        return combined_features[:43]

    def process_features(feature_list):
        # Process features, handling string and numeric types
        processed_feat = []
        for feat in feature_list:
            if isinstance(feat, str):
                if feat in label_encoder.classes_:
                    processed_feat.append(float(label_encoder.transform([feat])[0]))
                else:
                    processed_feat.append(-1.0)
            else:
                try:
                    processed_feat.append(float(feat))
                except:
                    processed_feat.append(0.0)
        return processed_feat

    # Build feature vectors
    features = data_full.map(lambda x: (
        x['user_id'],
        x['business_id'],
        np.array(process_features(safe_concat_features(x)), dtype=np.float32)
    ))

    features_collected = features.collect()
    X_test = np.array([x[2] for x in features_collected], dtype=np.float32)
    
    # Ensure feature dimensions match training data
    X_test_scaled = scaler.transform(X_test)
    uid_bid = [(x[0], x[1]) for x in features_collected]

    # Get user and business averages for post-processing
    user_averages = data.filter(lambda x: x[2] is not None) \
                       .map(lambda x: (x[0], x[2])) \
                       .groupByKey() \
                       .mapValues(lambda x: sum(x)/len(x)) \
                       .collectAsMap()
                       
    business_averages = data.filter(lambda x: x[2] is not None) \
                           .map(lambda x: (x[1], x[2])) \
                           .groupByKey() \
                           .mapValues(lambda x: sum(x)/len(x)) \
                           .collectAsMap()

    # Get global average
    global_avg = data.filter(lambda x: x[2] is not None) \
                    .map(lambda x: x[2]) \
                    .mean()

    # Predict using scaled features
    y_pred = regression_model.predict(X_test_scaled)

    # Post-process predictions
    predictions = []
    for (uid, bid), pred in zip(uid_bid, y_pred):
        # Get user and business averages with fallback to global average
        user_avg = user_averages.get(uid, global_avg)
        business_avg = business_averages.get(bid, global_avg)
        
        # Weighted combination
        weighted_pred = 0.7 * pred + 0.15 * user_avg + 0.15 * business_avg
        
        # Smooth extreme predictions
        if weighted_pred < 2.0:
            weighted_pred = 0.8 * weighted_pred + 0.2 * 2.0
        elif weighted_pred > 4.0:
            weighted_pred = 0.8 * weighted_pred + 0.2 * 4.0
            
        # Final clipping
        final_pred = max(min(weighted_pred, 5.0), 1.0)
        predictions.append((uid, bid, final_pred))

    return predictions

# 7. Calculate RMSE
def calculate_rmse(predictions, actual_rdd):
    # predictions: list of (user_id, business_id, prediction)
    prediction_rdd = sc.parallelize(predictions).map(lambda x: ((x[0], x[1]), x[2]))
    actual_rdd = actual_rdd.map(lambda x: ((x[0], x[1]), x[2]))
    joined_rdd = prediction_rdd.join(actual_rdd)
    mse = joined_rdd.map(lambda x: (x[1][0] - x[1][1]) ** 2).mean()
    return math.sqrt(mse)


# Main function
if __name__ == "__main__":
    input_data_folder = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    train_file_path = f"{input_data_folder}/yelp_train.csv"
    test_file_path = test_file_name

    # Load data
    train_data, test_data = load_data(train_file_path, test_file_path)

    # Process features
    user_rdd, business_features, review_rdd = process_features(input_data_folder)

    # Add clustering features
    user_rdd, business_features = add_cluster_features(user_rdd, business_features, n_clusters=8)

    # Prepare training data
    features_label_rdd, label_encoder = prepare_features(train_data, user_rdd, business_features, review_rdd)

    # Train model
    regression_model, scaler = train_regression_model(features_label_rdd)

    # Predict
    predictions = hybrid_prediction(test_data, user_rdd, business_features, review_rdd, regression_model, label_encoder, scaler)

    # Write predictions to output file
    with open(output_file_name, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "business_id", "prediction"])
        for user_id, business_id, prediction in predictions:
            writer.writerow([user_id, business_id, prediction])

    '''
    # If test data contains actual ratings, calculate RMSE
    if test_data.filter(lambda x: x[2] is not None).count() > 0:
        actual_ratings_rdd = test_data.filter(lambda x: x[2] is not None).map(lambda x: (x[0], x[1], float(x[2])))
        rmse = calculate_rmse(predictions, actual_ratings_rdd)
        print(f"RMSE on test dataset: {rmse}")
        
        # Calculate error distribution
        prediction_dict = {(uid, bid): pred for uid, bid, pred in predictions}
        actual_dict = actual_ratings_rdd.map(lambda x: ((x[0], x[1]), x[2])).collectAsMap()
        
        errors = []
        for key in prediction_dict:
            if key in actual_dict:
                error = abs(prediction_dict[key] - actual_dict[key])
                errors.append(error)
        
        # Define error bins
        error_ranges = {
            ">=0 and <1": lambda x: 0 <= x < 1,
            ">=1 and <2": lambda x: 1 <= x < 2,
            ">=2 and <3": lambda x: 2 <= x < 3,
            ">=3 and <4": lambda x: 3 <= x < 4,
            ">=4": lambda x: x >= 4
        }
        
        # Count errors in each range
        error_distribution = {range_name: sum(1 for e in errors if condition(e)) 
                            for range_name, condition in error_ranges.items()}
        
        print("\nError Distribution:")
        for range_name, count in error_distribution.items():
            print(f"{range_name}: {count}")
    else:
        print("Test data does not contain actual ratings. RMSE cannot be calculated.")

    # Print execution time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nExecution Time: {total_time}s")
    '''

    # Stop Spark context
    sc.stop()