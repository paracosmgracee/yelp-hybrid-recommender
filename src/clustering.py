import json
import sys
import os
import numpy as np
from sklearn.cluster import KMeans
from pyspark import SparkContext
import math

sc = SparkContext("local[*]", "Hybrid Clustering")
sc.setLogLevel("ERROR")

def save_json(obj, path):
    def convert_types(o):
        if isinstance(o, dict):
            return {k: convert_types(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [convert_types(v) for v in o]
        elif isinstance(o, np.integer):
            return int(o)
        else:
            return o
    with open(path, "w") as f:
        json.dump(convert_types(obj), f, indent=2)

def process_business_features(business_rdd, top_n=20):
    top_categories = business_rdd.flatMap(lambda x: x[1]["categories"].split(", ") if x[1]["categories"] else []) \
                                 .map(lambda cat: (cat, 1)) \
                                 .reduceByKey(lambda a, b: a + b) \
                                 .takeOrdered(top_n, key=lambda x: -x[1])
    categories_dict = {cat: idx for idx, (cat, _) in enumerate(top_categories)}
    bc_categories_dict = sc.broadcast(categories_dict)

    def encode_features(x):
        info = x[1]
        cat_vec = [0] * len(categories_dict)
        for c in info["categories"].split(", ") if info["categories"] else []:
            if c in bc_categories_dict.value:
                cat_vec[bc_categories_dict.value[c]] = 1
        features = [
            float(info.get("stars", 0)),
            float(info.get("review_count", 0)),
            math.log1p(float(info.get("review_count", 0))),
            float(str(info.get("price_range", 1)).replace("None", "1")),
            int(info.get("is_open", 0)),
        ] + cat_vec
        return (x[0], features)

    feature_rdd = business_rdd.map(encode_features)
    return feature_rdd, categories_dict

def apply_kmeans(rdd, k=8):
    features = rdd.map(lambda x: x[1]).collect()
    ids = rdd.map(lambda x: x[0]).collect()
    model = KMeans(n_clusters=k, random_state=42).fit(features)
    return dict(zip(ids, model.labels_))

def load_data(data_dir):
    def read_json(path):
        return sc.textFile(path).map(json.loads)

    business_rdd = read_json(os.path.join(data_dir, "business.json")) \
        .map(lambda row: (row["business_id"], {
            "stars": row.get("stars", 0),
            "review_count": row.get("review_count", 0),
            "categories": row.get("categories", ""),
            "price_range": (row.get("attributes", {}) or {}).get("RestaurantsPriceRange2", "1"),
            "is_open": row.get("is_open", 0)
        }))

    user_rdd = read_json(os.path.join(data_dir, "user.json")) \
        .map(lambda row: (row["user_id"], [
            row.get("average_stars", 0.0),
            row.get("review_count", 0),
            math.log1p(row.get("review_count", 0)),
            row.get("fans", 0),
            row.get("useful", 0),
            row.get("funny", 0),
            row.get("cool", 0),
            abs(row.get("average_stars", 3.0) - 3.0),
            float(row.get("review_count", 0)) / max(1, 2024 - int(row.get("yelping_since", "2024")[:4]))
        ]))

    return user_rdd, business_rdd

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clustering.py <output_dir>")
        sys.exit(1)

    output_dir = sys.argv[1]
    user_rdd, business_rdd = load_data("data/full_data")

    business_features, categories_dict = process_business_features(business_rdd)
    business_clusters = apply_kmeans(business_features)
    user_clusters = apply_kmeans(user_rdd)

    save_json(business_clusters, os.path.join(output_dir, "business_clusters.json"))
    save_json(user_clusters, os.path.join(output_dir, "user_clusters.json"))
    save_json(categories_dict, os.path.join(output_dir, "categories_dict.json"))

    print(f"✅ Clustering completed and saved to {output_dir}")
    sc.stop()


'''
python src/clustering.py data/full_data/
'''