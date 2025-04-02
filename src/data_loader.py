import json
import sys
import os
from pyspark import SparkContext

# Initialize SparkContext
sc = SparkContext('local[*]', 'Hybrid Recommendation System')
sc.setLogLevel("ERROR")

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

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

def load_json_lines(filepath):
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f]

def process_business_data(input_data_folder):
    business_rdd = sc.textFile(f"{input_data_folder}/business.json") \
        .map(lambda row: json.loads(row)) \
        .map(lambda row: (
            row["business_id"],
            {
                "stars": row.get("stars", 0.0),
                "review_count": row.get("review_count", 0),
                "categories": row.get("categories", ""),
                "price_range": (row.get("attributes", {}) or {}).get("RestaurantsPriceRange2", "1"),
                "is_open": row.get("is_open", 1)
            }
        )).cache()
    return business_rdd

def process_user_data(input_data_folder):
    user_rdd = sc.textFile(f"{input_data_folder}/user.json") \
        .map(lambda row: json.loads(row)) \
        .map(lambda row: (
            row["user_id"],
            [
                row.get("average_stars", 0.0),
                row.get("review_count", 0),
                row.get("fans", 0),
                row.get("useful", 0),
                row.get("funny", 0),
                row.get("cool", 0)
            ]
        ))
    return user_rdd

def process_review_data(input_data_folder):
    review_rdd = sc.textFile(f"{input_data_folder}/review_train.json") \
        .map(lambda row: json.loads(row)) \
        .map(lambda row: (
            row["business_id"],
            (
                row.get("useful", 0),
                len(row.get("text", "")),
                row.get("funny", 0),
                row.get("cool", 0)
            )
        )) \
        .groupByKey() \
        .mapValues(lambda x: [float(sum(col)) / len(col) for col in zip(*x)])
    return review_rdd

def process_features(input_data_folder):
    business_rdd = process_business_data(input_data_folder)
    user_rdd = process_user_data(input_data_folder)
    review_rdd = process_review_data(input_data_folder)

    return user_rdd, business_rdd, review_rdd


'''
python src/data_loader.py data/full_data/user.json data/full_data/business.json data/full_data/review.json data/full_data/
'''

