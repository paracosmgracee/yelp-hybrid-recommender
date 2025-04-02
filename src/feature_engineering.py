import json
import sys
import os

def load_json_lines(filepath):
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f]

def extract_user_features(users):
    features = {}
    for user in users:
        user_id = user["user_id"]
        features[user_id] = {
            "average_stars": user.get("average_stars", 0),
            "review_count": user.get("review_count", 0),
            "fans": user.get("fans", 0),
            "useful": user.get("useful", 0),
            "funny": user.get("funny", 0),
            "cool": user.get("cool", 0)
        }
    return features

def extract_business_features(businesses):
    features = {}
    for business in businesses:
        business_id = business["business_id"]
        features[business_id] = {
            "stars": business.get("stars", 0),
            "review_count": business.get("review_count", 0),
            "is_open": business.get("is_open", 1),
            "categories": business.get("categories", "")
        }
    return features

def extract_review_features(reviews):
    features = {}
    for review in reviews:
        user_id = review["user_id"]
        business_id = review["business_id"]
        key = (user_id, business_id)
        features[key] = {
            "useful": review.get("useful", 0),
            "funny": review.get("funny", 0),
            "cool": review.get("cool", 0),
            "text_length": len(review.get("text", ""))
        }
    return features

def save_feature_json(obj, path):
    def convert_keys(d):
        if isinstance(d, dict):
            return {str(k): convert_keys(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [convert_keys(i) for i in d]
        else:
            return d

    obj = convert_keys(obj)

    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python feature_engineering.py <user_json> <business_json> <review_json> <output_dir>")
        sys.exit(1)

    user_path = sys.argv[1]
    business_path = sys.argv[2]
    review_path = sys.argv[3]
    output_dir = sys.argv[4]

    users = load_json_lines(user_path)
    businesses = load_json_lines(business_path)
    reviews = load_json_lines(review_path)

    user_features = extract_user_features(users)
    business_features = extract_business_features(businesses)
    review_features = extract_review_features(reviews)

    # Save feature files
    save_feature_json(user_features, os.path.join(output_dir, "user_features.json"))
    save_feature_json(business_features, os.path.join(output_dir, "business_features.json"))
    save_feature_json(review_features, os.path.join(output_dir, "review_features.json"))

    print(f"✅ Extracted user, business, and review features and saved to {output_dir}")


'''
python src/feature_engineering.py \
  data/full_data/user.json \
  data/full_data/business.json \
  data/full_data/review_train.json \
  data/full_data/
'''