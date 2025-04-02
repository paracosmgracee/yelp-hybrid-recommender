import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower()
    return text

def compute_sentiment(text):
    if not text:
        return 0.0
    text = clean_text(text)
    return TextBlob(text).sentiment.polarity

def encode_categories(categories, categories_dict):
    encoded = [0] * len(categories_dict)
    for c in categories.split(", "):
        if c in categories_dict:
            encoded[categories_dict[c]] = 1
    return encoded

def process_user_features(user):
    return [
        user.get("average_stars", 0.0),
        user.get("review_count", 0),
        np.log1p(user.get("review_count", 0)),
        user.get("fans", 0),
        user.get("useful", 0),
        user.get("funny", 0),
        user.get("cool", 0),
        abs(user.get("average_stars", 3.0) - 3.0),
        float(user.get("review_count", 0)) / max(1, 2024 - int(user.get("yelping_since", "2024")[:4]))
    ]

def process_business_features(business, categories_dict):
    categories = business.get("categories", "")
    encoded = encode_categories(categories, categories_dict)
    stars = float(business.get("stars", 0))
    review_count = int(business.get("review_count", 0))
    price_range = float(str(business.get("price_range", "1")).replace("None", "1"))
    is_open = int(business.get("is_open", 0))
    
    return [
        stars,
        review_count,
        np.log1p(review_count),
        price_range,
        is_open,
        stars * price_range,
        stars * np.log1p(review_count)
    ] + encoded

def process_review_features(review):
    text = review.get("text", "")
    return [
        review.get("useful", 0),
        len(text),
        review.get("funny", 0),
        review.get("cool", 0),
        compute_sentiment(text)
    ]