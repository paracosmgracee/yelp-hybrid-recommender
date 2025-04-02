import numpy as np
from textblob import TextBlob

def build_feature_vector(
    user_id, business_id,
    user_features: dict,
    business_features: dict,
    review_features: dict,
    user_clusters: dict,
    business_clusters: dict,
    categories_dict: dict,
    use_sentiment: bool = True,
    use_interactions: bool = True
) -> list:
    """
    Build a feature vector for a given user and business, combining all available features.
    
    Args:
        user_id (str): The user ID.
        business_id (str): The business ID.
        user_features (dict): User-level features.
        business_features (dict): Business-level features.
        review_features (dict): Review-level features.
        user_clusters (dict): User clustering labels.
        business_clusters (dict): Business clustering labels.
        categories_dict (dict): One-hot category encoding map.
        use_sentiment (bool): Whether to include sentiment features.
        use_interactions (bool): Whether to include feature interaction terms.

    Returns:
        list: A combined feature vector.
    """

    # --- User features ---
    user_feat = user_features.get(user_id, {})
    user_avg_stars = user_feat.get("average_stars", 0.0)
    user_review_count = user_feat.get("review_count", 0)
    user_fans = user_feat.get("fans", 0)
    user_useful = user_feat.get("useful", 0)
    user_funny = user_feat.get("funny", 0)
    user_cool = user_feat.get("cool", 0)

    # --- Business features ---
    business_feat = business_features.get(business_id, {})
    business_stars = business_feat.get("stars", 0.0)
    business_review_count = business_feat.get("review_count", 0)
    is_open = business_feat.get("is_open", 1)
    price_range = business_feat.get("price_range", 1)

    # Category one-hot encoding
    category_vector = [0] * len(categories_dict)
    if "categories" in business_feat and business_feat["categories"]:
        categories = business_feat["categories"].split(", ")
        for cat in categories:
            if cat in categories_dict:
                idx = categories_dict[cat]
                category_vector[idx] = 1

    # --- Review features ---
    review_key = f"{user_id}_{business_id}"
    review_feat = review_features.get(review_key, {})
    review_useful = review_feat.get("useful", 0)
    review_funny = review_feat.get("funny", 0)
    review_cool = review_feat.get("cool", 0)
    review_text_length = review_feat.get("text_length", 0)
    review_text = review_feat.get("text", "")

    # --- Cluster features ---
    user_cluster = user_clusters.get(user_id, -1)
    business_cluster = business_clusters.get(business_id, -1)

    # --- Sentiment features ---
    user_sentiment, review_sentiment = 0.0, 0.0
    if use_sentiment:
        if isinstance(review_text, str) and len(review_text) > 3:
            review_sentiment = TextBlob(review_text).sentiment.polarity
        user_sentiment = review_sentiment  # could be extended to user's all reviews

    # --- Feature interactions ---
    stars_price_interaction = business_stars * price_range if use_interactions else 0
    stars_diff = abs(user_avg_stars - business_stars) if use_interactions else 0
    rating_interaction = user_avg_stars * business_stars if use_interactions else 0

    # --- Final vector ---
    feature_vector = [
        user_avg_stars, user_review_count, user_fans, user_useful, user_funny, user_cool,
        business_stars, business_review_count, is_open, price_range,
        review_useful, review_funny, review_cool, review_text_length,
        user_cluster, business_cluster,
        user_sentiment, review_sentiment,
        stars_price_interaction, stars_diff, rating_interaction
    ] + category_vector

    return feature_vector