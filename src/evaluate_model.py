import sys
import csv
import math
from collections import defaultdict
from tqdm import tqdm

def load_csv(path, has_label=True):
    data = {}
    try:
        with open(path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in tqdm(reader, desc=f"Loading {path}", unit="line"):  # 显示进度条
                uid, bid = row[0], row[1]
                rating = float(row[2]) if has_label else None
                data[(uid, bid)] = rating
    except FileNotFoundError:
        print(f"Error: The file {path} was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    return data

def calculate_metrics(predicted, actual):
    total_squared_error = 0
    total_absolute_error = 0
    n = 0
    error_distribution = defaultdict(int)

    for key in tqdm(predicted, desc="Calculating metrics", unit="key"):  # 显示进度条
        if key in actual:
            pred = predicted[key]
            truth = actual[key]
            error = abs(pred - truth)
            total_squared_error += (pred - truth) ** 2
            total_absolute_error += error
            n += 1

            if error < 1:
                error_distribution['[0,1)'] += 1
            elif error < 2:
                error_distribution['[1,2)'] += 1
            elif error < 3:
                error_distribution['[2,3)'] += 1
            elif error < 4:
                error_distribution['[3,4)'] += 1
            else:
                error_distribution['[4,∞)'] += 1

    rmse = math.sqrt(total_squared_error / n) if n != 0 else 0
    mae = total_absolute_error / n if n != 0 else 0
    return rmse, mae, error_distribution

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate_model.py <prediction_csv> <ground_truth_csv>")
        sys.exit(1)

    prediction_path = sys.argv[1]
    truth_path = sys.argv[2]

    predicted = load_csv(prediction_path, has_label=True)
    actual = load_csv(truth_path, has_label=True)

    rmse, mae, error_dist = calculate_metrics(predicted, actual)

    print(f"📊 Evaluation Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print("Error Distribution:")
    for k in sorted(error_dist.keys()):
        print(f"  {k}: {error_dist[k]}")


'''
python src/evaluate_model.py output/prediction_full.csv data/full_data/yelp_val.csv
'''