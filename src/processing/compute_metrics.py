import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import json
import argparse
import tarfile

# --- Parse arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--input-model", type=str, required=True)
parser.add_argument("--input-data", type=str, required=True)
parser.add_argument("--output-metrics", type=str, required=True)
parser.add_argument("--target_column", type=str)
parser.add_argument("--feature_columns", type=str)
parser.add_argument("--categorical_columns", type=str)
args = parser.parse_args()

# --- Load dataset ---
df = pd.read_csv(args.input_data)

# --- Feature and categorical lists ---
feature_cols = args.feature_columns.split(",")
categorical_cols = args.categorical_columns.split(",") if args.categorical_columns else []

# --- Drop NA for feature columns ---
df = df.dropna(subset=feature_cols)

# --- Unpack model.tar.gz if present (model + encoder if present) ---
tar_path = os.path.join(args.input_model, "model.tar.gz")
if os.path.exists(tar_path):
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=args.input_model)
    print(f"✅ Extracted model artifacts to {args.input_model}")

# --- Load encoders ---
encoders_path = os.path.join(args.input_model, "encoders.joblib")
if os.path.exists(encoders_path):
    encoders = joblib.load(encoders_path)
    for col, le in encoders.items():
        # Transform the column using the same encoder as training
        df[col] = le.transform(df[col])
else:
    print("⚠️ encoders.joblib not found. Assuming no categorical columns.")
    encoders = {}

# --- Prepare X and y ---
X = df[feature_cols]
y = df[args.target_column]

# --- Load trained model ---
model_path = os.path.join(args.input_model, "model.joblib")
model = joblib.load(model_path)

# --- Compute predictions & metrics ---
y_pred = model.predict(X)
metrics = {
    "binary_classification_metrics": {
        "accuracy": {"value": accuracy_score(y, y_pred)},
        "f1": {"value": f1_score(y, y_pred)},
        "precision": {"value": precision_score(y, y_pred)},
        "recall": {"value": recall_score(y, y_pred)},
        "auc": {"value": roc_auc_score(y, y_pred)},
    }
}

# --- Save metrics ---
os.makedirs(args.output_metrics, exist_ok=True)
output_path = os.path.join(args.output_metrics, "metrics.json")
with open(output_path, "w") as f:
    json.dump(metrics, f)

print(f"✅ Metrics written to {output_path}")