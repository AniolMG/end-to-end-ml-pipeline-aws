import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import json
import argparse
import tarfile

parser = argparse.ArgumentParser()
parser.add_argument("--input-model", type=str, required=True)
parser.add_argument("--input-data", type=str, required=True)
parser.add_argument("--output-metrics", type=str, required=True)
args = parser.parse_args()

# Load dataset
df = pd.read_csv(args.input_data)
df = df.dropna(subset=["Age"])
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
X = df[["Age", "Sex", "Pclass"]]
y = df["Survived"]

# Unpack model.tar.gz if present
model_dir = args.input_model
tar_path = os.path.join(model_dir, "model.tar.gz")
if os.path.exists(tar_path):
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=model_dir)
    print(f"✅ Extracted model artifacts to {model_dir}")
else:
    print("⚠️ model.tar.gz not found in model directory")


# Load trained model
model_path = os.path.join(args.input_model, "titanic_model.joblib") 
model = joblib.load(model_path)

# Compute predictions & metrics
y_pred = model.predict(X)
metrics = {
    "binary_classification_metrics": {
        "accuracy": {
            "value": accuracy_score(y, y_pred),
            # "standard_deviation": "NaN"
        },
        "f1": {
            "value": f1_score(y, y_pred),
            # "standard_deviation": "NaN"
        },
        "precision": {
            "value": precision_score(y, y_pred),
            # "standard_deviation": "NaN"
        },
        "recall": {
            "value": recall_score(y, y_pred),
            # "standard_deviation": "NaN"
        },
        "auc": {
            "value": roc_auc_score(y, y_pred),
            # "standard_deviation": "NaN"
        }
    }
}

# Save metrics to output directory
output_path = os.path.join(args.output_metrics, "metrics.json")
with open(output_path, "w") as f:
    json.dump(metrics, f)
print(f"✅ Metrics written to {output_path}")