import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
import joblib
import json
import boto3

# --- Parse hyperparameters ---
parser = argparse.ArgumentParser()
parser.add_argument("--max_depth", type=int, default=8)
parser.add_argument("--eta", type=float, default=0.3)
parser.add_argument("--objective", type=str, default="binary:logistic")
parser.add_argument("--num_round", type=int, default=200)
parser.add_argument("--bucket", type=str)
args = parser.parse_args()

# --- SageMaker paths ---
input_dir = "/opt/ml/input/data/train"
output_dir = "/opt/ml/model"

# --- Load data ---
data_path = os.path.join(input_dir, "titanic_data.csv")
df = pd.read_csv(data_path)
df = df.dropna(subset=['Age'])
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
X = df[['Age', 'Sex', 'Pclass']]
y = df['Survived']

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5, stratify=y
)

# --- Train model ---
model = XGBClassifier(
    n_estimators=args.num_round,
    max_depth=args.max_depth,
    learning_rate=args.eta,
    objective=args.objective,
    random_state=5,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# --- Save model to /opt/ml/model ---
model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "titanic_model.joblib")

joblib.dump(model, model_path)
print(f"✅ Model saved to {model_path}")

"""# --- Evaluate ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")

# --- Save model ---
os.makedirs(output_dir, exist_ok=True)
joblib.dump(model, os.path.join(output_dir, "titanic_model.joblib"))

# --- Save metrics in SageMaker-compatible format ---
metrics = {
    "metrics": {
        "accuracy": {"value": accuracy},
        "f1": {"value": f1},
        "precision": {"value": precision},
        "recall": {"value": recall},
        "auc": {"value": auc}
    }
}

# Save locally and upload to S3
metrics_dir = "metrics"
os.makedirs(metrics_dir, exist_ok=True)
metrics_path = os.path.join(metrics_dir, "titanic_metrics.json")

with open(metrics_path, "w") as f:
    json.dump(metrics, f)

s3 = boto3.client("s3")
s3.upload_file(metrics_path, args.bucket, "metrics/titanic_metrics.json")"""
