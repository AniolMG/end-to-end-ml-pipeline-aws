import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
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