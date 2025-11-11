import os
import argparse
import pandas as pd
from xgboost import XGBClassifier
import joblib

# --- Parse hyperparameters ---
parser = argparse.ArgumentParser()
parser.add_argument("--max_depth", type=int, default=8)
parser.add_argument("--eta", type=float, default=0.3)
parser.add_argument("--objective", type=str, default="binary:logistic")
parser.add_argument("--num_round", type=int, default=200)
parser.add_argument("--bucket", type=str)
parser.add_argument("--train_file", type=str, required=True)
args = parser.parse_args()

# --- SageMaker paths ---
input_dir = "/opt/ml/input/data/train"
output_dir = "/opt/ml/model"

# --- Load data ---
data_path = os.path.join(input_dir, args.train_file)
df = pd.read_csv(data_path)
df = df.dropna(subset=['Age'])
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
X = df[['Age', 'Sex', 'Pclass']]
y = df['Survived']

# --- Train model ---
model = XGBClassifier(
    n_estimators=args.num_round,
    max_depth=args.max_depth,
    learning_rate=args.eta,
    objective=args.objective,
    random_state=5,
    eval_metric="logloss"
)
model.fit(X, y)

# --- Save model to /opt/ml/model ---
model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "titanic_model.joblib")

joblib.dump(model, model_path)
print(f"✅ Model saved to {model_path}")