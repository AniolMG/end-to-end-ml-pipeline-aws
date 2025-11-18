import os
import argparse
import pandas as pd
from xgboost import XGBClassifier
import joblib
from sklearn.preprocessing import LabelEncoder

# --- Parse arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--max_depth", type=int, default=8)
parser.add_argument("--eta", type=float, default=0.3)
parser.add_argument("--objective", type=str, default="binary:logistic")
parser.add_argument("--num_round", type=int, default=200)
parser.add_argument("--train_file", type=str, required=True)
parser.add_argument("--target_column", type=str)
parser.add_argument("--feature_columns", type=str)
parser.add_argument("--categorical_columns", type=str)
args = parser.parse_args()

# --- SageMaker paths ---
input_dir = "/opt/ml/input/data/train"
output_dir = "/opt/ml/model"

# --- Load data ---
data_path = os.path.join(input_dir, args.train_file)
df = pd.read_csv(data_path)

# --- Feature list ---
feature_cols = args.feature_columns.split(",")
categorical_cols = (
    args.categorical_columns.split(",") if args.categorical_columns else []
)

# --- Drop NA for feature columns ---
df = df.dropna(subset=feature_cols)

# --- Encode categorical variables ---
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# --- Prepare X and y ---
X = df[feature_cols]
y = df[args.target_column]

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

# --- Save model ---
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "model.joblib")
joblib.dump(model, model_path)

# --- Save encoders ---
enc_path = os.path.join(output_dir, "encoders.joblib")
joblib.dump(encoders, enc_path)

print(f"✅ Model saved to {model_path}")
print(f"✅ Encoders saved to {enc_path}")
