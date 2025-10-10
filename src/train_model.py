import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import joblib

# --- Parse SageMaker hyperparameters ---
parser = argparse.ArgumentParser()

parser.add_argument("--max_depth", type=int, default=8)
parser.add_argument("--eta", type=float, default=0.3)
parser.add_argument("--objective", type=str, default="binary:logistic")
parser.add_argument("--num_round", type=int, default=200)

args = parser.parse_args()

# --- SageMaker input/output directories ---
input_dir = "/opt/ml/input/data/train"
output_dir = "/opt/ml/model"

# --- Load dataset ---
data_path = os.path.join(input_dir, "titanic_data.csv")
df = pd.read_csv(data_path)

# --- Preprocess ---
df = df.dropna(subset=['Age'])
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
X = df[['Age', 'Sex', 'Pclass']]
y = df['Survived']

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5, stratify=y
)

# --- Model training ---
model = XGBClassifier(
    n_estimators=args.num_round,
    max_depth=args.max_depth,
    learning_rate=args.eta,
    objective=args.objective,
    random_state=5,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# --- Evaluation ---
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred):.4f}")

# --- Save model ---
os.makedirs(output_dir, exist_ok=True)
joblib.dump(model, os.path.join(output_dir, "titanic_model.joblib"))
