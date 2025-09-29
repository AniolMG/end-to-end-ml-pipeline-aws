import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost import XGBoost

# Automatically get the notebook's IAM role
role = sagemaker.get_execution_role()

# S3 paths
train_s3 = "s3://ml-pipeline-project-aniolmg/data/titanic_data.csv"
output_s3 = "s3://ml-pipeline-project-aniolmg/models/"

# Define XGBoost estimator
xgb_estimator = XGBoost(
    entry_point="train_model.py",
    role=role,
    instance_count=1,              # only one instance
    instance_type="ml.m5.large",   # sufficient for Titanic dataset
    framework_version="1.7-1",
    py_version="py3",
    output_path=output_s3,
    hyperparameters={
        "max_depth": 8,
        "eta": 0.3,
        "objective": "binary:logistic",
        "num_round": 200
    }
)

# Launch training job
xgb_estimator.fit({"train": TrainingInput(train_s3, content_type="csv")})