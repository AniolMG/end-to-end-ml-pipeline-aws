import boto3
import sagemaker
from pipeline_definition import get_pipeline


def main():
    region = boto3.Session().region_name
    role = sagemaker.get_execution_role()

    pipeline = get_pipeline(region=region, role=role)

    pipeline.upsert(role_arn=role)

    execution = pipeline.start(
        parameters={
            "MaxDepth": 6,
            "TrainFile": "titanic_train.csv",
            "Target": "Survived",
            "FeatureColumns": "Age,Sex,Pclass",
            "CategoricalColumns": "Sex",
        }
    )

    print("⏳ Pipeline started. Waiting for completion...")
    execution.wait()
    print("✅ Pipeline executed successfully!")


if __name__ == "__main__":
    main()