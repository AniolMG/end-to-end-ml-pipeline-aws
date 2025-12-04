import os
import joblib
import boto3
import tarfile
import tempfile

MODEL_PACKAGE_GROUP = "TitanicModel"  # SageMaker Model Package Group name

_cached_model = None
_cached_encoders = None


def load_model_and_encoders():
    global _cached_model, _cached_encoders

    if _cached_model is not None:
        return _cached_model, _cached_encoders

    sm_client = boto3.client("sagemaker")
    
    # Get latest approved model package
    response = sm_client.list_model_packages(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending"
    )

    packages = response.get("ModelPackageSummaryList", [])
    if not packages:
        raise RuntimeError(f"No approved model package found in {MODEL_PACKAGE_GROUP}")

    latest_package_arn = packages[0]["ModelPackageArn"]

    # Get model artifact S3 URL
    package_desc = sm_client.describe_model_package(ModelPackageName=latest_package_arn)
    model_url = package_desc["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]

    # Download and extract the model
    s3 = boto3.client("s3")
    bucket = model_url.split("/")[2]
    key = "/".join(model_url.split("/")[3:])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        local_tar = os.path.join(tmpdir, "model.tar.gz")
        s3.download_file(bucket, key, local_tar)

        with tarfile.open(local_tar, "r:gz") as tar:
            tar.extractall(path=tmpdir)

        # Expecting files inside the tar: model.joblib, encoders.joblib
        model_path = os.path.join(tmpdir, "model.joblib")
        encoders_path = os.path.join(tmpdir, "encoders.joblib")

        _cached_model = joblib.load(model_path)
        _cached_encoders = joblib.load(encoders_path)

    return _cached_model, _cached_encoders
