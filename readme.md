# Author: Aniol Molero Grau
## Project: End-to-End ML Pipeline (AWS Cloud Version)
## S3 + IAM + SageMaker Studio Pipelines + Model Registry

This project is the cloud-based continuation of [my previous hybrid ML architecture](https://github.com/AniolMG/mini-ml-pipeline). Here, the goal is to fully migrate the workflow to AWS.

---

## Project Summary
This repository contains (will contain) an end-to-end ML workflow built entirely on AWS, using:

- **Amazon S3** for data & artifact storage  
- **IAM** for secure, least-privilege access control  
- **SageMaker Studio** for preprocessing, training, and pipeline orchestration  
- **SageMaker Model Registry** for versioning, promoting, and organizing trained models
- ...

---

## Dataset
Kaggle "Titanic" dataset (https://www.kaggle.com/c/titanic/data).  
EDA was completed in the previous project, therefore this repository does not include EDA notebooks.

---

## High-Level AWS Pipeline

1. **S3 + IAM Setup**  
   Secure bucket storage and IAM policies for SageMaker Studio, training jobs, processing jobs, and S3 access.

2. **IAM Configuration for SageMaker**  
   Execution role with permissions to read/write S3, access SageMaker resources, and run pipelines.  
   Ensures all pipeline steps can execute in AWS securely.

3. **SageMaker Studio Pipeline**  
   Full training workflow implemented using a SageMaker Pipeline:  
   - Data loading from S3  
   - Training a simple XGBoost classifier  
   - Evaluation and metric computation  
   - Pipeline scripts:  
     - **[pipeline_definition.py](src/pipelines/pipeline_definition.py)**  
     - **[train_model.py](src/training/train_model.py)**  
     - **[compute_metrics.py](src/processing/compute_metrics.py)**  
     - **[run_pipeline.py](src/pipelines/run_pipeline.py)**  

4. **SageMaker Model Registry**  
   Tracks all trained model versions, metrics, and artifacts:  
   - Versioning (v1, v2, …)  
   - Stages (Pending / Staging / Production / Archived)  
   - Input/output schemas  
   - Metrics & parameters

5. **Prepare Lambda Deployment Files**  
   Organize Python code and dependencies for Lambda deployment:  
   - **[handler.py](src/lambda_inference/handler.py)** → Lambda entrypoint  
   - **[model_loader.py](src/lambda_inference/model_loader.py)** → Downloads and loads model  
   - **[inference.py](src/lambda_inference/inference.py)** → Preprocessing & prediction logic  
   - **[requirements.txt](src/lambda_inference/requirements.txt)** → Python dependencies  
   - **[Dockerfile](src/lambda_inference/Dockerfile)** → Multi-stage build to create minimal container

6. **IAM User for ECR Public / Lambda / S3 / SageMaker Access**  
   User with permissions to:  
   - Push Docker images to ECR Public  
   - Access S3 bucket and SageMaker Model Registry  
   - Deploy and manage Lambda functions  
   - Monitor logs in CloudWatch

7. **Containerization & Deployment to ECR Public**  
   Package the Lambda code and dependencies in a Docker container and push to ECR Public:  
   - Build and test locally using Amazon Linux 2023  
   - Authenticate with ECR Public  
   - Tag and push the Docker image  
   - Image is ready for Lambda deployment or other AWS services


---

``Note: In this guide, we use the eu-west-3 region as a reference. This region should work for all users, but in a real deployment you should choose the region that is most appropriate for your model’s target location.``

## 1️⃣ S3 + IAM Setup

The first component of the pipeline is secure cloud storage.

### S3 Bucket
Created: ``ml-pipeline-project-aniolmg``


**Bucket Policy** (allows SageMaker jobs to read/list objects):
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowSageMakerServiceAccess",
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::ml-pipeline-project-aniolmg",
                "arn:aws:s3:::ml-pipeline-project-aniolmg/*"
            ]
        }
    ]
}
```

**CORS Configuration** (Required for SageMaker Studio to interact with the bucket):
```json
[
    {
        "AllowedHeaders": ["*"],
        "AllowedMethods": ["POST", "PUT", "GET", "HEAD", "DELETE"],
        "AllowedOrigins": ["https://*.sagemaker.aws"],
        "ExposeHeaders": [
            "ETag",
            "x-amz-delete-marker",
            "x-amz-id-2",
            "x-amz-request-id",
            "x-amz-server-side-encryption",
            "x-amz-version-id"
        ]
    }
]
````

---

## 2️⃣ IAM Configuration (Sagemaker + S3)

SageMaker needs permission to read/write data and save models to the project bucket.

My execution role includes:

- AmazonSageMakerFullAccess
- Plus the following custom S3 policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowBucketListingAndInfo",
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket",
                "s3:GetBucketLocation"
            ],
            "Resource": "arn:aws:s3:::ml-pipeline-project-aniolmg"
        },
        {
            "Sid": "AllowObjectActions",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket",
                "s3:DeleteObject",
                "s3:AbortMultipartUpload",
                "s3:ListMultipartUploadParts",
                "s3:PutObjectAcl",
                "s3:GetObjectAcl"
            ],
            "Resource": [
                "arn:aws:s3:::ml-pipeline-project-aniolmg",
                "arn:aws:s3:::ml-pipeline-project-aniolmg/*"
            ]
        }
    ]
}
````
This ensures SageMaker jobs can read/write objects in this bucket and only this bucket.

---

## 3️⃣ SageMaker Studio Pipeline

The training workflow is implemented using a SageMaker Pipeline.

**Pipeline Steps:**
- Data loading from S3
- Training a simple XGBoost classifier
- Evaluation and metric generation
- Model registration to Model Registry

**Pipeline Parameters:**
- MaxDepth (integer)
- Eta (float)
- NumRound (integer)
- Objective (string)
- TrainFile (string)
- TestFile (string)
- Target (string)
- FeatureColumns (string)
- CategoricalColumns (string)

**Code Files:**
   - Pipeline definition → **[pipeline_definition.py](src/pipelines/pipeline_definition.py)**  
   - Training script → **[train_model.py](src/training/train_model.py)**  
   - Evaluation/metrics script → **[compute_metrics.py](src/processing/compute_metrics.py)**
   - Running the pipeline → **[run_pipeline.py](src/pipelines/run_pipeline.py)**

Trained models and evaluation results are saved automatically into S3, while metadata (such as metrics and parameters) is stored in SageMaker Model Registry.

## 4️⃣️ SageMaker Model Registry

Just like MLflow Model Registry in the previous project, SageMaker provides:

- Versioning (Model v1, v2, …)
- Stages (Staging / Production / Archived)
- Artifact storage paths
- Input/output schemas
- Metrics & parameters

After each pipeline run, the trained XGBoost model is automatically registered and appears in the Registry UI.  
**Note:** For now, the model registration is set to `PendingManualApproval`, it must be manually approved to move to Staging or Production.

## 5️⃣ Preparing Files for Building the Docker Image

Before building the Docker image for Lambda deployment, you need to organize your Python code and dependencies.

### 5.1 Required Files

Your `src/lambda_inference` folder should contain the following files:

- **[handler.py](src/lambda_inference/handler.py)**: Entry point for the Lambda function. It receives API requests, parses input data, calls the predictor, and returns predictions.
- **[model_loader.py](src/lambda_inference/model_loader.py)**: Responsible for downloading the latest approved model from SageMaker Model Registry, extracting it, and loading it into memory.
- **[inference.py](src/lambda_inference/inference.py)**: Contains the `Predictor` class that applies preprocessing and encoders to incoming data and makes predictions using the loaded model.

These three files together implement the Lambda function logic. The Docker container needs them to serve predictions.

- **requirements.txt**: Lists Python dependencies (e.g., pandas, joblib, boto3, xgboost) that must be installed in the Lambda environment.
- **Dockerfile**: Defines how to build the container image for Lambda deployment.

### 5.2 Dockerfile Multi-Stage Build

The Dockerfile is written in **two stages**:

1. **Builder Stage** (`public.ecr.aws/lambda/python:3.11 as builder`):
   - Installs all Python dependencies into a temporary directory.
   - Removes unnecessary files (tests, \_\_pycache\_\_) to reduce size.

2. **Final Lambda Image** (`public.ecr.aws/lambda/python:3.11`):
   - Copies only the installed dependencies from the builder stage into `/opt/python`.
   - Copies the Lambda Python code (`handler.py`, `model_loader.py`, `inference.py`).
   - Sets the Lambda entrypoint.

**Advantages of multi-stage build**:
- Keeps the final image small by excluding build tools and temporary files.
- Ensures Lambda only has the minimal runtime environment plus your dependencies and code.
- Faster deployments and reduced cold-start times.

By organizing your files and using a multi-stage Docker build, you ensure a clean, reproducible, and efficient Lambda container.

## 6️⃣ IAM User Setup for ECR Public, Lambda, S3, and SageMaker Access

To push Docker images to ECR Public and allow your Lambda function to fetch models from SageMaker and S3, you need an IAM user with sufficient permissions.

### 6.1 Policy for IAM User

Below is a full policy JSON that grants access to:
- S3 bucket containing your ML artifacts
- Private ECR repositories in your account (to push/pull images)
- ECR authorization token (needed for Docker login to ECR)
- Lambda (to create and manage Lambda functions)
- SageMaker (to access the Model Registry and fetch model locations)
- CloudWatch Logs (for monitoring Lambda execution)

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "S3AccessToMLBucket",
            "Effect": "Allow",
            "Action": [
                "s3:*"
            ],
            "Resource": [
                "arn:aws:s3:::ml-pipeline-project-aniolmg",
                "arn:aws:s3:::ml-pipeline-project-aniolmg/*"
            ]
        },
        {
            "Sid": "PrivateECRFullAccess",
            "Effect": "Allow",
            "Action": [
                "ecr:*"
            ],
            "Resource": [
                "arn:aws:ecr:eu-west-3:344809604964:repository/*"
            ]
        },
        {
            "Sid": "ECRAuthToken",
            "Effect": "Allow",
            "Action": [
                "ecr:GetAuthorizationToken"
            ],
            "Resource": "*"
        },
        {
            "Sid": "LambdaAccess",
            "Effect": "Allow",
            "Action": [
                "lambda:*"
            ],
            "Resource": "*"
        },
        {
            "Sid": "SageMakerAccess",
            "Effect": "Allow",
            "Action": [
                "sagemaker:*"
            ],
            "Resource": [
                "arn:aws:sagemaker:eu-west-3:344809604964:model-package-group/TitanicModel",
                "arn:aws:sagemaker:eu-west-3:344809604964:*"
            ]
        },
        {
            "Sid": "CloudWatchLogsForLambda",
            "Effect": "Allow",
            "Action": [
                "logs:*"
            ],
            "Resource": "*"
        }
    ]
}
```

Create a user with this policy using the AWS IAM service. The policy could be more fine-grained, but it's fine for this test project. Once it's created, access it (click on it) and go to ``Security credentials``.

Then, look for the tab that says ``Access keys (0)`` and click ``Create access key``.

For the use case choose ``Command Line Interface (CLI)``. And check the box that says ``I understand the above recommendation and want to proceed to create an access key``. Click ``Next``.

Add a descriptive tag if you want. This is optional. Click ``Create access key``.

Now your Access key and Secret access key have ben generated. It's **very important** that you write down these somewhere (like a .txt in your PC), especially the Secret access key, since this is the only time you will be able to see it. You can also click ``Download .csv file`` to download a file with both keys written down. We will use the keys later.

Also, make sure you don't push these keys to any public repository, since other people could use the credentials to assume the identity of your IAM user. This is only a test project with no sensible data, so no damage could be done, but still it's recommended to always employ good practises.

## 7️⃣ Containerization & Deployment to ECR

Once the model is registered and approved, the next step is to **package it in a Docker container** for deployment. This ensures the same environment used during training is replicated in production.

The following steps are for pushing to a **private ECR repository**, which can be used by AWS Lambda or any other AWS service. Private ECR incurs storage charges, so only keep images you need.

``Note: AWS charges about 0.10$ per GB per month in ECR storage, so be mindful about how much you upload, and delete it when not needed anymore or after finishing this guide``

``Note:`` ECR public allows up to 50GB of storage per month for free, but as of right now, ``**AWS Lambda cannot use ECR public repositories as an image**``, so we have to use private ECR repositories

---

### 7.1 Build the Docker Image

1. Start an **Amazon Linux 2023** container (to bypass Windows credential issues) and mount your project directory and Docker socket:

``Note: run this in the local folder (host) that contains the Dockerfile``

```powershell
docker run -it --rm `
  -v ${PWD}:/workspace `
  -v /var/run/docker.sock:/var/run/docker.sock `
  amazonlinux:2023
```

2. Inside the container, install required tools:

```powershell
dnf update -y
dnf install -y awscli docker
```

3. Navigate to your project folder and build the image:

``You can choose any name you want for the image``

```powershell
cd /workspace
docker build -t amg-lambda-xgboost .
```

---

### 7.2 Test the Container Locally

Run the container to ensure it works correctly with the Lambda Runtime Interface Emulator (RIE):

``Note: You will need the access key and secret access key for an IAM user with permissions to fetch models from SageMaker/S3 (the one we created in the previous step)``

```powershell
docker run -p 9000:8080 --env AWS_ACCESS_KEY_ID=<your-access-key> --env AWS_SECRET_ACCESS_KEY=<your-secret-access-key> --env AWS_DEFAULT_REGION=<your-region> amg-lambda-xgboost
```

Test with a sample request:

```powershell
curl -X POST http://localhost:9000/2015-03-31/functions/function/invocations ^
  -H "Content-Type: application/json" ^
  -d "{\"instances\":[{\"Age\":80,\"Sex\":\"male\",\"Pclass\":3}]}"
```

Expected response:

```json
{"statusCode":200,"body":"{\"predictions\":[0]}"}
```

---

### 7.3 Configure your AWS credentials

```powershell
aws configure  # Enter your private IAM user credentials (Access Key / Secret) from the previous step
```

---

### 7.4 Create the Private ECR Repository
``Use the region you prefer``

```powershell
aws ecr create-repository --repository-name amg-lambda-xgboost --region eu-west-3
```

``Note: You might get an error like "Unable to redirect output to pager. Received the following error when opening pager:
[Errno 2] No such file or directory: 'less'". Pay no attention to it``

After this, you should see the repository in the AWS Console under **ECR > Private Registry > Repositories**. Note the repository URI, it will be used in the next steps.

---

### 7.5 Authenticate Docker to Private ECR

```powershell
aws ecr get-login-password --region eu-west-3 | docker login --username AWS --password-stdin <your-aws-account-id>.dkr.ecr.eu-west-3.amazonaws.com
```

---

### 7.6 Tag the Docker Image for Private ECR

```powershell
docker tag amg-lambda-xgboost:latest <your-aws-account-id>.dkr.ecr.eu-west-3.amazonaws.com/amg-lambda-xgboost:latest
```

---

### 7.7 Push the Docker Image to Private ECR

```powershell
docker push <your-aws-account-id>.dkr.ecr.eu-west-3.amazonaws.com/amg-lambda-xgboost:latest
```

``Note: ECR logins expire after 12 hours. If you followed this guide and after running the command you get an error "403 Forbidden", re-authenticate to ECR (step 7.5) and run this command again.``

After this step, the container image is fully pushed to **Private ECR**, ready to be used in **AWS Lambda** or other AWS services.
