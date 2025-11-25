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
   Secure bucket storage + IAM policies for Studio, training jobs, processing jobs, and S3 access.

2. **SageMaker Studio Pipelines**  
   Preprocessing, training, and evaluation fully executed in AWS.  
   Scripts:  
   - Pipeline definition → **[train_model_sagemaker_pipeline.ipynb](src/train_model_sagemaker_pipeline.ipynb)**  
   - Training script → **[train_model.py](src/train_model.py)**  
   - Evaluation/metrics script → **[compute_metrics.py](src/compute_metrics.py)**  

3. **Model Registry**  
   All trained model versions, artifacts, metrics, and parameters are stored and tracked in SageMaker Model Registry.

4. **(Next Step) Model Deployment** 

5. ...

---

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

## 2️⃣ IAM Configuration

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
- Pipeline definition: [train_model_sagemaker_pipeline.ipynb](src/train_model_sagemaker_pipeline.ipynb)
- Training script: [train_model.py](src/train_model.py)
- Metric computation: [compute_metrics.py](src/compute_metrics.py)

Trained models and evaluation results are saved automatically into S3, while metadata (such as metrics and parameters) is stored in SageMaker Model Registry.

## 4️ SageMaker Model Registry

Just like MLflow Model Registry in the previous project, SageMaker provides:

- Versioning (Model v1, v2, …)
- Stages (Staging / Production / Archived)
- Artifact storage paths
- Input/output schemas
- Metrics & parameters

After each pipeline run, the trained XGBoost model is automatically registered and appears in the Registry UI.  
**Note:** For now, the model registration is set to `PendingManualApproval`, it must be manually approved to move to Staging or Production.
