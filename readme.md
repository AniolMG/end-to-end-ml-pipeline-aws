# Author: Aniol Molero Grau
## Project: End-to-End ML Pipeline (AWS Cloud Version)
## S3 + IAM + SageMaker Studio Pipelines + Model Registry

This project is the cloud-based continuation of [my previous hybrid ML architecture](https://github.com/AniolMG/mini-ml-pipeline). Here, the goal is to fully migrate the workflow to AWS.

---

## Project Summary
This repository implements a fully **cloud-native, end-to-end ML pipeline** on AWS for the Titanic dataset, from data storage to model deployment and serving.

The workflow includes:

- **Data and artifact storage in Amazon S3** with secure bucket policies
- **IAM roles and policies** for secure, least-privilege access for SageMaker, Lambda, and ECR
- **SageMaker Studio Pipelines** for training, evaluation, and model registration, ensuring reproducible and automated ML workflows
- **SageMaker Model Registry** for versioning, staging, and approving trained models
- **Lambda function packaged as a container** to serve predictions, automatically using the latest approved model
- **Public HTTP API via API Gateway** for serving predictions to external clients
- **Private HTTP API via API Gateway + IAM authentication** for serving only authorized users


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

6. **IAM User Setup for ECR, Lambda, S3, and SageMaker Access**  
   Create an IAM user with sufficient permissions to:  
   - Push Docker images to private ECR  
   - Access S3 bucket containing ML artifacts  
   - Manage Lambda functions  
   - Read from SageMaker Model Registry  
   - Monitor CloudWatch Logs  
   The user will need Access Key and Secret Key for local Docker build and deployment.

7. **Containerization & Deployment to ECR**  
   Package the Lambda function and dependencies in a Docker container:  
   - Build the image with a multi-stage Dockerfile using Amazon Linux 2023  
   - Test locally with Lambda Runtime Interface Emulator (RIE)  
   - Authenticate Docker to private ECR  
   - Tag and push the Docker image to private ECR  
   After this, the container is ready for Lambda deployment.

8. **Deploying the Lambda Function**  
   Create a Lambda function using the Docker image:  
   - Create an execution IAM role with permissions to read SageMaker models, access S3 artifacts, and write CloudWatch logs  
   - Create the Lambda function from the container image in private ECR  
   - Adjust memory (2048 MB) and timeout (30 seconds) for cold starts  
   - Test the function using a JSON payload containing sample Titanic passenger data  
   The Lambda function now serves predictions using the latest approved model.

9. **Setting Up the API Gateway**  
   Expose the Lambda function through a public HTTP API:  
   - Create an HTTP API in API Gateway  
   - Define a POST route, e.g., `/predict`  
   - Attach the Lambda function as the route integration  
   - Deploy the API to a stage, e.g., `prod`  
   - Use the stage's Invoke URL combined with `/predict` to send requests  
   - Test the endpoint from a local Python environment using the `requests` library  
   The API now allows external clients to send data and receive predictions from the Lambda function.

10. **Securing the API**
    Use IAM aothorization to make the API secure:
    - Make the API private by restricting access to authorized IAM users only.
    - Use API Gateway authorization mechanisms
    - Ensure only IAM users with the proper permissions can invoke the endpoint
    - Test again using the `requests`, `requests-aws4auth` and `boto3` libraries.
    The API now requires IAM authentication to be used by external clients.



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

## 6️ IAM User Setup for ECR, Lambda, S3, and SageMaker Access

To push Docker images to ECR and allow your Lambda function to fetch models from SageMaker and S3, you need an IAM user with sufficient permissions.

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
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::ml-pipeline-project-aniolmg",
                "arn:aws:s3:::ml-pipeline-project-aniolmg/*"
            ]
        },
        {
            "Sid": "ECRCreateRepository",
            "Effect": "Allow",
            "Action": [
                "ecr:CreateRepository"
            ],
            "Resource": "*"
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
            "Sid": "PrivateECRPushPull",
            "Effect": "Allow",
            "Action": [
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage",
                "ecr:PutImage",
                "ecr:InitiateLayerUpload",
                "ecr:UploadLayerPart",
                "ecr:CompleteLayerUpload",
                "ecr:DescribeRepositories"
            ],
            "Resource": "arn:aws:ecr:eu-west-3:<your-aws-account-id>:repository/amg-lambda-xgboost"
        },
        {
            "Sid": "LambdaLimitedAccess",
            "Effect": "Allow",
            "Action": [
                "lambda:CreateFunction",
                "lambda:UpdateFunctionCode",
                "lambda:UpdateFunctionConfiguration",
                "lambda:GetFunction",
                "lambda:InvokeFunction"
            ],
            "Resource": [
                "arn:aws:lambda:eu-west-3:<your-aws-account-id>:function:TitanicLambda",
                "arn:aws:lambda:eu-west-3:<your-aws-account-id>:function:TitanicLambda:*"
            ]
        },
        {
            "Sid": "SageMakerReadAccess",
            "Effect": "Allow",
            "Action": [
                "sagemaker:ListModelPackages",
                "sagemaker:DescribeModelPackage",
                "sagemaker:DescribeModelPackageGroup"
            ],
            "Resource": [
                "arn:aws:sagemaker:eu-west-3:<your-aws-account-id>:model-package-group/TitanicModel",
                "arn:aws:sagemaker:eu-west-3:<your-aws-account-id>:model-package/*"
            ]
        },
        {
            "Sid": "CloudWatchLogsForLambda",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": [
                "arn:aws:logs:eu-west-3:<your-aws-account-id>:log-group:/aws/lambda/TitanicLambda:*"
            ]
        }
    ]
}
```

Create a user with this policy using the AWS IAM service. The policy perhaps could be more fine-grained, but it's fine for this test project. Once it's created, access it (click on it) and go to ``Security credentials``.

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
docker run -it --rm -v ${PWD}:/workspace -v /var/run/docker.sock:/var/run/docker.sock amazonlinux:2023
```

2. Inside the container, install required tools:

```powershell
dnf update -y
dnf install -y awscli docker
```

3. Navigate to your project folder and build the image:

``You can choose any name you want for the image. Make sure that you use the --platform linux/amd64 argument so that the image is compatible with the Lambda runtime.``

```powershell
cd /workspace
docker build --platform linux/amd64 -t amg-lambda-xgboost .
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
curl -X POST http://localhost:9000/2015-03-31/functions/function/invocations -H "Content-Type: application/json" -d "{\"instances\":[{\"Age\":80,\"Sex\":\"male\",\"Pclass\":3}]}"
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

## 8️⃣ Deploying the Lambda Function

Once the Docker image is available in **Private ECR**, the next step is to create the Lambda function that will serve predictions.

---

### 8.1 Create an IAM Role for Lambda Execution

Before creating the function, we need an IAM role that gives Lambda permissions to:
- Write logs to CloudWatch
- Read the latest approved model from SageMaker Model Registry
- Access S3 to fetch model artifacts

Assuming the function will be called `TitanicLambda`, the IAM role should have the following policies:

**CloudWatch Logs Policy:**
```json
{
	"Version": "2012-10-17",
	"Statement": [
		{
			"Effect": "Allow",
			"Action": "logs:CreateLogGroup",
			"Resource": "arn:aws:logs:eu-west-3:<your-aws-account-id>:*"
		},
		{
			"Effect": "Allow",
			"Action": [
				"logs:CreateLogStream",
				"logs:PutLogEvents"
			],
			"Resource": [
				"arn:aws:logs:eu-west-3:<your-aws-account-id>:log-group:/aws/lambda/TitanicLambda:*"
			]
		}
	]
}
```

**SageMaker Read Access:**
```json
{
	"Version": "2012-10-17",
	"Statement": [
		{
			"Sid": "ReadModelRegistry",
			"Effect": "Allow",
			"Action": [
				"sagemaker:ListModelPackages",
				"sagemaker:DescribeModelPackage"
			],
			"Resource": "*"
		}
	]
}
```

**S3 Access for Model Artifacts:**
```json
{
	"Version": "2012-10-17",
	"Statement": [
		{
			"Sid": "ListMLBucket",
			"Effect": "Allow",
			"Action": [
				"s3:ListBucket"
			],
			"Resource": "arn:aws:s3:::ml-pipeline-project-aniolmg"
		},
		{
			"Sid": "GetModelArtifacts",
			"Effect": "Allow",
			"Action": [
				"s3:GetObject"
			],
			"Resource": "arn:aws:s3:::ml-pipeline-project-aniolmg/*"
		}
	]
}
```

**Why these policies?**
- CloudWatch logs allow monitoring the Lambda function.
- SageMaker permissions let the function discover and load the latest approved model.
- S3 permissions allow fetching the actual model files stored in the bucket.

---

### 8.2 Create Lambda Function from Container Image

1. Go to **AWS Lambda** > **Create function**.
2. Choose **Container image**.
3. Enter the function name: `TitanicLambda`.
4. Click **Browse images**, and choose the image you just pushed to **Private ECR**.
   - **Important:** Make sure to select the actual **Image**, not the Image index or any other element.
5. Open the **Change default execution role** dropdown, select **Use an existing role**, and choose the IAM role you created in step 8.1.
6. Click **Create function**.

---

### 8.3 Update Function Configuration for Cold Start

By default, Lambda has low memory and short timeout. For this model, we recommend:
- Memory: **2048 MB**
- Timeout: **30 seconds**

These values ensure the function can load the model and make predictions without timing out during cold starts.

---

### 8.4 Test the Lambda Function

**Sample JSON payload:**
```json
{
  "instances": [
    { "Age": 22, "Sex": "male", "Pclass": 3 },
    { "Age": 12, "Sex": "male", "Pclass": 1 },
    { "Age": 38, "Sex": "female", "Pclass": 1 }
  ]
}
```

**Expected response:**
```json
{
  "statusCode": 200,
  "body": "{\"predictions\": [0, 1, 1]}"
}
```

**Notes:**
- The first invocation (cold start) may take around **10 seconds**, because Lambda pulls the container and loads the model.
- Subsequent invocations (warm starts) are much faster, typically around **5 ms**.
- You can test the function directly in the Lambda console using the **Test** button.

---

After this step, the Lambda function is fully deployed and ready to serve predictions from the container image.

## 9️⃣ Setting Up the API Gateway for Titanic Predictions

Once the Lambda function is ready, the next step is to expose it through an API so that it can be called from external clients.

---

### 9.1 Create an HTTP API

1. Go to **API Gateway** in the AWS Console and click **Create API**.
2. In the **HTTP API** section, click **Build**.
3. Configure the API:
   - Name: `Titanic-HTTP-API`
   - Click **Next**.
   - No need to configure routes yet, click **Next**.
   - We'll define stages later, click **Next**.
   - Click **Create**.

> The API is created, but not yet functional. Next, we'll define a route and attach our Lambda function.

---

### 9.2 Define the Route

1. Navigate to **Develop > Routes**.
2. Click **Create**.
3. Configure the route:
   - **Method:** `POST` (we'll send JSON payloads)
   - **Route:** `/predict`
   - Click **Create**.

---

### 9.3 Attach Lambda Integration

1. Still in **Develop > Routes**, select the `POST` method under `/predict`.
2. Inside the "Route Details" box click **Attach integration** → **Create and attach an integration**.
3. Set up the integration:
   - **Integration type:** `Lambda function`
   - **Region:** Select the correct AWS region
   - **Lambda function:** Choose the function we created before (ARN ending with `:TitanicLambda`)
   - Click **Create**

> Now, the route `/predict` is connected to your Lambda function.

---

### 9.4 Deploy the API

1. Go to **Deploy > Stages**.
2. Click **Create** under the **Stages for Titanic-HTTP-API** box.
3. Enter a **Stage name**, e.g., `prod`.
4. Click **Create**.
5. Click **Deploy** on the top right:
   - Choose the `prod` stage in the dropdown
   - Click **Deploy**

6. After deployment, in **Stage details**, you'll see the **Invoke URL**:

```
https://<api-id>.execute-api.<region>.amazonaws.com/prod/
```

> The full endpoint for predictions is:
```
https://<api-id>.execute-api.<region>.amazonaws.com/prod/predict
```

---

### 9.5 Test the API Gateway

1. On your local machine, create a Python virtual environment in the root folder of your project:

```powershell
python -m venv apiTestEnv
```

2. Activate it:

```powershell
.\apiTestEnv\Scripts\activate.bat
```

3. Install the `requests` library:

```powershell
pip install requests
```

4. Run the `testAPI.py` script:

```powershell
python src/testAPI.py
```

5. When prompted, enter your API Gateway URL and input passenger data.

6. Expected example output:

```text
Response from Lambda API:
{'predictions': [1, 0, 1]}
```

> Now your public API to serve Titanic predictions is fully set up and functional.

## 🔟 Making the API Private (Authenticate via IAM)

In the **API Gateway** service in the AWS console:

1. Go to **Develop > Routes**.
2. Click on **POST** under the `/predict` endpoint.
3. Click **Attach authorization**.
4. In the dropdown, select **IAM (built-in)**.
5. Click **Attach authorizer**.

> Now, if you try to run `testAPI.py`, it shouldn't work since the API requires authentication (it might take some time for the update to propagate).

---

### 10.1 Update IAM Policy

Add the following statement to the IAM policy you created in step 6:

```json
{
    "Sid": "APIGatewayInvoke",
    "Effect": "Allow",
    "Action": [
        "execute-api:Invoke"
    ],
    "Resource": [
        "arn:aws:execute-api:eu-west-3:<your-aws-account-id>:<api-id>/*/POST/predict"
    ]
}
```

- Replace `<api-id>` with your API Gateway ID (found in the API details or in the API URL `https://<api-id>.execute-api.eu-west-3.amazonaws.com`).

---

### 10.2 Install Required Libraries

Use the same virtual environment as before and install the necessary packages:

```bash
pip install requests requests-aws4auth boto3
```

---

### 10.3 Run the IAM-Authenticated Test Script

The new script `test_API_IAM_authentication.py` works similarly to the previous `testAPI.py`, but now includes IAM authentication.

```bash
python src/test_API_IAM_authentication.py
```

1. Enter your **AWS Access Key ID** and **AWS Secret Access Key** for the IAM user (same credentials used in step 7.3).
2. Select the corresponding **AWS region**.
3. Input passengers' details as before.

> The API will now only respond to requests authenticated with valid IAM credentials.

