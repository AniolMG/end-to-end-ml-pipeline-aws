# End-to-End ML Pipeline on AWS

**Description:** End-to-end cloud ML pipeline using AWS.

---

## Project Overview (current)

1. Data storage and management using **Amazon S3**.
2. ...


---

## Step 1: S3 + IAM (Completed)

**Purpose:**  
Set up secure cloud storage and access management for the ML pipeline.

**What Was Done:**  
- Configured **Amazon S3** as the central storage for raw data, processed features, and model artifacts.  
- Implemented secure access using **AWS Identity and Access Management (IAM)**:  
  - Created **custom policies** to allow fine-grained S3 access.  
  - Set up **groups** and assigned policies.  
  - Created **users** for project access.  
- Enabled **IAM Identity Center (SSO)** to use a production-like authentication workflow.  
  - Created **permission sets** mapping IAM policies for SSO users.  
  - Verified access to S3 through AWS Management Console and AWS CLI.  
- Tested programmatic access using **AWS CLI**, confirming that files can be securely uploaded and downloaded.

