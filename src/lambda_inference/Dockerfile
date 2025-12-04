# ===============================
# Stage 1: Build dependencies
# ===============================
FROM public.ecr.aws/lambda/python:3.11 as builder

# Install pip (already included in Lambda base) and build essentials for compiling wheels if needed
RUN yum install -y gcc gcc-c++ make && \
    python3.11 -m ensurepip && \
    pip install --upgrade pip

WORKDIR /lambda_build

# Copy requirements.txt
COPY requirements.txt .

# Install dependencies into /lambda_build
RUN pip install --no-cache-dir -r requirements.txt -t /lambda_build

# Remove test directories from numpy/xgboost/joblib to save space
RUN find /lambda_build -type d -name "tests" -exec rm -rf {} + && \
    find /lambda_build -type d -name "__pycache__" -exec rm -rf {} +

# ===============================
# Stage 2: Final Lambda image
# ===============================
FROM public.ecr.aws/lambda/python:3.11

# Copy only the installed libraries from the builder
COPY --from=builder /lambda_build /opt/python

# Copy Lambda code
COPY handler.py model_loader.py inference.py ./

# Set Lambda entrypoint
CMD ["handler.lambda_handler"]