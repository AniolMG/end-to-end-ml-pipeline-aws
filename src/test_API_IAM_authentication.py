import requests
import json
from requests_aws4auth import AWS4Auth

# Ask the user for the API Gateway URL
api_url = input("Enter the API Gateway URL (e.g., https://xxxx.execute-api.eu-west-3.amazonaws.com/prod/predict): ")

# Ask for AWS credentials
aws_access_key_id = input("Enter AWS Access Key ID: ").strip()
aws_secret_access_key = input("Enter AWS Secret Access Key: ").strip()
region = input("Enter AWS region (e.g., eu-west-3): ").strip()

# Create AWS4Auth object (without session token)
auth = AWS4Auth(
    aws_access_key_id,
    aws_secret_access_key,
    region,
    'execute-api'
)

# Function to ask for passenger details
def get_passenger_info():
    age = int(input("Enter Age: "))
    sex = input("Enter Sex (male/female): ").strip().lower()
    pclass = int(input("Enter Passenger class (1, 2, or 3): "))
    return {"Age": age, "Sex": sex, "Pclass": pclass}

# Ask for multiple passengers
passengers = []
while True:
    passengers.append(get_passenger_info())
    more = input("Do you want to add another passenger? (y/n): ").strip().lower()
    if more != 'y':
        break

# Prepare request payload
payload = {"instances": passengers}

# Make POST request with AWS4Auth
try:
    response = requests.post(api_url, auth=auth, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
    response.raise_for_status()
    print("Response from Lambda API:")
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"Error making request: {e}")
