import requests
import json

# Ask the user for the API Gateway URL
api_url = input("Enter the API Gateway URL (e.g., https://xxxx.execute-api.eu-west-3.amazonaws.com/prod/predict): ")

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

# Make POST request
try:
    response = requests.post(api_url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
    response.raise_for_status()  # Raise an error if status code is not 2xx
    print("Response from Lambda API:")
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"Error making request: {e}")
