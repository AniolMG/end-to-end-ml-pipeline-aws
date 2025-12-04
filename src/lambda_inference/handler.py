import json
import base64
from inference import Predictor

predictor = Predictor()  # loads model + encoders once per container


def lambda_handler(event, context):
    """
    Expected input:
    {
        "instances": [
            {"Age": 22, "Sex": "male", "Pclass": 3},
            {"Age": 38, "Sex": "female", "Pclass": 1}
        ]
    }
    """
    body = event.get("body")
    if body:
        # API Gateway sends JSON as string
        body = json.loads(body)
    else:
        body = event

    instances = body.get("instances")
    if not instances:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Missing 'instances'"}),
        }

    preds = predictor.predict(instances)

    return {
        "statusCode": 200,
        "body": json.dumps({"predictions": preds}),
    }