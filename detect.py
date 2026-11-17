import os
import cv2
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from zmq import CLIENT

# Load the API key

load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    raise ValueError("Roboflow API key not found. Check your .env file.")



# Connect to Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

# Configure detection settings
custom_configuration = InferenceConfiguration(
    confidence_threshold=0.25,
    iou_threshold=0.5
)


CLIENT.configure(custom_configuration)


# Load the test road image

image_path = "test_road.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(
"test_road.jpg not found. Check that it is in your project folder."
)

# Send the image to the pothole detection model

result = CLIENT.infer(
    image_path,
model_id="pothole-detection-yolo-8j0qo/5"
)

# Get predictions
predictions = result.get("predictions", [])
print("ALL CONFIDENCE VALUES:")

for prediction in predictions:
    print(prediction["confidence"])

print("TOTAL DETECTED:", len(predictions))
# Draw bounding boxes around detected potholes
for prediction in result.get("predictions", []):
    x = prediction["x"]
    y = prediction["y"]
    width = prediction["width"]
    height = prediction["height"]


    x1 = int(x - width / 2)
    y1 = int(y - height / 2)
    x2 = int(x + width / 2)
    y2 = int(y + height / 2)

    confidence = prediction.get("confidence", 0)
    label = prediction.get("class", "pothole")

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    text = f"{label}: {confidence:.2f}"
    cv2.putText(
        image,
        text,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
)


# Save the annotated image

output_path = "detected_road.jpg"
saved = cv2.imwrite(output_path, image)

if not saved:
    raise RuntimeError("Could not save the detected image.")

print("Detection completed!")
print("Number of detections:", len(result.get("predictions", [])))
print("Saved annotated image to:", os.path.abspath(output_path))


