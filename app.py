import os
import uuid
import cv2

from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory,
    render_template
)
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    raise ValueError("Roboflow API key not found. Check your .env file.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=BASE_DIR,
    static_folder=os.path.join(BASE_DIR, "static")
)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

custom_configuration = InferenceConfiguration(
    confidence_threshold=0.25,
    iou_threshold=0.5
)

CLIENT.configure(custom_configuration)
    
    
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

custom_configuration = InferenceConfiguration(
    confidence_threshold=0.25,
    iou_threshold=0.5
)

CLIENT.configure(custom_configuration)

MODEL_ID = "pothole-detection-yolo-8j0qo/5"

MODEL_ID = "pothole-detection-yolo-8j0qo/5"


@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/<page>.html")
def frontend_page(page):
    allowed_pages = {
        "index",
        "login",
        "report",
        "results",
        "my-reports",
        "admin",
        "map",
        "manage-reports"
    }

    if page not in allowed_pages:
        return "Page not found", 404

    return send_from_directory(BASE_DIR, f"{page}.html")


@app.route("/css/<path:filename>")
def css_files(filename):
    return send_from_directory(os.path.join(BASE_DIR, "css"), filename)


@app.route("/js/<path:filename>")
def js_files(filename):
    return send_from_directory(os.path.join(BASE_DIR, "js"), filename)


@app.route("/detect", methods=["POST"])
def detect_potholes():
    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded. Use the field name 'image'."
        }), 400

    uploaded_file = request.files["image"]

    if not uploaded_file.filename:
        return jsonify({"error": "No image selected."}), 400

    file_id = uuid.uuid4().hex
    input_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.jpg")
    output_filename = f"{file_id}_detected.jpg"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    try:
        uploaded_file.save(input_path)

        image = cv2.imread(input_path)

        if image is None:
            return jsonify({
                "error": "The uploaded file is not a valid image."
            }), 400

        result = CLIENT.infer(input_path, model_id=MODEL_ID)
        predictions = result.get("predictions", [])

        for prediction in predictions:
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

            cv2.rectangle(
                image, (x1, y1), (x2, y2), (0, 255, 0), 2
            )

            cv2.putText(
                image,
                f"{label}: {confidence:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        if not cv2.imwrite(output_path, image):
            return jsonify({
                "error": "Could not save the processed image."
            }), 500

        return jsonify({
            "message": "Detection completed successfully",
            "pothole_count": len(predictions),
            "predictions": predictions,
            "annotated_image": f"/outputs/{output_filename}"
        })

    except Exception:
        app.logger.exception("Pothole detection failed")
        return jsonify({
            "error": "Pothole detection failed. Check the server terminal."
        }), 500

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


@app.route("/outputs/<path:filename>")
def get_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)