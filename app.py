from flask import Flask, render_template, request, jsonify
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
MODEL_PATH = BASE_DIR / "models" / "pneumonia_cnn_model_fixed.keras"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

model = None
grad_model = None


def load_model():
    global model, grad_model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    grad_input = tf.keras.Input(shape=(224, 224, 3))
    x = grad_input
    conv_output = None

    for layer in model.layers:
        x = layer(x)

        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_output = x

    if conv_output is None:
        raise ValueError("No Conv2D layer found for Grad-CAM.")

    grad_model = tf.keras.Model(
        inputs=grad_input,
        outputs=[conv_output, x]
    )

    print("CNN model loaded successfully.")
    print("Grad-CAM model created successfully.")


load_model()


def predict_and_gradcam(img):
    import cv2

    img_rgb = img.convert("RGB")
    resized = img_rgb.resize((224, 224))

    arr = np.array(resized, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    score = float(model.predict(arr, verbose=0)[0][0])

    pneumonia_probability = score
    normal_probability = 1.0 - score

    if pneumonia_probability >= 0.5:
        result = "PNEUMONIA"
        confidence = pneumonia_probability
    else:
        result = "NORMAL"
        confidence = normal_probability

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(arr)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise RuntimeError("Could not calculate Grad-CAM gradients.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    original = np.array(img_rgb)

    heatmap_resized = cv2.resize(
        heatmap,
        (original.shape[1], original.shape[0])
    )

    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    return result, confidence, Image.fromarray(overlay)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({
            "error": "Please upload a chest X-ray image."
        }), 400

    file = request.files["image"]

    if not file.filename:
        return jsonify({
            "error": "No image selected."
        }), 400

    try:
        image = Image.open(file.stream).convert("RGB")

        result, confidence, overlay = predict_and_gradcam(image)

        filename = "gradcam_result.png"
        output_path = UPLOAD_DIR / filename
        overlay.save(output_path)

        return jsonify({
            "prediction": result,
            "confidence": round(confidence * 100, 2),
            "gradcam": f"/static/uploads/{filename}"
        })

    except Exception as e:
        app.logger.exception("Prediction failed")

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        debug=False,
        host="127.0.0.1",
        port=5000
    )
