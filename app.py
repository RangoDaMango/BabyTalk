from flask import Flask, render_template, request, jsonify
import os

from feature_extractor import extract_features
from model_manager import predict
from memory import store_sample

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    audio = request.files["audio"]
    context = request.form.get("context")

    filename = audio.filename
    path = os.path.join(UPLOAD_FOLDER, filename)

    audio.save(path)

    features = extract_features(path)

    prediction = predict(features)

    return jsonify({
        "prediction": prediction,
        "file": filename,
        "context": context
    })


@app.route("/correct", methods=["POST"])
def correct():

    label = request.form.get("label")
    filename = request.form.get("file")

    filepath = os.path.join("uploads", filename)

    store_sample(filepath, label)

    return jsonify({"status": "updated"})


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)
