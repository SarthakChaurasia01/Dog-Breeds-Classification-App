from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model_2.h5")

# Load class labels
with open("labels.json", "r") as f:
    class_names = json.load(f)

def preprocess_image(img):
    img = img.resize((224, 224))  # or model input shape
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    img = Image.open(file.stream).convert("RGB")
    img = preprocess_image(img)

    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = float(np.max(prediction[0]))

    return jsonify({
        'class': predicted_class,
        'confidence': round(confidence * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
