from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import json
import os

app = Flask(__name__)

# Load model and labels
model = load_model("model/model_2.h5")
with open("labels.json", "r") as f:
    class_names = json.load(f)

# Resize dimensions (depends on your model input)
IMAGE_SIZE = (224, 224)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        image = request.files["file"]
        img = Image.open(image).convert("RGB").resize(IMAGE_SIZE)
        img_array = np.expand_dims(np.array(img)/255.0, axis=0)
        pred = model.predict(img_array)[0]
        prediction = class_names[np.argmax(pred)]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
