import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Define your class names (trimmed version here for brevity)
dog_breeds = [
    "Chihuahua", "Japanese Spaniel", "Maltese Dog", "Pekinese", "Shih-Tzu",
    "Blenheim Spaniel", "Papillon", "Toy Terrier", "Rhodesian Ridgeback", "Afghan Hound",
    "Basset", "Beagle", "Bloodhound", "Bluetick", "Black-and-Tan Coonhound",
    "Walker Hound", "English Foxhound", "Redbone", "Borzoi", "Irish Wolfhound",
    # Add remaining breeds...
]

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_2.h5")
    return model

model = load_model()

st.title("🐶 Dog Breed Classification")
st.write("Upload an image of a dog and the model will try to predict its breed.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img_size = (160, 160)  # Your model's input size
    image = image.resize(img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, 0)

    # Prediction
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    st.markdown(f"### 🐕 Predicted Breed: **{dog_breeds[predicted_index]}**")
    st.markdown(f"### 🔍 Confidence: `{confidence:.2%}`")
