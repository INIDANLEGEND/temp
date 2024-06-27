import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import traceback
import sys
import os

# Redirect sys.stdout to a file to avoid BrokenPipeError
sys.stdout = open('/tmp/streamlit_stdout.log', 'w')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@st.cache_resource
def load_emotion_model():
    # Load the pre-trained emotion detection model
    return load_model('ResNet50.h5')

model = load_emotion_model()

def preprocess_image(image):
    resized_image = cv2.resize(image, (224, 224))
    # Normalize the pixel values
    normalized_image = resized_image / 255.0
    # Expand dimensions to match the model input
    input_image = np.expand_dims(normalized_image, axis=0)
    return input_image

def predict_emotion(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    max_index = np.argmax(predictions[0])
    emotion = emotion_labels[max_index]
    return emotion

# Streamlit app
st.title("Emotion Detection from Image")
st.write("Upload an image, and the application will predict the emotion.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    # Predict emotion
    emotion = predict_emotion(image)
    # Display the prediction
    if emotion:
        st.write(f"Predicted Emotion: {emotion}")
