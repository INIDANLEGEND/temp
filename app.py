import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model

# Load the pre-trained emotion detection model
model = load_model('ResNet50.h5')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_image(image):
    resized_image = cv2.resize(image, (224, 224))
    # Normalize the pixel values
    normalized_image = resized_image / 255.0
    # Expand dimensions to match the model input
    input_image = np.expand_dims(normalized_image, axis=0)
    input_image = np.expand_dims(input_image, axis=-1)
    return input_image

def predict_emotion(image):
    preprocessed_image = preprocess_image(image)
    try:
        predictions = model.predict(preprocessed_image)
    except BrokenPipeError as e:
        st.error("A BrokenPipeError occurred. Please try again.")
    # Log the error details for further analysis
        with open('error_log.txt', 'a') as f:
            f.write(str(e))

    max_index = np.argmax(predictions[0])
    emotion = emotion_labels[max_index]
    return emotion

# Streamlit app
st.title("Emotion Detection from Image")
st.write("Upload an image, and the application will predict the emotion.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Predict emotion
    emotion = predict_emotion(image)

    # Display the prediction
    st.write(f"Predicted Emotion: {emotion}")
