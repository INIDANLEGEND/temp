# import streamlit as st
# from PIL import Image
# import numpy as np
# import cv2
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import traceback
# from streamlit_webrtc import webrtc_streamer


# # Define emotion labels
# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# @st.cache_resource
# def load_emotion_model():
#     # Load the pre-trained emotion detection model
#     return load_model('ResNet50.h5')

# model = load_emotion_model()

# def preprocess_image(image):
#     resized_image = cv2.resize(image, (224, 224))
#     # Normalize the pixel values
#     normalized_image = resized_image / 255.0
#     # Expand dimensions to match the model input
#     input_image = np.expand_dims(normalized_image, axis=0)
#     return input_image

# def predict_emotion(image):
#     try:
#         preprocessed_image = preprocess_image(image)
#         predictions = model.predict(preprocessed_image)
#         max_index = np.argmax(predictions[0])
#         emotion = emotion_labels[max_index]
#         return emotion
#     except Exception as e:
#         st.error("An error occurred during prediction.")
#         with open('error_log.txt', 'a') as f:
#             f.write(traceback.format_exc())
#         return None

# # Load OpenCV face detector
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Streamlit app
# st.title("Emotion Detection from Image")
# st.write("Upload an image, and the application will predict the emotion.")

# # Upload image
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

#     # Display the uploaded image
#     st.image(image, caption='Uploaded Image', use_column_width=True)
#     # Predict emotion
#     emotion = predict_emotion(image)
#     # Display the prediction
#     if emotion:
#         st.write(f"Predicted Emotion: {emotion}")

# # Live image capture
# st.write("Or capture a live video using your webcam:")

# start_capture = st.button("Start Video Capture")
# stop_capture = st.button("Stop Video Capture")

# if start_capture:
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         st.error("Could not open webcam.")
#     else:
#         stframe = st.empty()
#         while True:
#             ret, frame = cap.read()
#             if ret:
#                 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                 faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#                 for (x, y, w, h) in faces:
#                     face = frame[y:y+h, x:x+w]
#                     emotion = predict_emotion(face)
#                     cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#                     cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
#                 stframe.image(frame, channels="BGR")

#             if stop_capture:
#                 break
#         cap.release()


import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import traceback
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, ClientSettings

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
    try:
        preprocessed_image = preprocess_image(image)
        predictions = model.predict(preprocessed_image)
        max_index = np.argmax(predictions[0])
        emotion = emotion_labels[max_index]
        return emotion
    except Exception as e:
        st.error("An error occurred during prediction.")
        with open('error_log.txt', 'a') as f:
            f.write(traceback.format_exc())
        return None

class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            emotion = predict_emotion(face)
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        return frame.from_ndarray(image, format="bgr24")

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

# Live image capture
st.write("Or capture a live video using your webcam:")

webrtc_ctx = webrtc_streamer(
    key="emotion-detection",
    mode=WebRtcMode.SENDRECV,
    client_settings=ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    ),
    video_processor_factory=EmotionProcessor,
    async_processing=True,
)

if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.flip = st.checkbox("Flip", value=True)
