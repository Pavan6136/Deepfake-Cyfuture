import os
import streamlit as st
import numpy as np
import cv2
import requests
import tensorflow as tf
from tensorflow import keras
from transformers import pipeline


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


API_URL = "http://0.0.0.0:8080"
model_path = "/home/pavan/cyfuture/deepfake_model.keras"


if not os.path.exists(model_path):
    st.error(f"❌ Model file not found at: {model_path}. Please upload the correct model.")
    st.stop()


model = keras.models.load_model(model_path)


text_generator = pipeline("text-generation", model="gpt2")

# styling is done here
st.markdown(
    """
    <style>
        body {
            background-color: #f5f7fa;
        }
        .stTitle {
            font-size: 36px !important;
            text-align: center;
            color: #003366;
        }
        .stFileUploader {
            text-align: center;
            font-size: 18px !important;
        }
        .stImage {
            border-radius: 10px;
            border: 2px solid #007ACC;
        }
        .stMarkdown {
            font-size: 18px;
        }
        .stButton button {
            background-color: #007ACC !important;
            color: white !important;
            border-radius: 10px !important;
            font-size: 16px !important;
            padding: 10px !important;
        }
        .gpt-box {
            background-color: #cce5ff;
            padding: 12px;
            border-radius: 10px;
            font-weight: bold;
            color: black;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("🕵️‍♂️ Deepfake Image Detector")

# to upload Image
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image_bytes = uploaded_file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    st.image(image, caption="📷 Uploaded Image", use_container_width=True)


    image_resized = cv2.resize(image, (224, 224)) / 255.0
    image_resized = np.expand_dims(image_resized, axis=0)


    prediction = model.predict(image_resized, verbose=0)[0][0]
    result = "🟢 Real" if prediction > 0.5 else "🔴 Fake"


    st.subheader("🔍 Prediction Result")
    st.markdown(f"<h3 style='color: green;'>{result}</h3>", unsafe_allow_html=True)


    gpt_prompt = f"Deepfake detection is important because detecting fake media helps prevent misinformation. {result} images are classified based on AI analysis."
    gpt_text = text_generator(gpt_prompt, max_length=50)[0]["generated_text"]

    st.subheader("🤖 AI-Generated Explanation")
    st.markdown(f"<p class='gpt-box'>{gpt_text}</p>", unsafe_allow_html=True)


    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
    response = requests.post(f"{API_URL}/predict/", files=files)

    if response.status_code == 200:
        st.success("✅ Prediction stored in backend!")
    else:
        st.error("❌ Failed to send prediction to backend.")


    last_prediction_response = requests.get(f"{API_URL}/last_prediction/")
    if last_prediction_response.status_code == 200:
        last_prediction = last_prediction_response.json().get("last_prediction", "No predictions yet")
        st.info(f"📌 Last Stored Prediction: {last_prediction}")
    else:
        st.error("❌ Failed to retrieve last prediction from backend.")

