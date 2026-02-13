# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
import tempfile
import pickle
import cv2

from severity import estimate_severity
from recommend import recommend_solution
from gradcam import generate_gradcam, overlay_gradcam

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
MODEL_PATH = "models/cnn_model.h5"
CLASS_PATH = "models/class_names.pkl"

# ---------------------------------------------------
# Load Model + Class Names
# ---------------------------------------------------
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "rb") as f:
    class_names = pickle.load(f)

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.title("🌿 Crop Disease Detection System with Explainable AI")
st.write("Upload a leaf image to detect disease, severity, recommendation, and Grad-CAM localization.")

uploaded_file = st.file_uploader(
    "📌 Upload Leaf Image",
    type=["jpg", "png", "jpeg"]
)

# ---------------------------------------------------
# Prediction Pipeline
# ---------------------------------------------------
if uploaded_file:

    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Leaf Image", use_container_width=True)

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_path = tmp_file.name

    # Prepare image for CNN prediction
    img = image.load_img(temp_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict probabilities
    prediction = model.predict(img_array)[0]

    # Predicted class
    class_index = np.argmax(prediction)
    predicted_disease = class_names[class_index]

    # ---------------------------------------------------
    # Disease Output
    # ---------------------------------------------------
    st.subheader("🌿 Disease Prediction Result")
    st.write("✅ Detected Disease:", predicted_disease)

    # ---------------------------------------------------
    # Probability Graph with Disease Names (Top 5)
    # ---------------------------------------------------
    st.subheader("📊 Top-5 Prediction Probabilities")

    top_indices = prediction.argsort()[-5:][::-1]
    top_probs = prediction[top_indices]
    top_labels = [class_names[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(top_labels[::-1], top_probs[::-1])
    ax.set_xlabel("Probability")
    ax.set_title("Top-5 Predicted Disease Classes")

    st.pyplot(fig)

    # ---------------------------------------------------
    # Improved Severity Estimation
    # ---------------------------------------------------
    st.subheader("🔥 Disease Severity Estimation")

    severity, mask = estimate_severity(temp_path)

    st.write(f"Severity Level: **{severity:.2f}%**")

    st.subheader("🩻 Diseased Area Mask Output")
    st.image(mask, caption="Detected Infected Region", use_container_width=True)

    # ---------------------------------------------------
    # Recommendation
    # ---------------------------------------------------
    st.subheader("💡 Farmer Recommendation")

    solution = recommend_solution(predicted_disease)
    st.write("Recommended Action:", solution)

    # ---------------------------------------------------
    # Grad-CAM Explainability Module
    # ---------------------------------------------------
    st.subheader("📌 Grad-CAM Disease Localization")

    last_conv_layer = "conv3"

    heatmap = generate_gradcam(model, img_array, last_conv_layer)
    overlay = overlay_gradcam(temp_path, heatmap)

    st.image(
        cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
        caption="Grad-CAM Heatmap Overlay (Model Focus Region)",
        use_container_width=True
    )