import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# -------------------------------
# Page configuration (must be first Streamlit command)
# -------------------------------
st.set_page_config(
    page_title="Lung Cancer Detection",
    page_icon="🩺",
    layout="wide"
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "VGG_lung_model.h5")
    return tf.keras.models.load_model(model_path)

model = load_model()

# -------------------------------
# Class names
# -------------------------------
class_names = ["Normal", "Adenocarcinoma", "Squamous Cell Carcinoma"]

# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("🩺 Lung Cancer Detection using VGG16")
st.markdown(
    """
    Upload a lung histopathology image, and the model will classify it as one of the following:
    - **Normal**
    - **Adenocarcinoma**
    - **Squamous Cell Carcinoma**
    """
)

# File uploader
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image for VGG16
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Model prediction
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds[0])
    confidence = preds[0][predicted_class] * 100

    # Display prediction results
    st.subheader("📊 Prediction Results")
    st.success(f"**Predicted Class:** {class_names[predicted_class]}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    # Show probability distribution as bar chart
    st.bar_chart({class_names[i]: float(preds[0][i]) for i in range(len(class_names))})
