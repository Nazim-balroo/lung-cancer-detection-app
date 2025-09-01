import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# -------------------------------
# Page configuration
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
# Class names and explanations
# -------------------------------
class_names = ["Normal", "Adenocarcinoma", "Squamous Cell Carcinoma"]

explanations = {
    "Normal": "🟢 No signs of cancer were detected in the image. The lung tissue appears healthy.",
    "Adenocarcinoma": "🔴 A type of non-small cell lung cancer that originates in mucus-secreting glands.",
    "Squamous Cell Carcinoma": "🟡 A form of lung cancer arising in the squamous cells lining the airways, often linked to smoking."
}

# -------------------------------
# UI Header
# -------------------------------
st.title("🩺 Lung Cancer Detection using VGG16")

st.markdown(
    """
    Upload a lung histopathology image (H&E stained), and this AI-powered tool will classify it into:
    - 🟢 **Normal**
    - 🔴 **Adenocarcinoma**
    - 🟡 **Squamous Cell Carcinoma**
    
    > ⚠️ **Disclaimer**: This tool is for educational and research purposes only. It does **not** provide medical advice. Always consult a qualified healthcare provider for diagnosis.
    """
)

# -------------------------------
# Image Upload
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload Lung Tissue Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)

        # Preprocess for VGG16
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        preds = model.predict(img_array)
        predicted_class_index = np.argmax(preds[0])
        predicted_class = class_names[predicted_class_index]
        confidence = float(preds[0][predicted_class_index]) * 100

        # Display prediction
        st.subheader("📊 Prediction Results")
        st.success(f"**Predicted Class:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2f}%")

        # Explanation
        st.markdown(f"📘 **What this means:** {explanations[predicted_class]}")

        # Show probability distribution
        st.markdown("### 📈 Class Probabilities")
        st.bar_chart({class_names[i]: float(preds[0][i]) for i in range(len(class_names))})

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
