import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        r"C:\Users\NAZIM BALROO\OneDrive\Desktop\nazim1\VGG_lung_model.h5"
    )

model = load_model()

# Correct Class Names
class_names = ["Normal", "Adenocarcinoma", "Squamous Cell Carcinoma"]

# Streamlit App
st.set_page_config(page_title="Lung Cancer Detection", page_icon="🩺", layout="wide")

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
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))  # Resize to VGG16 input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds[0])
    confidence = preds[0][predicted_class] * 100

    st.subheader("📊 Prediction Results")
    st.success(f"**Predicted Class:** {class_names[predicted_class]}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    # Show probability distribution
    st.bar_chart({class_names[i]: preds[0][i] for i in range(len(class_names))})
