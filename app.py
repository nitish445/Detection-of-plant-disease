import io

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model


# Cache the model to improve performance
@st.cache_resource
def load_model_cached():
    try:
        model = load_model("model.h5")  # Load the trained model
        st.sidebar.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        return None

# Image preprocessing function
def clean_image(image, target_size=(224, 224)):
    """Preprocess the image to match model input size."""
    image = image.convert("RGB")  # Ensure 3-channel RGB
    image = image.resize(target_size, Image.LANCZOS)  # Resize to model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Prediction function
def get_prediction(model, image):
    """Generate predictions from the model."""
    predictions = model.predict(image)
    return predictions, np.argmax(predictions, axis=1)

# Convert predictions into readable results
def make_results(predictions, predictions_arr):
    labels = ["Healthy", "Multiple Diseases", "Rust", "Scab"]
    result = {
        "status": labels[int(predictions_arr[0])],
        "prediction": f"{predictions[0][predictions_arr[0]] * 100:.2f}%"
    }
    return result, labels, predictions

# Hide Streamlit menu/footer
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Load model
model = load_model_cached()
input_height, input_width = (224, 224)  # Default image size

# Streamlit UI
st.title("🌿 Plant Disease Detection")
st.write("Upload an image of a plant's leaf to detect diseases.")
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

# Process image
if uploaded_file is not None:
    if model is None:
        st.error("❌ Model not loaded. Please check the model path.")
    else:
        try:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Preprocess and debug image
            processed_image = clean_image(image, target_size=(input_width, input_height))
            st.image(processed_image[0], caption="Processed Image", use_column_width=True)
            
            # Make predictions
            with st.spinner("Running inference..."):
                predictions, predictions_arr = get_prediction(model, processed_image)
            
            # Debugging: Print all class probabilities
            result, labels, predictions = make_results(predictions, predictions_arr)
            
            st.write("### 🔍 Prediction Probabilities:")
            for i, label in enumerate(labels):
                st.write(f"{label}: {predictions[0][i] * 100:.2f}%")

            # Display result
            st.success(f"🌱 The plant is **{result['status']}** with **{result['prediction']}** confidence.")
        except Exception as e:
            st.error(f"⚠️ Error processing image: {e}")
