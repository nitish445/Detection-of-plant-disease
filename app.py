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
        model = load_model("model.h5")  # Directly load the saved model
        print("Model input shape:", model.input_shape)  # Print model's expected input shape
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Define utility functions here to avoid import issues
def clean_image(image, target_size=(224, 224)):
    """Preprocess the image to match the model input shape.
    The target_size parameter should match your model's expected input dimensions.
    """
    image = np.array(image)
    # Resize the image to the target size
    image = np.array(Image.fromarray(image).resize(target_size, Image.LANCZOS))
    # Normalize pixel values (0-1)
    image = image / 255.0
    # Adding batch dimensions
    image = image[np.newaxis, :, :, :3]  # Ensure 3 channels (RGB)
    return image

def get_prediction(model, image):
    """Generate predictions from the model for the given image."""
    # No need to check shape here as we'll handle exceptions in the main code
    predictions = model.predict(image)
    predictions_arr = np.argmax(predictions, axis=1)
    return predictions, predictions_arr

def make_results(predictions, predictions_arr):
    """Convert model predictions into human-readable results."""
    labels = ["Healthy", "Multiple Diseases", "Rust", "Scab"]
    result = {
        "status": labels[int(predictions_arr[0])],
        "prediction": f"{int(predictions[0][predictions_arr[0]] * 100)}%"
    }
    return result

# Removing Streamlit menu and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Load the model
model = load_model_cached()

# Display model information if loaded successfully
if model is not None:
    st.sidebar.write("Model loaded successfully")
    st.sidebar.write(f"Expected input shape: {model.input_shape}")
    # Determine the correct input size from the model
    if model.input_shape and len(model.input_shape) == 4:
        input_height, input_width = model.input_shape[1], model.input_shape[2]
        st.sidebar.write(f"Using input dimensions: {input_height}x{input_width}")
    else:
        # Default to 224x224 if we can't determine from model
        input_height, input_width = 224, 224
        st.sidebar.write(f"Using default dimensions: {input_height}x{input_width}")
else:
    input_height, input_width = 224, 224  # Default fallback

# Title and description
st.title('🌿 Plant Disease Detection')
st.write("Upload an image of a plant's leaf to check if it is healthy or has a disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

# Process the uploaded file
if uploaded_file is not None:
    if model is None:
        st.error("Model failed to load. Please check the model path and file integrity.")
    else:
        try:
            st.text("Processing the image...")
            
            # Read and display image
            image = Image.open(io.BytesIO(uploaded_file.read()))
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess the image with the correct dimensions from the model
            processed_image = clean_image(image, target_size=(input_width, input_height))
            
            # Make predictions
            with st.spinner('Running inference...'):
                predictions, predictions_arr = get_prediction(model, processed_image)
            
            # Generate results
            result = make_results(predictions, predictions_arr)
            
            # Display result
            st.success(f"🌱 The plant is **{result['status']}** with **{result['prediction']}** confidence.")
        except Exception as e:
            st.error(f"Error processing image: {e}")
            st.write("Please try another image or check if the image format is supported.")