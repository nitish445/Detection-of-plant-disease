# Importing Necessary Libraries
import numpy as np
import tensorflow as tf
from PIL import Image


# Cleaning Image
def clean_image(image):
    """Preprocess the image to match the model input shape."""
    image = np.array(image)

    # Resize the image correctly
    image = np.array(Image.fromarray(image).resize((512, 512), Image.LANCZOS))

    # Normalize pixel values (0-1)
    image = image / 255.0

    # Adding batch dimensions
    image = image[np.newaxis, :, :, :3]  # Ensure 3 channels (RGB)

    return image


# Get Predictions from the Model
def get_prediction(model, image):
    """Generate predictions from the model for the given image."""
    
    # Ensure image has the correct shape (batch_size, height, width, channels)
    if image.shape != (1, 512, 512, 3):
        raise ValueError(f"Invalid input shape: {image.shape}, expected (1, 512, 512, 3)")
    
    predictions = model.predict(image)
    predictions_arr = np.argmax(predictions, axis=1)

    return predictions, predictions_arr


# Create Results from Predictions
def make_results(predictions, predictions_arr):
    """Convert model predictions into human-readable results."""
    labels = ["Healthy", "Multiple Diseases", "Rust", "Scab"]

    result = {
        "status": labels[int(predictions_arr[0])],
        "prediction": f"{int(predictions[0][predictions_arr[0]] * 100)}%"
    }

    return result
