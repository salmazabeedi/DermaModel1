from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
import os
import boto3
from botocore.exceptions import NoCredentialsError

# AWS S3 Configuration
S3_BUCKET = 'skinmodel'  # Your S3 bucket name
MODEL_FILE_NAME = 'SkinModelWork.h5'  # Name of the model file in S3
LOCAL_MODEL_PATH = '/tmp/SkinModelWork.h5'  # Path to save the model temporarily

# Function to download the model from S3
def download_model_from_s3():
    s3 = boto3.client('s3')
    try:
        # Download the model from S3 to local disk
        s3.download_file(S3_BUCKET, MODEL_FILE_NAME, LOCAL_MODEL_PATH)
        print(f"Downloaded model {MODEL_FILE_NAME} from S3.")
    except NoCredentialsError:
        print("Credentials not available for S3.")
        raise

# Download model at the start of the app
download_model_from_s3()

# Load the trained model
model = tf.keras.models.load_model(LOCAL_MODEL_PATH)

# Define label mapping
label_mapping = {0: 'Melanocytic Nevi', 1: 'Melanoma', 2: 'Benign Keratosis', 3: 'Basal Cell Carcinoma', 4: 'Actinic Keratosi', 5: 'Vascular Lesion', 6: 'Dermatofibroma'}

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

def preprocess_image(image):
    # Convert image to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize the image using Pillow (Image) and convert to numpy array
    image = image.resize((28, 28))  # Resize image to (28, 28) or the expected input size for your model
    
    # Convert the resized image to a NumPy array
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    
    # Ensure the image has 3 channels (RGB)
    if image_array.shape[-1] != 3:
        raise ValueError("Image must have 3 channels (RGB).")
    
    return image_array.reshape((1, 28, 28, 3))  # Reshape to match model input shape

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    base64_image = request.json['image']
    
    # Decode the base64 string to image
    try:
        # Open the image from base64 string
        image_data = Image.open(io.BytesIO(base64.b64decode(base64_image)))
        image_data = preprocess_image(image_data)
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 400

    # Make prediction
    predictions = model.predict(image_data)
    predicted_class = np.argmax(predictions, axis=-1)[0]
    predicted_label = label_mapping[predicted_class]
    prediction_confidence = np.max(predictions)  # Get confidence of the prediction

    return jsonify({
        'predicted_label': predicted_label,
        'confidence': float(prediction_confidence)  # Return confidence
    })

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5590)))
