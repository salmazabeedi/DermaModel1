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
from dotenv import load_dotenv  # Import dotenv

# Load environment variables from .env file (for local development)
load_dotenv()  # This loads the variables from your .env file into the environment

# AWS S3 Credentials (from environment variables or .env file)
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
s3_bucket_name = os.getenv('AWS_S3_BUCKET_NAME')
model_file_name = 'SkinModelWork.h5'

# Create an S3 client using boto3
s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name='us-east-1'  # Adjust region based on your S3 bucket's region
)

# Load the model from S3 when the app starts
def download_model_from_s3():
    try:
        # Download the model file from S3
        s3_client.download_file(s3_bucket_name, model_file_name, model_file_name)
        print(f"Model {model_file_name} downloaded successfully!")
        # Load the model into TensorFlow
        model = tf.keras.models.load_model(model_file_name)
        return model
    except NoCredentialsError:
        print("AWS credentials are not available. Please check your environment variables.")
        return None
    except Exception as e:
        print(f"Error downloading the model: {str(e)}")
        return None

# Load model into memory when app starts
model = download_model_from_s3()

# Define label mapping
label_mapping = {
    0: 'Melanocytic Nevi', 1: 'Melanoma', 2: 'Benign Keratosis', 
    3: 'Basal Cell Carcinoma', 4: 'Actinic Keratosi', 5: 'Vascular Lesion', 
    6: 'Dermatofibroma'
}

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Preprocess the image to match model input
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

# Prediction endpoint
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

    # Ensure the model is loaded
    if model is None:
        return jsonify({'error': 'Model is not loaded'}), 500

    # Make prediction
    try:
        predictions = model.predict(image_data)
        predicted_class = np.argmax(predictions, axis=-1)[0]
        predicted_label = label_mapping[predicted_class]
        prediction_confidence = np.max(predictions)  # Get confidence of the prediction

        return jsonify({
            'predicted_label': predicted_label,
            'confidence': float(prediction_confidence)  # Return confidence
        })
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500
    
# Basic index route for testing
@app.route('/')
def index():
    return "DermaAware Skin Model"

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5590)))
