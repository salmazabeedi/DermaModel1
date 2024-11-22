from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
CORS(app)

# Load the model with error handling
MODEL_PATH = '/app/models/SkinModelWork.h5'
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        print(f"Model file {MODEL_PATH} not found.")
except Exception as e:
    print(f"Error loading model: {str(e)}")

# Define label mapping
label_mapping = {
    0: 'Melanocytic Nevi',
    1: 'Melanoma',
    2: 'Benign Keratosis',
    3: 'Basal Cell Carcinoma',
    4: 'Actinic Keratosis',
    5: 'Vascular Lesion',
    6: 'Dermatofibroma'
}

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0
    
    if image_array.shape[-1] != 3:
        raise ValueError("Image must have 3 channels (RGB).")
    
    return image_array.reshape((1, 28, 28, 3))

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    base64_image = request.json['image']
    
    try:
        image_data = Image.open(io.BytesIO(base64.b64decode(base64_image)))
        image_data = preprocess_image(image_data)
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 400

    predictions = model.predict(image_data)
    predicted_class = np.argmax(predictions, axis=-1)[0]
    predicted_label = label_mapping.get(predicted_class, "Unknown")
    prediction_confidence = np.max(predictions)

    return jsonify({
        'predicted_label': predicted_label,
        'confidence': float(prediction_confidence)
    })

@app.route('/')
def index():
    return "DermaAware Skin Model"

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5590)))
