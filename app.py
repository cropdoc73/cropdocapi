import tensorflow as tf
import numpy as np
import json
import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define the upload directory
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load disease data from JSON file
with open('data.json', 'r', encoding='utf-8') as f:
    disease_data = json.load(f)

# Tensorflow Model Prediction
def model_prediction(file):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(file, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        result_index = model_prediction(file_path)
        
        # Get predicted class name from data.json
        predicted_class_name = [
            'Apple__Apple_scab', 'Apple__Black_rot', 'Apple__Cedar_apple_rust', 'Apple__healthy',
            'Blueberry__healthy', 'Cherry_(including_sour)__Powdery_mildew',
            'Cherry_(including_sour)__healthy', 'Corn_(maize)__Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)__Common_rust_', 'Corn_(maize)__Northern_Leaf_Blight', 'Corn_(maize)__healthy',
            'Grape__Black_rot', 'Grape__Esca_(Black_Measles)', 'Grape__Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape__healthy', 'Orange__Haunglongbing_(Citrus_greening)', 'Peach__Bacterial_spot',
            'Peach__healthy', 'Pepper,_bell__Bacterial_spot', 'Pepper,_bell__healthy',
            'Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy',
            'Raspberry__healthy', 'Soybean__healthy', 'Squash__Powdery_mildew',
            'Strawberry__Leaf_scorch', 'Strawberry__healthy', 'Tomato__Bacterial_spot',
            'Tomato__Early_blight', 'Tomato__Late_blight', 'Tomato__Leaf_Mold',
            'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite',
            'Tomato__Target_Spot', 'Tomato__Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
            'Tomato__healthy'
        ]

        # Get details from data.json based on predicted class name
        prediction_name = predicted_class_name[result_index]
        if prediction_name in disease_data:
            disease_details = disease_data[prediction_name]
            # Get description from data.json
            description = disease_details.get("description", "")
            
            # Format description into numbered points
            formatted_description = ""
            for line in description.split('\n'):
                if line.strip():  # Ensure line is not empty
                    formatted_description += f"{line.strip()}\n"

            response = {
                "name": disease_details["name"],
                "description": formatted_description.strip(),  # Remove trailing newline
                "recommended_pesticide": disease_details.get("recommended_pesticide", ""),
                "pesticide_url": disease_details.get("picture_url", "")
            }
            return jsonify(response)
        else:
            return jsonify({
                "error": f"No details found for prediction: {prediction_name}",
                "available_diseases": list(disease_data.keys())
            }), 404

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(ebug=True, host='0.0.0.0', port=8080)
