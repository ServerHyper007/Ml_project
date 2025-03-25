import os
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define upload folder and ensure it exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the pre-trained model
model = load_model('autism (1).h5') 

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 
    return img_array

@app.route('/', methods=['GET'])
def index():
    # Render the HTML page if needed
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the POST request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    file = request.files['image']
    
    # Check if a file was selected
    if file.filename == '':
        return jsonify({'error': 'No image file selected.'}), 400

    # Secure the filename and save the file
    filename = secure_filename(file.filename)
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(img_path)
    
    # Preprocess the image and predict
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    
    # Determine result based on prediction threshold
    if prediction > 0.015:
        result = 'Non Autistic'
    else:
        result = 'Autistic'
    
    # Return the response as JSON
    return jsonify({
        'filename': filename,
        'result': result,
        'prediction_value': float(prediction[0][0])
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use the port provided by the environment, defaulting to 5000
    app.run(host='0.0.0.0', port=port, debug=True)
