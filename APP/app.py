from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model and labels
model = load_model('keras_model.h5')

with open('labels.txt', 'r') as f:
    labels = [line.strip().split(' ')[1] for line in f.readlines()]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded', 400
        
        file = request.files['file']
        if file.filename == '':
            return 'No image selected', 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict
        predictions = model.predict(img_array)
        confidence = float(np.max(predictions[0])) * 100
        predicted_class = labels[np.argmax(predictions[0])]

        return render_template(
            'predict.html',
            prediction=predicted_class,
            confidence=round(confidence, 2),
            img_path=filepath
        )
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
