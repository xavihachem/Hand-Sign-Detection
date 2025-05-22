import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from detect_from_image import detect_hand_sign

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload and result directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the image
        try:
            result = detect_hand_sign(file_path)
            
            if result is None:
                return jsonify({
                    'error': 'No hand detected in the image or hand is too close to the edge'
                })
            
            # Get the result image path
            result_filename = os.path.splitext(filename)[0] + "_result.jpg"
            result_path = os.path.splitext(file_path)[0] + "_result.jpg"
            
            # Move the result image to the results folder
            if os.path.exists(result_path):
                os.rename(result_path, os.path.join(app.config['RESULT_FOLDER'], result_filename))
            
            return jsonify({
                'success': True,
                'sign': result,
                'original_image': filename,
                'result_image': result_filename
            })
            
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
