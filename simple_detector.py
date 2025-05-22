import cv2
import mediapipe as mp
import numpy as np
import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define hand signs and their corresponding finger states
# We'll detect a variety of common hand signs with some tolerance for variations
HAND_SIGNS = {
    "Open Hand": [1, 1, 1, 1, 1],     # All fingers extended
    "Fist": [0, 0, 0, 0, 0],          # All fingers closed
    "Thumbs Up": [1, 0, 0, 0, 0],     # Only thumb extended
    "Peace": [0, 1, 1, 0, 0],         # Index and middle finger extended
    "Point": [0, 1, 0, 0, 0],         # Only index finger extended
    "OK Sign": [1, 0, 0, 0, 0],       # Only thumb extended with index curled
    "Rock On": [1, 1, 0, 0, 1],       # Thumb, index, and pinky extended
    "Call Me": [1, 0, 0, 0, 1],       # Thumb and pinky extended
    "Three": [0, 1, 1, 1, 0],         # Index, middle, and ring fingers extended
    "Four": [0, 1, 1, 1, 1],          # All fingers except thumb extended
}

def is_finger_extended(landmarks, finger_tip_idx, finger_pip_idx, hand_type="Right"):
    """Check if a finger is extended based on its landmarks."""
    # Get the required landmarks
    tip = landmarks[finger_tip_idx]  # Fingertip
    pip = landmarks[finger_pip_idx]  # PIP joint (middle joint)
    mcp_idx = finger_pip_idx - 2    # MCP joint (knuckle)
    mcp = landmarks[mcp_idx]
    wrist = landmarks[0]
    
    # Special case for thumb
    if finger_tip_idx == 4:  # Thumb
        # Use the angle between thumb and index finger
        index_mcp = landmarks[5]  # Index finger MCP
        
        # For right hand: thumb is extended if it points left/up
        # For left hand: thumb is extended if it points right/up
        if hand_type == "Right":
            # Thumb is extended if it's to the left of the index finger base
            # or if it's higher than the wrist
            return tip.x < index_mcp.x or tip.y < wrist.y
        else:
            # For left hand, opposite x direction
            return tip.x > index_mcp.x or tip.y < wrist.y
    
    # For other fingers (index, middle, ring, pinky)
    # A finger is extended if:
    # 1. The fingertip is higher than the PIP joint (y-coordinate is smaller)
    # 2. The distance from fingertip to MCP is greater than from PIP to MCP
    
    # Calculate distances
    tip_to_mcp_dist = ((tip.x - mcp.x)**2 + (tip.y - mcp.y)**2)**0.5
    pip_to_mcp_dist = ((pip.x - mcp.x)**2 + (pip.y - mcp.y)**2)**0.5
    
    # Check if finger is extended
    is_extended = tip.y < pip.y and tip_to_mcp_dist > pip_to_mcp_dist
    
    # Special case for Peace sign detection (index and middle fingers)
    if finger_tip_idx in [8, 12]:  # Index or middle finger
        # Make it easier to detect these fingers as extended for Peace sign
        if tip.y < mcp.y - 0.05:  # If tip is significantly above MCP
            is_extended = True
    
    return is_extended

def detect_hand_sign(image_path):
    """Detect hand sign from an image."""
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        return None, "Could not read the image"
    
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image with MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
        
        results = hands.process(image_rgb)
        
        # If no hands are detected
        if not results.multi_hand_landmarks:
            return None, "No hands detected in the image"
        
        # Get the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Determine hand type (Left or Right)
        hand_type = "Right"
        if results.multi_handedness:
            hand_type = results.multi_handedness[0].classification[0].label
        
        # Draw hand landmarks on the image
        image_output = image.copy()
        mp_drawing.draw_landmarks(
            image_output,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
        # Check finger states
        landmarks = hand_landmarks.landmark
        finger_states = [
            is_finger_extended(landmarks, 4, 3, hand_type),  # Thumb
            is_finger_extended(landmarks, 8, 6),             # Index finger
            is_finger_extended(landmarks, 12, 10),           # Middle finger
            is_finger_extended(landmarks, 16, 14),           # Ring finger
            is_finger_extended(landmarks, 20, 18)            # Pinky
        ]
        
        # Match finger states to predefined hand signs with some tolerance
        detected_sign = "Unknown"
        best_match_score = 0
        
        # No finger state debugging information
        
        # Special case for specific finger combinations
        
        # Peace sign - index and middle fingers extended
        if finger_states[1] == 1 and finger_states[2] == 1 and finger_states[3] == 0 and finger_states[4] == 0:
            detected_sign = "Peace"
            
        # Point sign - only index finger extended (even if thumb is also extended)
        elif finger_states[1] == 1 and finger_states[2] == 0 and finger_states[3] == 0 and finger_states[4] == 0:
            detected_sign = "Point"
        else:
            for sign, states in HAND_SIGNS.items():
                # Skip signs we've already checked for in special cases
                if sign in ["Peace", "Point"]:
                    continue
                    
                # Calculate how many fingers match the expected state
                match_score = sum(1 for a, b in zip(finger_states, states) if a == b)
                
                # Perfect match
                if match_score == 5:
                    detected_sign = sign
                    break
                # Close match (4 out of 5 fingers correct)
                elif match_score == 4 and match_score > best_match_score:
                    best_match_score = match_score
                    detected_sign = sign
        
        # No detection text on image
        
        # Save the output image
        output_path = os.path.splitext(image_path)[0] + "_result.jpg"
        cv2.imwrite(output_path, image_output)
        
        return detected_sign, output_path

# Flask application
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
            sign, result_path = detect_hand_sign(file_path)
            
            if sign is None:
                return jsonify({'error': result_path})  # result_path contains error message
            
            # Get the result image filename
            result_filename = os.path.basename(result_path)
            
            # Move the result image to the results folder if it's not already there
            if os.path.dirname(result_path) != app.config['RESULT_FOLDER']:
                # Copy the file instead of renaming to avoid conflicts
                result_dest_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
                # Remove destination file if it exists
                if os.path.exists(result_dest_path):
                    os.remove(result_dest_path)
                # Copy the file
                import shutil
                shutil.copy2(result_path, result_dest_path)
                # Remove the original file
                os.remove(result_path)
            
            return jsonify({
                'success': True,
                'sign': sign,
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
