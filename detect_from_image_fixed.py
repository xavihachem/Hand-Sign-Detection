import cv2
import numpy as np
import math
import os
import argparse
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

def detect_hand_sign(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    # Initialize the hand detector and classifier
    detector = HandDetector(maxHands=1)
    
    # Use the model file from the Model directory
    model_path = "Model/model.h5"
    if not os.path.exists(model_path):
        print(f"Error: Could not find model file at {model_path}")
        return None
    
    # Check for labels file
    if os.path.exists("Model/labels.txt"):
        labels_path = "Model/labels.txt"
    else:
        # Create a default labels file if it doesn't exist
        labels_path = "Model/labels.txt"
        os.makedirs("Model", exist_ok=True)
        with open(labels_path, "w") as f:
            f.write("A\nB\nC")
    
    # Initialize classifier with the correct paths
    classifier = Classifier(model_path, labels_path)
    
    # Get the labels
    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    
    # Process the image
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if not hands:
        print("No hands detected in the image.")
        return None
    
    # Process detected hand
    hand = hands[0]
    x, y, w, h = hand['bbox']
    
    # Prepare the image for classification
    offset = 20
    imgSize = 300
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    
    # Crop the hand region with boundary checks
    y_min = max(0, y - offset)
    y_max = min(img.shape[0], y + h + offset)
    x_min = max(0, x - offset)
    x_max = min(img.shape[1], x + w + offset)
    
    imgCrop = img[y_min:y_max, x_min:x_max]
    
    # Handle cases where the crop might go out of bounds
    if imgCrop.size == 0:
        print("Hand is too close to the edge of the image.")
        return None
    
    # Resize and center the cropped image
    aspectRatio = h / w
    if aspectRatio > 1:
        k = imgSize / h
        wCal = math.ceil(k * w)
        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
        wGap = math.ceil((imgSize - wCal) / 2)
        imgWhite[:, wGap:wCal + wGap] = imgResize
    else:
        k = imgSize / w
        hCal = math.ceil(k * h)
        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
        hGap = math.ceil((imgSize - hCal) / 2)
        imgWhite[hGap:hCal + hGap, :] = imgResize
    
    # Get prediction
    prediction, index = classifier.getPrediction(imgWhite, draw=False)
    
    # Draw the result on the output image
    cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                  (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
    cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
    cv2.rectangle(imgOutput, (x-offset, y-offset),
                  (x + w+offset, y + h+offset), (255, 0, 255), 4)
    
    # Save the output image
    output_path = os.path.splitext(image_path)[0] + "_result.jpg"
    cv2.imwrite(output_path, imgOutput)
    
    print(f"Detected hand sign: {labels[index]}")
    print(f"Result image saved to: {output_path}")
    
    return labels[index]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect hand signs from an image')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    args = parser.parse_args()
    
    detect_hand_sign(args.image_path)
