import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load OpenCV's Deep Learning Face Detector
face_proto = "models/opencv_face_detector.pbtxt"
face_model = "models/opencv_face_detector_uint8.pb"

# Load the model
net = cv2.dnn.readNetFromTensorflow(face_model, face_proto)

# Manually specify image paths (since `os` cannot be used)
image_paths = [
    "images/person.jpg",
    "images/person2.jpg",
    "images/person3.jpg"
]

# Load and process stored face images
stored_faces = []
labels = []
face_size = (100, 100)  # Standardized face size
label_counter = 0

for image_path in image_paths:
    stored_image = cv2.imread(image_path)

    if stored_image is not None:
        height, width = stored_image.shape[:2]
        blob = cv2.dnn.blobFromImage(stored_image, scalefactor=1.0, size=(300, 300), mean=(104, 117, 123), swapRB=False)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x, y, x2, y2) = box.astype("int")
                face = stored_image[y:y2, x:x2]

                # Normalize the face (resize, grayscale, equalize)
                gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                gray_face = cv2.resize(gray_face, face_size)
                gray_face = cv2.equalizeHist(gray_face)

                stored_faces.append(gray_face)
                labels.append(label_counter)

    label_counter += 1

if len(stored_faces) == 0:
    print("Error: No valid faces detected in the 'images/' folder!")
    exit()

# Train the LBPH Recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(stored_faces, np.array(labels))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    """Receives image from frontend, processes it, and returns face match result"""
    try:
        data = request.json["image"]
        image_data = base64.b64decode(data.split(",")[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Process the received image
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104, 117, 123), swapRB=False)
        net.setInput(blob)
        detections = net.forward()

        live_gray_face = None
        box_coords = None
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x, y, x2, y2) = box.astype("int")
                face = image[y:y2, x:x2]

                # Normalize the live face
                live_gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                live_gray_face = cv2.resize(live_gray_face, face_size)
                live_gray_face = cv2.equalizeHist(live_gray_face)
                box_coords = [x, y, x2, y2]  # Save face coordinates

        if live_gray_face is None:
            return jsonify({"match": False, "message": "No face detected", "box": None})

        # Compare live face with stored faces
        label, confidence = face_recognizer.predict(live_gray_face)
        match_threshold = 110  # Adjusted threshold for better recognition

        match = confidence < match_threshold
        response = {
            "match": match,
            "message": "Face Matched" if match else "Face Not Matched",
            "box": [int(coord) for coord in box_coords] if box_coords else None
        }
        print("Server Response:", response)  # Debugging log
        
        return jsonify(response)
    
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"match": False, "message": "Error processing request", "box": None})

# Run the app (uncomment this if running locally)
if __name__ == "__main__":
    app.run(debug=True)










'''import cv2
import numpy as np
import os
import base64
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load OpenCV's Deep Learning Face Detector
face_proto = "models/opencv_face_detector.pbtxt"
face_model = "models/opencv_face_detector_uint8.pb"

if not os.path.exists(face_model) or not os.path.exists(face_proto):
    print("Error: Model files not found! Place them in the 'models/' directory.")
    exit()

# Load the model
net = cv2.dnn.readNetFromTensorflow(face_model, face_proto)

# Load and process stored face images
image_folder = "images/"
stored_faces = []
labels = []
face_size = (100, 100)  # Ensure all faces are the same size
label_counter = 0

for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)
    stored_image = cv2.imread(image_path)

    if stored_image is not None:
        height, width = stored_image.shape[:2]
        blob = cv2.dnn.blobFromImage(stored_image, scalefactor=1.0, size=(300, 300), mean=(104, 117, 123), swapRB=False)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x, y, x2, y2) = box.astype("int")
                face = stored_image[y:y2, x:x2]

                # Normalize the face (resize, grayscale, equalize)
                gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                gray_face = cv2.resize(gray_face, face_size)
                gray_face = cv2.equalizeHist(gray_face)

                stored_faces.append(gray_face)
                labels.append(label_counter)

    label_counter += 1

if len(stored_faces) == 0:
    print("Error: No valid faces detected in the 'images/' folder!")
    exit()

# Train the LBPH Recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(stored_faces, np.array(labels))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    """Receives image from frontend, processes it, and returns face match result"""
    try:
        data = request.json["image"]
        image_data = base64.b64decode(data.split(",")[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Process the received image
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104, 117, 123), swapRB=False)
        net.setInput(blob)
        detections = net.forward()

        live_gray_face = None
        box_coords = None
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x, y, x2, y2) = box.astype("int")
                face = image[y:y2, x:x2]

                # Normalize the live face
                live_gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                live_gray_face = cv2.resize(live_gray_face, face_size)
                live_gray_face = cv2.equalizeHist(live_gray_face)
                box_coords = [x, y, x2, y2]  # Save face coordinates

        if live_gray_face is None:
            return jsonify({"match": False, "message": "No face detected", "box": None})

        # Compare live face with stored faces
        label, confidence = face_recognizer.predict(live_gray_face)
        match_threshold = 110  # Increased threshold for better recognition

        match = confidence < match_threshold
        # response = {"match": match, "message": "Face Matched" if match else "Face Not Matched", "box": box_coords}
        response = {
    "match": match,
    "message": "Face Matched" if match else "Face Not Matched",
    "box": [int(coord) for coord in box_coords] if box_coords else None
}
        print("Server Response:", response)  # Debugging log
        
        return jsonify(response)
    
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"match": False, "message": "Error processing request", "box": None})

if __name__ == "__main__":
    app.run(debug=True)'''