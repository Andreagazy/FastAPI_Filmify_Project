from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2

# Initialize FastAPI app
app = FastAPI()

# Load model and class names
def load_model_and_classes():
    model = tf.keras.models.load_model('prediksi_ekspresi.h5')
    with open('class_names.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

model, class_names = load_model_and_classes()

haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Preprocess image for face detection and emotion prediction
def preprocess_and_detect_face(image_bytes):
    # Load image from bytes
    img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        raise ValueError("No faces detected in the image.")

    # Select the first detected face
    x, y, w, h = faces[0]

    # Extract face region
    face = gray[y:y + h, x:x + w]

    # Resize to model input size (48x48)
    face_resized = cv2.resize(face, (48, 48))
    face_resized = face_resized.reshape(1, 48, 48, 1) / 255.0

    # Predict emotion
    predictions = model.predict(face_resized)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    return {
        "face_location": {
            "x": int(x), "y": int(y), "width": int(w), "height": int(h)
        },
        "predicted_class": predicted_class,
        "confidence": confidence,
        "confidence_scores": {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
    }

# Predict endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded file
        image_bytes = await file.read()

        # Process image and detect face
        face_prediction = preprocess_and_detect_face(image_bytes)

        # Return prediction
        return JSONResponse({"prediction": face_prediction})

    except ValueError as ve:
        return JSONResponse({"message": str(ve)}, status_code=400)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
