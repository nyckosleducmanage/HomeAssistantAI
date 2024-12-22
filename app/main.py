from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import os
import json
from PIL import Image
import numpy as np
from io import BytesIO

# Initialisation de l'application FastAPI
app = FastAPI()

# Charger le modèle YOLOv8
model = YOLO("/app/model_v1.pt")

@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    """
    Endpoint pour analyser une image envoyée en POST.
    """
    try:
        # Lire l'image envoyée
        contents = await file.read()
        image = Image.open(BytesIO(contents))

        # Exécuter la détection
        results = model(np.array(image))

        # Récupérer les classes et les scores
        detected_classes = results[0].boxes.cls.cpu().numpy()  # Classes détectées
        confidences = results[0].boxes.conf.cpu().numpy()  # Scores de confiance

        # Déterminer l'état
        state = "unknown"
        if 0 in detected_classes:  # Classe 0 correspond à "open"
            state = "close"
        elif 1 in detected_classes:  # Classe 1 correspond à "close"
            state = "open"

        # Réponse JSON
        response = {
            "state": state,
            "confidences": confidences.tolist(),
        }
        return response

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "UP"}
