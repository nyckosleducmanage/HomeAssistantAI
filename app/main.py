import os
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from ultralytics import YOLO

from roi import crop_gate

# Initialisation de l'application FastAPI
app = FastAPI()

# Classifieur ouvert/ferme (YOLO classification), entraine sur le crop de roi.py
model = YOLO(os.environ.get("MODEL_PATH", "/app/model_v2.pt"))

# Seuil de probabilite d'ouverture au-dela duquel l'etat renvoye est "open"
OPEN_THRESHOLD = float(os.environ.get("OPEN_THRESHOLD", "0.7"))

# Index de la classe "ouvert" (nom du dossier d'entrainement)
OPEN_IDX = next(i for i, name in model.names.items() if name == "ouvert")


@app.post("/analyze/")
def analyze_image(file: UploadFile = File(...)):
    """
    Endpoint pour analyser une image envoyée en POST.
    """
    try:
        image = Image.open(BytesIO(file.file.read()))
        image.load()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image illisible : {e}")

    # Image PIL passee telle quelle : un tableau numpy serait lu comme du BGR
    result = model(crop_gate(image), device="cpu", verbose=False)[0]
    p_open = float(result.probs.data[OPEN_IDX])
    state = "open" if p_open >= OPEN_THRESHOLD else "close"
    print(f"{image.width}x{image.height} {state} p_open={p_open:.4f}", flush=True)

    return {
        "state": state,
        # Liste conservee pour send_ai.py : confiance de l'etat renvoye
        "confidences": [round(p_open if state == "open" else 1 - p_open, 4)],
        "p_open": round(p_open, 4),
    }


@app.get("/")
def root():
    return {"message": "UP"}
