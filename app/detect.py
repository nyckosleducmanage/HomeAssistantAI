from ultralytics import YOLO
import os
import time
import json
from PIL import Image

model = YOLO('/app/best.pt')

image_directory = '/portail'
output_directory = '/portail/results'

os.makedirs(output_directory, exist_ok=True)

def analyze_images():
    """
    Analyse toutes les images dans le répertoire spécifié.
    """
    image_files = [f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    results_summary = {}

    for image_file in image_files:
        image_path = os.path.join(image_directory, image_file)
        
        print(f"Analyse de l'image : {image_file}")
        results = model(image_path)

        detected_classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        if 0 in detected_classes:
            state = "open"
        elif 1 in detected_classes:
            state = "close"
        else:
            state = "unknown"

        for result in results:
            output_path = os.path.join(output_directory, os.path.basename(image_path))
            annotated_image = Image.fromarray(result.plot())
            annotated_image.save(output_path)

        results_summary[image_file] = {
            "state": state,
            "confidences": confidences.tolist()
        }

    summary_path = os.path.join(output_directory, 'results_summary.json')
    with open(summary_path, 'w') as json_file:
        json.dump(results_summary, json_file, indent=4)
    
    print(f"Résumé des résultats sauvegardé dans : {summary_path}")

if __name__ == "__main__":
    while True:
        print("Début de l'analyse des images...")
        analyze_images()
        print("Analyse terminée.")
