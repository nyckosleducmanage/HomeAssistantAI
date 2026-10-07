from PIL import Image

# Zone du portail en coordonnees relatives (gauche, haut, droite, bas).
# Limitee a la hauteur du vantail pour masquer un vehicule stationne derriere.
# A mettre a jour si la camera est deplacee, puis reentrainer le modele.
ROI = (0.40, 0.38, 0.80, 0.72)

# Image carree : le classifieur applique un recadrage central carre.
SIZE = 224


def crop_gate(image: Image.Image) -> Image.Image:
    """Recadre l'image sur le portail (entrainement et API)."""
    image = image.convert("RGB")
    w, h = image.size
    box = (round(ROI[0] * w), round(ROI[1] * h), round(ROI[2] * w), round(ROI[3] * h))
    return image.crop(box).resize((SIZE, SIZE), Image.BILINEAR)
