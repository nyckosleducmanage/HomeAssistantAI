"""Entraine le classifieur ouvert/ferme sur le GPU du Mac (MPS)."""
import argparse
from pathlib import Path

from ultralytics import YOLO

# Chemin absolu : Ultralytics imbrique un chemin relatif sous runs/classify/
RUNS = Path(__file__).resolve().parents[1] / "work" / "runs"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="work/dataset")
    p.add_argument("--name", default="v2")
    p.add_argument("--model", default="yolo26n-cls.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    YOLO(args.model).train(
        data=args.data,
        imgsz=224,
        epochs=args.epochs,
        # Pas d'arret anticipe ; utiliser last.pt
        patience=args.epochs,
        batch=64,
        device=args.device,
        project=str(RUNS),
        name=args.name,
        exist_ok=True,
        seed=0,
        # Augmentations compatibles avec un cadrage fixe
        fliplr=0.0,
        erasing=0.0,
        scale=0.15,
    )


if __name__ == "__main__":
    main()
