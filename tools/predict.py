"""Passe le modele sur des photos et ecrit un CSV : fichier, dossier, prediction, p_ouvert.

Sert a evaluer le modele sur les photos deja triees, et a pre-classer les autres.
"""
import argparse
import csv
import sys
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from roi import crop_gate  # noqa: E402


def load(f):
    # Exception large : Ultralytics masque un fichier absent derriere une erreur pi_heif
    try:
        return crop_gate(Image.open(f))
    except Exception as e:
        print(f"ignore : {f.name} : {type(e).__name__}")
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()

    model = YOLO(args.model)
    open_idx = next(i for i, n in model.names.items() if n == "ouvert")
    files = [f for d in args.dirs for f in sorted(Path(d).glob("*.jpg")) if f.stat().st_size > 0]

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "folder", "pred", "p_ouvert"])
        for i in range(0, len(files), args.batch):
            loaded = [(f, load(f)) for f in files[i:i + args.batch]]
            chunk = [f for f, c in loaded if c is not None]
            crops = [c for _, c in loaded if c is not None]
            if not crops:
                continue
            for f, r in zip(chunk, model.predict(crops, device=args.device, verbose=False)):
                p_open = float(r.probs.data[open_idx])
                w.writerow([f.name, f.parent.name, model.names[r.probs.top1], f"{p_open:.4f}"])
    print(f"{len(files)} photos -> {args.out}")


if __name__ == "__main__":
    main()
