"""Construit le dataset de classification a partir des photos triees.

Entree : un dossier par classe (ex. NouvellesPhotos/01-ferme, 02-ouvert).
Sortie : <out>/{train,val}/{ferme,ouvert}/ avec les images recadrees sur le portail.

La repartition se fait par journee entiere : des photos consecutives sont quasi identiques.

Les images sans date dans le nom (images generees, par exemple) vont toujours en train et
ne sont ni dupliquees ni sous-echantillonnees. Hors 16:9, seule la partie haute en 16:9
est conservee.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from roi import crop_gate  # noqa: E402

DATE_RE = re.compile(r"_(\d{8})_\d{6}\.jpg$")


def to_16x9(image):
    w, h = image.size
    if abs(w / h - 16 / 9) > 0.01:
        image = image.crop((0, 0, w, round(w * 9 / 16)))
    return image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="NouvellesPhotos")
    p.add_argument("--classes", nargs="+", default=["ferme=01-ferme", "ouvert=02-ouvert"],
                   help="nom_de_classe=sous-dossier[,sous-dossier...]")
    p.add_argument("--val-days", nargs="+", required=True, help="AAAAMMJJ des journees de validation")
    p.add_argument("--closed-step", type=int, default=2, help="train : garde 1 photo fermee sur N")
    p.add_argument("--open-repeat", type=int, default=8, help="train : duplique chaque photo ouverte N fois")
    p.add_argument("--out", default="work/dataset")
    args = p.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)

    for spec in args.classes:
        name, folders = spec.split("=")
        files = sorted((f for folder in folders.split(",") for ext in ("*.jpg", "*.png")
                        for f in (Path(args.src) / folder).glob(ext) if f.stat().st_size > 0),
                       key=lambda f: f.name)
        counts = {"train": 0, "val": 0, "sans date": 0}
        for i, f in enumerate(files):
            m = DATE_RE.search(f.name)
            split = ("val" if m.group(1) in args.val_days else "train") if m else "train"
            if m and split == "train" and name == "ferme" and i % args.closed_step:
                continue
            try:
                image = Image.open(f)
                crop = crop_gate(image if m else to_16x9(image))
            except OSError as e:
                print(f"ignore (illisible) : {f.name} : {e}")
                continue
            dest = out / split / name
            dest.mkdir(parents=True, exist_ok=True)
            repeat = args.open_repeat if m and split == "train" and name == "ouvert" else 1
            for r in range(repeat):
                suffix = f"_dup{r}" if r else ""
                safe_stem = re.sub(r"[^A-Za-z0-9_.-]", "_", f.stem)  # espaces et accents
                crop.save(dest / f"{safe_stem}{suffix}.jpg", quality=95)
            counts[split] += repeat
            counts["sans date"] += 0 if m else 1
        print(f"{name}: train={counts['train']} val={counts['val']} dont sans date={counts['sans date']} "
              f"(sur {len(files)} photos)")


if __name__ == "__main__":
    main()
