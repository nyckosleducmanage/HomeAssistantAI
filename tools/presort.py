"""Range les photos non triees dans des dossiers de proposition, d'apres un CSV de predict.py.

  10-propose-ferme   : ferme avec p_ouvert < --low (survol rapide)
  11-propose-ouvert  : ouvert avec p_ouvert >= --high (a regarder une par une)
  12-a-verifier      : entre les deux, ou "ferme" entoure de deux photos "ouvert"

Seuls les fichiers encore a la racine sont deplaces : un tri manuel en parallele ne
gene pas, un fichier deja deplace est simplement ignore. Sans --apply, rien ne bouge.
"""
import argparse
import csv
from pathlib import Path

BUCKETS = ("10-propose-ferme", "11-propose-ouvert", "12-a-verifier")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--root", default="NouvellesPhotos")
    p.add_argument("--low", type=float, default=0.01)
    p.add_argument("--high", type=float, default=0.95)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    root = Path(args.root)

    # Chronologie complete : le tri humain pour les photos triees, la prediction sinon.
    state = {f.name: "ferme" for f in (root / "01-ferme").glob("*.jpg")}
    state.update({f.name: "ouvert" for f in (root / "02-ouvert").glob("*.jpg")})
    preds = {r["file"]: float(r["p_ouvert"]) for r in csv.DictReader(open(args.csv))}
    for name, p_open in preds.items():
        state.setdefault(name, "ouvert" if p_open >= 0.5 else "ferme")
    timeline = sorted(state)
    pos = {name: i for i, name in enumerate(timeline)}

    def between_opens(name):
        i = pos[name]
        return 0 < i < len(timeline) - 1 and \
            state[timeline[i - 1]] == state[timeline[i + 1]] == "ouvert"

    plan = {b: [] for b in BUCKETS}
    for name, p_open in preds.items():
        if not (root / name).exists():
            continue
        if p_open >= args.high:
            plan["11-propose-ouvert"].append(name)
        elif p_open < args.low and not between_opens(name):
            plan["10-propose-ferme"].append(name)
        else:
            plan["12-a-verifier"].append(name)

    for bucket, names in plan.items():
        print(f"{bucket}: {len(names)}")
        if not args.apply:
            continue
        (root / bucket).mkdir(exist_ok=True)
        for name in names:
            src, dst = root / name, root / bucket / name
            if src.exists() and not dst.exists():
                src.rename(dst)
    if not args.apply:
        print("(a blanc, relancer avec --apply)")


if __name__ == "__main__":
    main()
