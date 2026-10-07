# Détection de l'état du portail pour Home Assistant

Ce projet fournit une API qui indique si un portail est ouvert ou fermé à partir d'une image de caméra. Home Assistant envoie une image à intervalle régulier, met à jour un capteur et déclenche une notification si le portail reste ouvert.

## Fonctionnement

1. `app/roi.py` recadre l'image sur le portail. Les coordonnées sont relatives et ne dépendent pas de la résolution. Le recadrage est limité à la hauteur du vantail afin qu'un véhicule stationné derrière le portail fermé ne soit pas interprété comme une ouverture.
2. Un modèle de classification Ultralytics (`yolo26n-cls`) classe l'image recadrée en `ouvert` ou `ferme`.
3. L'API renvoie `open` si la probabilité d'ouverture est supérieure ou égale à `OPEN_THRESHOLD` (0.7 par défaut), sinon `close`.

Le même recadrage est utilisé à l'entraînement et par l'API. Si la caméra est déplacée, mettez à jour `app/roi.py` puis réentraînez le modèle.

## Prérequis

- Un Mac Apple Silicon pour l'entraînement (GPU via MPS).
- Un Python **arm64**. Un Python x86_64 exécuté sous Rosetta ne dispose pas des versions récentes de torch ni de l'accès au GPU.
- Docker sur le serveur qui héberge l'API.

```bash
uv venv --python cpython-3.12-macos-aarch64-none .venv
uv pip install --python .venv/bin/python -r tools/requirements.txt
```

## Préparer les données

Les photos sont stockées dans `NouvellesPhotos/`. Ce dossier n'est pas versionné.

Le nom des fichiers doit contenir la date et l'heure de prise de vue au format `<prefixe>_AAAAMMJJ_HHMMSS.jpg`. La date est utilisée pour répartir les journées entre entraînement et validation.

| Dossier | Contenu |
|---|---|
| `01-ferme/` | Photos triées manuellement, portail fermé |
| `02-ouvert/` | Photos triées manuellement, portail ouvert, y compris ouverture piéton, entrouvert ou en mouvement |
| `03-a-valider/` | Corrections de tri proposées par le modèle et validées manuellement (`vers-ferme`, `vers-ouvert`) |
| `04-...`, `05-...` | Compléments ciblés (nuit, neige, véhicule stationné), avec des sous-dossiers `ferme` et `ouvert` |
| `10-`, `11-`, `12-` | Pré-tri par le modèle, voir [Pré-trier de nouvelles photos](#pré-trier-de-nouvelles-photos) |

Règle de tri : toute position autre que complètement fermé est classée `ouvert`.

## Entraîner le modèle

Une classe peut regrouper plusieurs dossiers.

```bash
.venv/bin/python tools/prepare_dataset.py \
  --classes "ferme=01-ferme,10-propose-ferme,12-a-verifier/ferme,03-a-valider/vers-ferme" \
            "ouvert=02-ouvert,11-propose-ouvert,12-a-verifier/ouvert,03-a-valider/vers-ouvert" \
  --val-days AAAAMMJJ AAAAMMJJ AAAAMMJJ \
  --closed-step 4 --open-repeat 5 --out work/dataset_final
.venv/bin/python tools/train.py --data work/dataset_final --name final
cp work/runs/final/weights/last.pt app/model_v2.pt
```

- `--val-days` : journées réservées à la validation. Choisissez des journées qui contiennent des ouvertures.
- `--closed-step` et `--open-repeat` : rééquilibrage des classes. La validation conserve la répartition réelle.
- Utilisez `last.pt` et non `best.pt` : la précision de validation atteint son maximum dès la première epoch, `best.pt` correspond alors à un modèle peu entraîné.
- Les images sans date dans le nom (images générées, par exemple) sont toujours placées en entraînement et ne sont pas dupliquées. Si leur format n'est pas 16:9, le script conserve la partie haute de l'image en 16:9.

## Pré-trier de nouvelles photos

Un modèle existant peut pré-trier les photos non triées. Seules les propositions incertaines nécessitent une vérification individuelle.

```bash
.venv/bin/python tools/predict.py --model app/model_v2.pt --dirs NouvellesPhotos --out work/pred.csv
.venv/bin/python tools/presort.py --csv work/pred.csv           # simulation
.venv/bin/python tools/presort.py --csv work/pred.csv --apply   # déplacement
```

| Dossier | Contenu |
|---|---|
| `10-propose-ferme/` | Probabilité d'ouverture inférieure à 0.01 |
| `11-propose-ouvert/` | Probabilité d'ouverture supérieure ou égale à 0.95 |
| `12-a-verifier/` | Valeurs intermédiaires et changements d'état isolés |

`tools/contact_sheet.py` génère une planche de miniatures pour faciliter la relecture.

## Déployer l'API

Le fichier du modèle (`app/model_v2.pt`) n'est pas versionné. Copiez-le avant la construction de l'image. Construisez l'image directement sur le serveur Docker pour éviter l'émulation d'architecture.

```bash
ssh <serveur> mkdir -p <dossier>/app
scp Dockerfile_Pour_API <serveur>:<dossier>/
scp app/main.py app/roi.py app/model_v2.pt <serveur>:<dossier>/app/
ssh <serveur> "cd <dossier> && docker build -f Dockerfile_Pour_API -t yolo-api:v2 ."
```

Exemple de fichier Docker Compose :

```yaml
services:
  yolo-api:
    image: yolo-api:v2
    container_name: yolo-api
    ports:
      - "9510:9510"
    environment:
      - PYTHONUNBUFFERED=1
      - OPEN_THRESHOLD=0.7
    restart: always
```

Pour revenir à la version précédente, remplacez le tag de l'image et redéployez.

Chaque analyse écrit une ligne dans les journaux du conteneur : résolution de l'image reçue, état et probabilité d'ouverture.

## Tester l'API

```bash
curl -X POST "http://<hote_api>:9510/analyze/" -F "file=@snapshot.jpg"
```

Réponse :

```json
{"state": "close", "confidences": [0.9998], "p_open": 0.0002}
```

| Champ | Description |
|---|---|
| `state` | `open` ou `close` |
| `confidences` | Confiance associée à l'état renvoyé |
| `p_open` | Probabilité d'ouverture |

Une image illisible renvoie le code HTTP 400.

## Intégrer à Home Assistant

### Enregistrer et envoyer l'image

L'entité caméra doit être celle qui a servi à constituer le jeu d'entraînement.

```yaml
alias: Envoyer une image à l'API
triggers:
  - minutes: /1
    trigger: time_pattern
actions:
  - action: camera.snapshot
    target:
      entity_id: camera.<entite_camera>
    data:
      filename: /config/www/<dossier>/snapshot.jpg
  - delay:
      seconds: 5
  - action: shell_command.send_image
    data: {}
```

### Notifier si le portail reste ouvert

```yaml
alias: Notifier si le portail est ouvert depuis 15 minutes
triggers:
  - trigger: state
    entity_id:
      - sensor.portail_ai_state
    to: open
    for:
      minutes: 15
actions:
  - action: notify.<service_de_notification>
    data:
      title: Alerte portail
      message: Le portail est ouvert depuis 15 minutes.
```

### Déclarer la commande shell

Ajoutez la section suivante au fichier `configuration.yaml` :

```yaml
shell_command:
  send_image: "python3 /config/send_ai.py"
```

### Script send_ai.py

Placez ce fichier dans le répertoire `/config` de Home Assistant. Le nom du capteur doit correspondre à celui utilisé par l'automatisation de notification.

```python
import requests

api_url = "http://<hote_api>:9510/analyze/"
file_path = "/config/www/<dossier>/snapshot.jpg"

ha_url = "http://<hote_ha>:8123/api/states/sensor.portail_ai_state"
ha_token = "<jeton_ha>"  # Ne pas versionner

try:
    with open(file_path, "rb") as image_file:
        response = requests.post(api_url, files={"file": image_file}, timeout=20)
    response.raise_for_status()
    result = response.json()
    state = result.get("state", "unknown")
    confidences = result.get("confidences", [])
except Exception:
    state = "error"
    confidences = []

headers = {
    "Authorization": f"Bearer {ha_token}",
    "Content-Type": "application/json",
}
data = {
    "state": state,
    "attributes": {
        "confidences": confidences,
    },
}

ha_response = requests.post(ha_url, headers=headers, json=data, timeout=20)

# 201 à la création du capteur, 200 lors des mises à jour
if ha_response.status_code in (200, 201):
    print("Capteur mis à jour.")
else:
    print(f"Erreur lors de la mise à jour du capteur : {ha_response.status_code}")
```

## Dépannage

| Symptôme | Cause probable | Action |
|---|---|---|
| `p_open` reste proche de 0.5 quel que soit l'état | L'image envoyée ne provient pas de la caméra utilisée pour l'entraînement, ou le cadrage a changé | Vérifiez l'entité caméra de l'automatisation. Comparez `p_open` dans les journaux avec le score du modèle sur une photo prise à la même minute. |
| Ouvertures non détectées après un déplacement de la caméra | Le recadrage ne correspond plus | Mettez à jour `app/roi.py` et réentraînez le modèle. |
| Avertissement `pi-heif` dans les journaux | Fichier reçu non lisible comme image | Aucune. L'installation automatique de paquets est désactivée. |
