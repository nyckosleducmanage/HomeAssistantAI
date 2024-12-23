# Modèle IA de détection d'ouverture de Portail pour Home Assistant


Un détecteur d'ouverture en extérieur pose divers problèmes de maintenance. Une caméra est cependant braquée dans sa direction, et c'est tout naturellement que j'ai eu l'idée de créer un modèle afin de déterminer s'il est ouvert ou fermé.

J'ai donc pris une photo toutes les 5 minutes dans différents états : ouverture complète, ouverture piéton et fermeture. Un total d'environ 350 à 400 photos a été annoté via "Label Studio" avec l'interface de labellisation ci-dessous :

````LABELSTUDIO
<View>
    <Image name="image" value="$image" />
    <RectangleLabels name="label" toName="image">
        <Label value="open" background="green" />
        <Label value="close" background="red" />
    </RectangleLabels>
</View>
````

Ensuite, j'exporte les données depuis Label Studio au format YOLO (et non YOLOv8 OBB), pour les répartir à 80 % dans le dossier "train" et à 20 % dans le dossier "val".

Voici la Structure des données :

````KOTLIN
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   └── val/
│       ├── image3.jpg
├── labels/
│   ├── train/
│   │   ├── image1.txt
│   │   ├── image2.txt
│   └── val/
│       ├── image3.txt
````

# Entrainement du model

Dans le Dockerfile, modifier le CMD :

````Dockerfile
CMD ["python", "train.py"]
````

## Build du model pour lancer l'entrainement

````Bash
docker build --platform=linux/amd64 -t ha-ai-portail:v1 . > build.log 2>&1
````

## Démarrage du container pour lancer l'entrainement

````Bash
docker run --rm --platform=linux/amd64 ha-ai-portail:v1
````

## Logs d'entrainement

Retour type lors de l'entrainement du modèle.

````Bash
2024-12-22 13:08:56 Creating new Ultralytics Settings v0.0.6 file ✅ 
2024-12-22 13:08:56 View Ultralytics Settings with 'yolo settings' or at '/root/.config/Ultralytics/settings.json'
2024-12-22 13:08:56 Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings.
2024-12-22 13:08:56 Downloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt to 'yolov8n.pt'...
2024-12-22 13:08:57 New https://pypi.org/project/ultralytics/8.3.53 available 😃 Update with 'pip install -U ultralytics'
2024-12-22 13:08:59 Ultralytics 8.3.52 🚀 Python-3.9.21 torch-2.5.1+cu124 CPU (-)
2024-12-22 13:08:59 engine/trainer: task=detect, mode=train, model=yolov8n.pt, data=/app/dataset/data.yaml, epochs=30, time=None, patience=100, batch=16, imgsz=640, save=True, save_period=-1, cache=False, device=None, workers=8, project=None, name=train, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=runs/detect/train
2024-12-22 13:09:00 Downloading https://ultralytics.com/assets/Arial.ttf to '/root/.config/Ultralytics/Arial.ttf'...
2024-12-22 13:09:00 Overriding model.yaml nc=80 with nc=2
2024-12-22 13:09:00 
2024-12-22 13:09:00                    from  n    params  module                                       arguments                     
2024-12-22 13:09:00   0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
2024-12-22 13:09:00   1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
2024-12-22 13:09:00   2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]             
2024-12-22 13:09:00   3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
2024-12-22 13:09:00   4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]             
2024-12-22 13:09:00   5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
2024-12-22 13:09:00   6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           
2024-12-22 13:09:00   7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
2024-12-22 13:09:00   8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]           
2024-12-22 13:09:00   9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
2024-12-22 13:09:00  10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
2024-12-22 13:09:00  11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
2024-12-22 13:09:00  12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 
2024-12-22 13:09:00  13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
2024-12-22 13:09:00  14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
2024-12-22 13:09:00  15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]                  
2024-12-22 13:09:00  16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
2024-12-22 13:09:00  17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
2024-12-22 13:09:00  18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]                 
2024-12-22 13:09:00  19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
2024-12-22 13:09:00  20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
2024-12-22 13:08:57 
  0%|          | 0.00/6.25M [00:00<?, ?B/s]
 88%|████████▊ | 5.50M/6.25M [00:00<00:00, 56.3MB/s]
100%|██████████| 6.25M/6.25M [00:00<00:00, 43.3MB/s]
2024-12-22 13:09:00 
  0%|          | 0.00/755k [00:00<?, ?B/s]
100%|██████████| 755k/755k [00:00<00:00, 35.1MB/s]
2024-12-22 13:09:00 [W1222 12:09:00.906830329 NNPACK.cpp:61] Could not initialize NNPACK! Reason: Unsupported hardware.
2024-12-22 13:09:01 
train: Scanning /app/dataset/labels/train...:   0%|          | 0/315 [00:00<?, ?it/s]
train: Scanning /app/dataset/labels/train... 78 images, 0 backgrounds, 0 corrupt:  25%|██▍       | 78/315 [00:00<00:00, 670.79it/s]
train: Scanning /app/dataset/labels/train... 166 images, 1 backgrounds, 0 corrupt:  53%|█████▎    | 166/315 [00:00<00:00, 781.12it/s]
train: Scanning /app/dataset/labels/train... 280 images, 1 backgrounds, 0 corrupt:  89%|████████▉ | 280/315 [00:00<00:00, 934.25it/s]
train: Scanning /app/dataset/labels/train... 315 images, 1 backgrounds, 0 corrupt: 100%|██████████| 315/315 [00:00<00:00, 925.86it/s]
2024-12-22 13:09:01 
val: Scanning /app/dataset/labels/val...:   0%|          | 0/113 [00:00<?, ?it/s]
val: Scanning /app/dataset/labels/val... 113 images, 1 backgrounds, 0 corrupt: 100%|██████████| 113/113 [00:00<00:00, 1297.32it/s]
2024-12-22 13:10:53 
  0%|          | 0/20 [00:00<?, ?it/s]
       1/30         0G      2.142      5.237      1.503         28        640:   0%|          | 0/20 [00:08<?, ?it/s]
       1/30         0G      2.142      5.237      1.503         28        640:   5%|▌         | 1/20 [00:08<02:42,  8.56s/it]
       1/30         0G      2.139      4.958      1.504         34        640:   5%|▌         | 1/20 [00:14<02:42,  8.56s/it]
       1/30         0G      2.139      4.958      1.504         34        640:  10%|█         | 2/20 [00:14<02:06,  7.01s/it]
       1/30         0G      1.983      4.795      1.459         26        640:  10%|█         | 2/20 [00:19<02:06,  7.01s/it]
       1/30         0G      1.983      4.795      1.459         26        640:  15%|█▌        | 3/20 [00:19<01:45,  6.23s/it]
       1/30         0G      1.865      4.605      1.396         34        640:  15%|█▌        | 3/20 [00:26<01:45,  6.23s/it]
       1/30         0G      1.865      4.605      1.396         34        640:  20%|██        | 4/20 [00:26<01:42,  6.43s/it]
       1/30         0G      1.772      4.501      1.349         31        640:  20%|██        | 4/20 [00:31<01:42,  6.43s/it]
       1/30         0G      1.772      4.501      1.349         31        640:  25%|██▌       | 5/20 [00:31<01:29,  5.94s/it]
       1/30         0G       1.62      4.412      1.282         26        640:  25%|██▌       | 5/20 [00:36<01:29,  5.94s/it]
       1/30         0G       1.62      4.412      1.282         26        640:  30%|███       | 6/20 [00:36<01:17,  5.51s/it]
       1/30         0G      1.534      4.303      1.243         30        640:  30%|███       | 6/20 [00:40<01:17,  5.51s/it]
       1/30         0G      1.534      4.303      1.243         30        640:  35%|███▌      | 7/20 [00:40<01:08,  5.25s/it]
       1/30         0G      1.463      4.236      1.207         34        640:  35%|███▌      | 7/20 [00:45<01:08,  5.25s/it]
       1/30         0G      1.463      4.236      1.207         34        640:  40%|████      | 8/20 [00:45<01:01,  5.14s/it]
       1/30         0G      1.405      4.132      1.179         38        640:  40%|████      | 8/20 [00:50<01:01,  5.14s/it]
       1/30         0G      1.405      4.132      1.179         38        640:  45%|████▌     | 9/20 [00:50<00:56,  5.13s/it]
       1/30         0G       1.34      4.005      1.149         26        640:  45%|████▌     | 9/20 [00:56<00:56,  5.13s/it]
       1/30         0G       1.34      4.005      1.149         26        640:  50%|█████     | 10/20 [00:56<00:53,  5.33s/it]
       1/30         0G      1.291      3.889      1.126         31        640:  50%|█████     | 10/20 [01:01<00:53,  5.33s/it]
       1/30         0G      1.291      3.889      1.126         31        640:  55%|█████▌    | 11/20 [01:01<00:46,  5.20s/it]
       1/30         0G      1.257      3.782      1.107         33        640:  55%|█████▌    | 11/20 [01:07<00:46,  5.20s/it]
       1/30         0G      1.257      3.782      1.107         33        640:  60%|██████    | 12/20 [01:07<00:44,  5.50s/it]
       1/30         0G      1.219      3.657      1.088         33        640:  60%|██████    | 12/20 [01:13<00:44,  5.50s/it]
       1/30         0G      1.219      3.657      1.088         33        640:  65%|██████▌   | 13/20 [01:13<00:37,  5.41s/it]
       1/30         0G       1.18      3.546      1.072         29        640:  65%|██████▌   | 13/20 [01:17<00:37,  5.41s/it]
       1/30         0G       1.18      3.546      1.072         29        640:  70%|███████   | 14/20 [01:17<00:31,  5.21s/it]
       1/30         0G      1.159      3.435      1.058         44        640:  70%|███████   | 14/20 [01:22<00:31,  5.21s/it]
       1/30         0G      1.159      3.435      1.058         44        640:  75%|███████▌  | 15/20 [01:22<00:25,  5.04s/it]
       1/30         0G       1.13      3.332      1.041         38        640:  75%|███████▌  | 15/20 [01:27<00:25,  5.04s/it]
       1/30         0G       1.13      3.332      1.041         38        640:  80%|████████  | 16/20 [01:27<00:20,  5.15s/it]
       1/30         0G      1.102      3.256       1.03         26        640:  80%|████████  | 16/20 [01:32<00:20,  5.15s/it]
       1/30         0G      1.102      3.256       1.03         26        640:  85%|████████▌ | 17/20 [01:32<00:15,  5.07s/it]
       1/30         0G      1.072      3.183       1.02         26        640:  85%|████████▌ | 17/20 [01:38<00:15,  5.07s/it]
       1/30         0G      1.072      3.183       1.02         26        640:  90%|█████████ | 18/20 [01:38<00:10,  5.21s/it]
       1/30         0G      1.048      3.112      1.008         37        640:  90%|█████████ | 18/20 [01:43<00:10,  5.21s/it]
       1/30         0G      1.048      3.112      1.008         37        640:  95%|█████████▌| 19/20 [01:43<00:05,  5.29s/it]
       1/30         0G      1.033      3.064     0.9995         23        640:  95%|█████████▌| 19/20 [01:47<00:05,  5.29s/it]
       1/30         0G      1.033      3.064     0.9995         23        640: 100%|██████████| 20/20 [01:47<00:00,  4.83s/it]
       1/30         0G      1.033      3.064     0.9995         23        640: 100%|██████████| 20/20 [01:47<00:00,  5.38s/it]
2024-12-22 13:11:04 
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   0%|          | 0/4 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  25%|██▌       | 1/4 [00:03<00:10,  3.63s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  50%|█████     | 2/4 [00:06<00:06,  3.07s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  75%|███████▌  | 3/4 [00:09<00:02,  2.92s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:10<00:00,  2.49s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:10<00:00,  2.72s/it]
2024-12-22 13:09:00  21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 
2024-12-22 13:09:00  22        [15, 18, 21]  1    751702  ultralytics.nn.modules.head.Detect           [2, [64, 128, 256]]           
2024-12-22 13:09:01 Model summary: 225 layers, 3,011,238 parameters, 3,011,222 gradients, 8.2 GFLOPs
2024-12-22 13:09:01 
2024-12-22 13:09:01 Transferred 319/355 items from pretrained weights
2024-12-22 13:09:01 Freezing layer 'model.22.dfl.conv.weight'
2024-12-22 13:09:01 train: New cache created: /app/dataset/labels/train.cache
2024-12-22 13:09:01 val: New cache created: /app/dataset/labels/val.cache
2024-12-22 13:09:03 Plotting labels to runs/detect/train/labels.jpg... 
2024-12-22 13:09:05 optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
2024-12-22 13:09:06 optimizer: AdamW(lr=0.001667, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)
2024-12-22 13:09:06 Image sizes 640 train, 640 val
2024-12-22 13:09:06 Using 0 dataloader workers
2024-12-22 13:09:06 Logging results to runs/detect/train
2024-12-22 13:09:06 Starting training for 30 epochs...
2024-12-22 13:09:06 
2024-12-22 13:09:06       Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
2024-12-22 13:11:04                    all        113        112     0.0033          1       0.65      0.562
2024-12-22 13:11:05 
2024-12-22 13:11:05       Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
2024-12-22 13:13:05                    all        113        112    0.00331          1      0.505      0.429
2024-12-22 13:13:06 
2024-12-22 13:13:06       Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
2024-12-22 13:12:55 
  0%|          | 0/20 [00:00<?, ?it/s]
````

# Passer le model en production

Dans le Dockerfile, modifier le paramètre CMD pour utiliser le fichier detect.py.

````Dockerfile
CMD ["python", "detect.py"]
````

## Build du model pour la production

````Bash
docker build --platform=linux/amd64 -t ha-ai-portail:v1 . > build.log 2>&1
````

## Commande pour tester l'API du modèle
Il est nécessaire de monter le container avec le Docker Compose pour ce test.

````Bash
(envaiha) nicolas@MacBook-Pro-170 Yolov8 % curl -X POST "http://192.168.95.198:9510/analyze/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@camera_garage_snapshot.jpg"
````

# Exemple de retour du test

````Bash
2024-12-22T15:29:06.529925205Z 0: 384x640 1 close, 39.5ms
2024-12-22T15:29:06.530005502Z Speed: 2.2ms preprocess, 39.5ms inference, 0.6ms postprocess per image at shape (1, 3, 384, 640)
2024-12-22T15:29:06.530338406Z INFO:     192.168.95.190:44338 - "POST /analyze/ HTTP/1.1" 200 OK
2024-12-22T15:30:06.366469651Z 
2024-12-22T15:30:06.438789110Z 0: 384x640 1 close, 64.7ms
2024-12-22T15:30:06.438846168Z Speed: 2.2ms preprocess, 64.7ms inference, 3.5ms postprocess per image at shape (1, 3, 384, 640)
2024-12-22T15:30:06.439214425Z INFO:     192.168.95.190:44738 - "POST /analyze/ HTTP/1.1" 200 OK
2024-12-22T15:31:06.548125591Z 
2024-12-22T15:31:06.594002620Z 0: 384x640 1 close, 39.3ms
2024-12-22T15:31:06.594081770Z Speed: 2.2ms preprocess, 39.3ms inference, 2.8ms postprocess per image at shape (1, 3, 384, 640)
2024-12-22T15:31:06.594390134Z INFO:     192.168.95.190:45142 - "POST /analyze/ HTTP/1.1" 200 OK
````



## Docker Compose

````yaml
version: "3.9"

services:
  yolo-api:
    image: yolo-api:v1
    container_name: yolo-api
    build:
      context: .
    ports:
      - "9510:9510"
    environment:
      - PYTHONUNBUFFERED=1
    restart: always

networks:
  default:
    name: network
````


# Automatisation HA

## Automatisation pour envoyer l'image a l'API du modèle

````yaml
alias: Envoyer une image à l'API YOLO
triggers:
  - minutes: /1
    trigger: time_pattern
actions:
  - data:
      entity_id: camera.jardin_g4_pro_high
      filename: /config/www/ai_portail/camera_garage_snapshot.jpg
    action: camera.snapshot
  - delay:
      hours: 0
      minutes: 0
      seconds: 5
      milliseconds: 0
  - action: shell_command.send_image
    data: {}
````

## Automatisation pour envoyer les notifications

````yaml
alias: Notifier si le portail est ouvert depuis 15 minutes
description: ""
triggers:
  - entity_id:
      - sensor.portail_ai_state
    to: open
    trigger: state
    for:
      hours: 0
      minutes: 15
      seconds: 0
conditions: []
actions:
  - action: notify.mobile_app_ipad_pro_de_nicolas
    data:
      message: Le portail est ouvert depuis 15 minutes
      title: Alerte Portail
  - data:
      target:
        - media_player.salon
      message: Attention, le portail est ouvert depuis 15 minutes.
      data:
        type: announce
        method: all
    action: notify.alexa_media
````

## modification du fichier configuration.yaml

Ajouter cette section

````yaml
shell_command:
  send_image: "python3 /config/send_ai.py"
````

## Fichier send_ai.py

Fichier à placer dans le répertoire /config de Home Assistant (HA). Il envoie l'image à l'API du modèle, puis met à jour le capteur.

````python
import requests

yolo_url = "http://IPDUCONTAINERAPIMODEL:9510/analyze/"
file_path = "/config/www/ai_portail/camera_garage_snapshot.jpg"

ha_url = "http://IPDEHA:8123/api/states/sensor.portail_yolo_state"
ha_token = "MONTOKENHA"  # Remplacez par votre token Home Assistant

with open(file_path, "rb") as image_file:
    files = {"file": image_file}
    response = requests.post(yolo_url, files=files)

if response.status_code == 200:
    yolo_result = response.json()
    state = yolo_result.get("state", "unknown")
    confidences = yolo_result.get("confidences", [])
else:
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

ha_response = requests.post(ha_url, headers=headers, json=data)

if ha_response.status_code == 200:
    print("Capteur mis à jour avec succès.")
else:
    print(f"Erreur lors de la mise à jour du capteur : {ha_response.status_code}")
````
