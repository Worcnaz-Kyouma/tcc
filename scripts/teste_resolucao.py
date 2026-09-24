from pathlib import Path
from PIL import Image
from ultralytics import YOLO

BASE = Path.home() / "Documentos" / "Projects" / "TCC"
imagem = sorted((BASE / "coco" / "val2017").glob("*.jpg"))[0]

modelo = YOLO(str(BASE / "yolov8m.pt"))
img = Image.open(imagem).convert("RGB")
print(f"Imagem original: {img.size[0]}x{img.size[1]} (largura x altura)")

modelo(img, verbose=True, conf=0.001, iou=0.7)
