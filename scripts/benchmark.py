"""
benchmark.py
Nicolas Almeida Prado, UEPG:2026

Tips:
    python benchmark.py
    python benchmark.py --completo
    python benchmark.py --modelo yolo
    python benchmark.py --modelo ssd
    python benchmark.py --modelo faster
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Configuração central
# ---------------------------------------------------------------------------

BASE_DIR   = Path.home() / "Documentos" / "Projects" / "TCC"
COCO_DIR   = BASE_DIR / "coco" / "val2017"
ANNOT_FILE = BASE_DIR / "coco" / "annotations" / "instances_val2017.json"
OUTPUT_DIR = BASE_DIR / "resultados"

YOLO_PESOS = "yolov8m.pt"

# Classes-alvo: ID COCO oficial → nome
CLASSES_ALVO_COCO = {1: "person", 3: "car"}

# Aquecimento antes de medir FPS
N_WARMUP = 20

# Subconjunto piloto para testes rápidos
N_PILOTO = 100

# Limiar de confiança para salvar predições (baixo para preservar curva PR)
CONF_THRESHOLD = 0.001

# ---------------------------------------------------------------------------
# Utilitários comuns
# ---------------------------------------------------------------------------

def listar_imagens(limite: int | None) -> list[Path]:
    imagens = sorted(COCO_DIR.glob("*.jpg"))
    if not imagens:
        raise RuntimeError(f"Nenhuma imagem encontrada em {COCO_DIR}")
    return imagens if limite is None else imagens[:limite]


def carregar_id_map() -> dict[str, int]:
    with open(ANNOT_FILE) as f:
        dados = json.load(f)
    return {img["file_name"]: img["id"] for img in dados["images"]}


def coco_bbox(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    return [x1, y1, x2 - x1, y2 - y1]


def avaliar_map(predicoes: list[dict], annot_file: Path, cat_ids: list[int]) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(annot_file))

    if not predicoes:
        print("  AVISO: nenhuma predicao gerada, metricas zeradas.")
        return {"mAP_05": 0.0, "precision": 0.0, "recall": 0.0}

    coco_dt = coco_gt.loadRes(predicoes)
    avaliador = COCOeval(coco_gt, coco_dt, "bbox")
    avaliador.params.catIds = cat_ids
    avaliador.params.imgIds = sorted({p["image_id"] for p in predicoes})

    avaliador.evaluate()
    avaliador.accumulate()
    avaliador.summarize()

    # stats[1] = AP @IoU=0.50, conforme documentacao do COCOeval
    mAP = float(avaliador.stats[1])

    precisoes = []
    recalls   = []
    for cat_id in cat_ids:
        p, r = _extrair_pr_maximo_f1(avaliador, cat_id)
        precisoes.append(p)
        recalls.append(r)

    return {
        "mAP_05":    round(mAP, 4),
        "precision": round(sum(precisoes) / len(precisoes), 4),
        "recall":    round(sum(recalls)   / len(recalls),   4),
    }


def _extrair_pr_maximo_f1(avaliador, cat_id: int) -> tuple[float, float]:
    import numpy as np

    p = avaliador.eval["precision"]

    cat_ids_ordenados = avaliador.params.catIds
    if cat_id not in cat_ids_ordenados:
        return 0.0, 0.0
    k_idx = cat_ids_ordenados.index(cat_id)

    rec_thrs = avaliador.params.recThrs

    curva_p = p[0, :, k_idx, 0, 2]
    curva_r = np.array(rec_thrs)

    validos = curva_p >= 0
    if not validos.any():
        return 0.0, 0.0

    curva_p = curva_p[validos]
    curva_r = curva_r[validos]

    f1 = 2 * curva_p * curva_r / (curva_p + curva_r + 1e-9)
    idx_max = int(f1.argmax())
    return float(curva_p[idx_max]), float(curva_r[idx_max])


def salvar_predicoes(predicoes: list[dict], caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w") as f:
        json.dump(predicoes, f)
    print(f"  Predições salvas: {caminho}")


def warmup_cpu(modelo_fn, n: int, imagens: list[Path], device: torch.device) -> None:
    for path in imagens[:n]:
        img = Image.open(path).convert("RGB")
        modelo_fn(img)
    if device.type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# YOLOv8
# ---------------------------------------------------------------------------

def benchmark_yolo(imagens: list[Path], id_map: dict, device: torch.device) -> dict:
    from ultralytics import YOLO

    YOLO_PARA_COCO = {0: 1, 2: 3}

    print("\n[YOLOv8] Carregando modelo...")
    modelo = YOLO(YOLO_PESOS)
    if device.type == "cuda":
        modelo.to(device.index or 0)

    def inferir(img_path: Path):
        return modelo(img_path, verbose=False, conf=CONF_THRESHOLD, iou=0.7)

    # Aquecimento
    print(f"[YOLOv8] Aquecimento: {N_WARMUP} imagens...")
    for path in imagens[:N_WARMUP]:
        inferir(path)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("[YOLOv8] Aquecimento concluído.")

    # Inferência
    print(f"[YOLOv8] Inferência: {len(imagens)} imagens...")
    predicoes: list[dict] = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for path in imagens:
        resultados = inferir(path)
        image_id = id_map.get(path.name)
        if image_id is None:
            continue
        for r in resultados:
            for box in r.boxes:
                yolo_cls = int(box.cls)
                if yolo_cls not in YOLO_PARA_COCO:
                    continue
                score_val = float(box.conf)
                coco_cat  = YOLO_PARA_COCO[yolo_cls]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                predicoes.append({
                    "image_id":    image_id,
                    "category_id": coco_cat,
                    "bbox":        coco_bbox(x1, y1, x2, y2),
                    "score":       round(score_val, 6),
                })

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    fps = len(imagens) / (t1 - t0)
    print(f"[YOLOv8] FPS: {fps:.2f} | Predições: {len(predicoes)}")
    return {"modelo": "YOLOv8m", "fps": fps, "predicoes": predicoes}


# ---------------------------------------------------------------------------
# SSD
# ---------------------------------------------------------------------------

def benchmark_ssd(imagens: list[Path], id_map: dict, device: torch.device) -> dict:
    from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights

    print("\n[SSD] Carregando modelo...")
    pesos = SSD300_VGG16_Weights.COCO_V1
    modelo = ssd300_vgg16(weights=pesos).to(device)
    modelo.eval()
    transforms = pesos.transforms()
    print("[SSD] Modelo carregado.")

    def inferir(img_pil: Image.Image) -> dict:
        tensor = transforms(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            saida = modelo(tensor)
        return saida[0]

    # Aquecimento
    print(f"[SSD] Aquecimento: {N_WARMUP} imagens...")
    for path in imagens[:N_WARMUP]:
        inferir(Image.open(path).convert("RGB"))
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("[SSD] Aquecimento concluído.")

    # Inferência
    print(f"[SSD] Inferência: {len(imagens)} imagens...")
    predicoes: list[dict] = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for path in imagens:
        saida = inferir(Image.open(path).convert("RGB"))
        image_id = id_map.get(path.name)
        if image_id is None:
            continue
        for box, score, label in zip(
            saida["boxes"].cpu(),
            saida["scores"].cpu(),
            saida["labels"].cpu(),
        ):
            score_val = float(score)
            label_id  = int(label)
            if score_val < CONF_THRESHOLD:
                continue
            if label_id not in CLASSES_ALVO_COCO:
                continue
            x1, y1, x2, y2 = box.tolist()
            predicoes.append({
                "image_id":    image_id,
                "category_id": label_id,
                "bbox":        coco_bbox(x1, y1, x2, y2),
                "score":       round(score_val, 6),
            })

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    fps = len(imagens) / (t1 - t0)
    print(f"[SSD] FPS: {fps:.2f} | Predições: {len(predicoes)}")
    return {"modelo": "SSD300+VGG-16", "fps": fps, "predicoes": predicoes}


# ---------------------------------------------------------------------------
# Faster R-CNN
# ---------------------------------------------------------------------------

def benchmark_faster_rcnn(imagens: list[Path], id_map: dict, device: torch.device) -> dict:
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn_v2,
        FasterRCNN_ResNet50_FPN_V2_Weights,
    )

    print("\n[Faster R-CNN] Carregando modelo...")
    pesos = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    modelo = fasterrcnn_resnet50_fpn_v2(weights=pesos).to(device)
    modelo.eval()
    transforms = pesos.transforms()
    print("[Faster R-CNN] Modelo carregado.")

    def inferir(img_pil: Image.Image) -> dict:
        tensor = transforms(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            saida = modelo(tensor)
        return saida[0]

    # Aquecimento
    print(f"[Faster R-CNN] Aquecimento: {N_WARMUP} imagens...")
    for path in imagens[:N_WARMUP]:
        inferir(Image.open(path).convert("RGB"))
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("[Faster R-CNN] Aquecimento concluído.")

    # Inferência
    print(f"[Faster R-CNN] Inferência: {len(imagens)} imagens...")
    predicoes: list[dict] = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for path in imagens:
        saida = inferir(Image.open(path).convert("RGB"))
        image_id = id_map.get(path.name)
        if image_id is None:
            continue
        for box, score, label in zip(
            saida["boxes"].cpu(),
            saida["scores"].cpu(),
            saida["labels"].cpu(),
        ):
            score_val = float(score)
            label_id  = int(label)
            if score_val < CONF_THRESHOLD:
                continue
            if label_id not in CLASSES_ALVO_COCO:
                continue
            x1, y1, x2, y2 = box.tolist()
            predicoes.append({
                "image_id":    image_id,
                "category_id": label_id,
                "bbox":        coco_bbox(x1, y1, x2, y2),
                "score":       round(score_val, 6),
            })

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    fps = len(imagens) / (t1 - t0)
    print(f"[Faster R-CNN] FPS: {fps:.2f} | Predições: {len(predicoes)}")
    return {"modelo": "Faster R-CNN ResNet-50+FPN V2", "fps": fps, "predicoes": predicoes}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--completo", action="store_true",
                        help="Roda sobre o dataset completo (5000 imagens)")
    parser.add_argument("--modelo", choices=["yolo", "ssd", "faster", "todos"],
                        default="todos",
                        help="Modelo a executar (padrão: todos)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    limite = None if args.completo else N_PILOTO
    imagens = listar_imagens(limite)
    id_map  = carregar_id_map()
    cat_ids = list(CLASSES_ALVO_COCO.keys())
    sufixo  = "completo" if args.completo else f"piloto{len(imagens)}"

    print(f"Imagens: {len(imagens)} | Sufixo: {sufixo}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    resultados_csv: list[dict] = []

    rodar = {
        "yolo":   args.modelo in ("yolo",   "todos"),
        "ssd":    args.modelo in ("ssd",    "todos"),
        "faster": args.modelo in ("faster", "todos"),
    }

    # YOLOv8
    if rodar["yolo"]:
        r = benchmark_yolo(imagens, id_map, device)
        json_path = OUTPUT_DIR / f"yolo_predicoes_{sufixo}.json"
        salvar_predicoes(r["predicoes"], json_path)
        metricas = avaliar_map(r["predicoes"], ANNOT_FILE, cat_ids)
        resultados_csv.append({
            "modelo":    r["modelo"],
            "n_imagens": len(imagens),
            "fps":       round(r["fps"], 2),
            **metricas,
        })

    # SSD
    if rodar["ssd"]:
        r = benchmark_ssd(imagens, id_map, device)
        json_path = OUTPUT_DIR / f"ssd_predicoes_{sufixo}.json"
        salvar_predicoes(r["predicoes"], json_path)
        metricas = avaliar_map(r["predicoes"], ANNOT_FILE, cat_ids)
        resultados_csv.append({
            "modelo":    r["modelo"],
            "n_imagens": len(imagens),
            "fps":       round(r["fps"], 2),
            **metricas,
        })

    # Faster R-CNN
    if rodar["faster"]:
        r = benchmark_faster_rcnn(imagens, id_map, device)
        json_path = OUTPUT_DIR / f"faster_rcnn_predicoes_{sufixo}.json"
        salvar_predicoes(r["predicoes"], json_path)
        metricas = avaliar_map(r["predicoes"], ANNOT_FILE, cat_ids)
        resultados_csv.append({
            "modelo":    r["modelo"],
            "n_imagens": len(imagens),
            "fps":       round(r["fps"], 2),
            **metricas,
        })

    # Salva CSV consolidado
    if resultados_csv:
        csv_path = OUTPUT_DIR / f"resultados_{sufixo}.csv"
        campos = ["modelo", "n_imagens", "fps", "mAP_05", "precision", "recall"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            writer.writerows(resultados_csv)
        print(f"\n{'='*60}")
        print(f"CSV consolidado salvo: {csv_path}")
        print(f"{'='*60}")
        print(f"{'Modelo':<35} {'FPS':>6} {'mAP@0.5':>8} {'Precision':>10} {'Recall':>8}")
        print("-" * 70)
        for r in resultados_csv:
            print(
                f"{r['modelo']:<35} {r['fps']:>6.2f} "
                f"{r['mAP_05']:>8.4f} {r['precision']:>10.4f} {r['recall']:>8.4f}"
            )
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
