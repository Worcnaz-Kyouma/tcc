import csv
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

BASE_DIR   = Path.home() / "Documentos" / "Projects" / "TCC"
ANNOT_FILE = BASE_DIR / "coco" / "annotations" / "instances_val2017.json"
OUTPUT_DIR = BASE_DIR / "resultados"

CLASSES_ALVO_COCO = {1: "person", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}

ARQUIVOS = {
    "YOLOv8m":                      "yolo_predicoes_completo.json",
    "SSD300+VGG-16":                "ssd_predicoes_completo.json",
    "Faster R-CNN ResNet-50+FPN V2": "faster_rcnn_predicoes_completo.json",
}


def ponto_f1_maximo(avaliador, cat_id):
    cat_ids = list(avaliador.params.catIds)
    if cat_id not in cat_ids:
        return None
    k = cat_ids.index(cat_id)

    curva_p = avaliador.eval["precision"][0, :, k, 0, 2]
    curva_s = avaliador.eval["scores"][0, :, k, 0, 2]
    curva_r = np.array(avaliador.params.recThrs)

    validos = curva_p >= 0
    if not validos.any():
        return None

    curva_p, curva_s, curva_r = curva_p[validos], curva_s[validos], curva_r[validos]
    f1 = 2 * curva_p * curva_r / (curva_p + curva_r + 1e-9)
    i = int(f1.argmax())
    return float(curva_p[i]), float(curva_r[i]), float(f1[i]), float(curva_s[i])


def main():
    print(f"Carregando ground truth: {ANNOT_FILE}")
    coco_gt = COCO(str(ANNOT_FILE))
    cat_ids = sorted(CLASSES_ALVO_COCO)

    linhas = []

    for modelo, arquivo in ARQUIVOS.items():
        caminho = OUTPUT_DIR / arquivo
        if not caminho.exists():
            print(f"  AUSENTE: {caminho} — modelo ignorado.")
            continue

        print(f"\n=== {modelo} ===")
        with open(caminho) as f:
            predicoes = json.load(f)
        print(f"  {len(predicoes)} predições carregadas.")

        coco_dt = coco_gt.loadRes(predicoes)
        avaliador = COCOeval(coco_gt, coco_dt, "bbox")
        avaliador.params.catIds = cat_ids
        avaliador.params.imgIds = sorted({p["image_id"] for p in predicoes})
        avaliador.evaluate()
        avaliador.accumulate()

        print(f"  {'classe':<12} {'precision':>10} {'recall':>8} {'F1':>8} {'limiar':>9}")
        for cat_id in cat_ids:
            nome = CLASSES_ALVO_COCO[cat_id]
            r = ponto_f1_maximo(avaliador, cat_id)
            if r is None:
                print(f"  {nome:<12} {'sem dados':>10}")
                continue
            p, rec, f1, limiar = r
            print(f"  {nome:<12} {p:>10.4f} {rec:>8.4f} {f1:>8.4f} {limiar:>9.4f}")
            linhas.append({
                "modelo": modelo, "classe": nome,
                "precision": round(p, 4), "recall": round(rec, 4),
                "f1": round(f1, 4), "limiar_confianca": round(limiar, 4),
            })

        if linhas:
            do_modelo = [l for l in linhas if l["modelo"] == modelo]
            media_p = sum(l["precision"] for l in do_modelo) / len(do_modelo)
            media_r = sum(l["recall"] for l in do_modelo) / len(do_modelo)
            print(f"  {'MÉDIA':<12} {media_p:>10.4f} {media_r:>8.4f}"
                  f"   <- deve bater com a Tabela 3 do TCC")

    if not linhas:
        print("\nNenhum resultado produzido.")
        return

    destino = OUTPUT_DIR / "limiares_ponto_f1.csv"
    with open(destino, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["modelo", "classe", "precision",
                                          "recall", "f1", "limiar_confianca"])
        w.writeheader()
        w.writerows(linhas)
    print(f"\nCSV salvo: {destino}")


if __name__ == "__main__":
    main()
