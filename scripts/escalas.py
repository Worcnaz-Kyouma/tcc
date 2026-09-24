import csv, io, json, contextlib
from collections import Counter
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

BASE_DIR   = Path.home() / "Documentos" / "Projects" / "TCC"
ANNOT_FILE = BASE_DIR / "coco" / "annotations" / "instances_val2017.json"
OUTPUT_DIR = BASE_DIR / "resultados"
PREDICOES = {
    "YOLOv8m":      OUTPUT_DIR / "yolo_predicoes_completo.json",
    "SSD300":       OUTPUT_DIR / "ssd_predicoes_completo.json",
    "FasterRCNN":   OUTPUT_DIR / "faster_rcnn_predicoes_completo.json",
}
TETO_MODELO = {"YOLOv8m": 300, "SSD300": 200, "FasterRCNN": 100}

CATS   = {1: "person", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}
FAIXAS = ["all", "small", "medium", "large"]
M_100  = 2


def silencioso(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def avaliar(gt, dt, img_ids):
    E = COCOeval(gt, dt, "bbox")
    E.params.catIds = sorted(CATS)
    E.params.imgIds = img_ids
    silencioso(E.evaluate); silencioso(E.accumulate)
    return E


def ap50(E, k, a):
    p = E.eval["precision"][0, :, k, a, M_100]
    p = p[p > -1]
    return float(np.mean(p)) if p.size else float("nan")


def main():
    gt = silencioso(COCO, str(ANNOT_FILE))
    todas = sorted(gt.getImgIds())
    cat_ids = sorted(CATS)

    lims = [(0, 32**2), (32**2, 96**2), (96**2, 1e10)]
    with open(OUTPUT_DIR / "ap_por_escala_suporte.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["categoria", "n_total", "n_small", "n_medium", "n_large",
                    "pct_small", "pct_medium", "pct_large"])
        for c in cat_ids:
            anns = gt.loadAnns(gt.getAnnIds(catIds=[c], iscrowd=False))
            n = [sum(lo <= a["area"] < hi for a in anns) for lo, hi in lims]
            pct = [f"{100*x/len(anns):.1f}" for x in n]
            w.writerow([CATS[c], len(anns), *n, *pct])
            print("suporte", CATS[c], len(anns), n, pct)

    imgs_com_alvo = {a["image_id"] for a in
                     gt.loadAnns(gt.getAnnIds(catIds=cat_ids, iscrowd=False))}

    f_ap  = open(OUTPUT_DIR / "ap_por_escala.csv", "w", newline="")
    f_dg  = open(OUTPUT_DIR / "diagnostico_protocolo.csv", "w", newline="")
    w_ap, w_dg = csv.writer(f_ap), csv.writer(f_dg)
    w_ap.writerow(["modelo", "categoria"] + [f"AP50_{n}" for n in FAIXAS])
    w_dg.writerow(["modelo", "indicador", "valor"])

    for nome, caminho in PREDICOES.items():
        with open(caminho) as fh:
            preds = json.load(fh)
        dt = silencioso(gt.loadRes, str(caminho))
        img_bench = sorted({p["image_id"] for p in preds})

        E = avaliar(gt, dt, img_bench)
        linhas = []
        for k, c in enumerate(E.params.catIds):
            r = [ap50(E, k, a) for a in range(4)]
            linhas.append(r)
            w_ap.writerow([nome, CATS[c]] + [f"{x:.4f}" for x in r])
            print(nome, CATS[c], [round(x, 4) for x in r])
        media = np.nanmean(np.array(linhas), axis=0)
        w_ap.writerow([nome, "MEDIA"] + [f"{x:.4f}" for x in media])
        print(nome, "MEDIA", [round(float(x), 4) for x in media])

        sem_pred = imgs_com_alvo - set(img_bench)
        anns_perdidas = len(gt.getAnnIds(imgIds=list(sem_pred), catIds=cat_ids,
                                         iscrowd=False)) if sem_pred else 0
        E_all = avaliar(gt, dt, todas)
        map_bench = float(media[0])
        map_all = float(np.nanmean([ap50(E_all, k, 0) for k in range(len(cat_ids))]))
        diag = {
            "imagens_avaliadas_benchmark": len(img_bench),
            "imagens_com_alvo_sem_predicao": len(sem_pred),
            "anotacoes_alvo_ignoradas": anns_perdidas,
            "mAP50_imgIds_benchmark": f"{map_bench:.4f}",
            "mAP50_imgIds_5000": f"{map_all:.4f}",
        }

        por_img_cat = Counter((p["image_id"], p["category_id"]) for p in preds)
        por_img = Counter(p["image_id"] for p in preds)
        v = np.array(list(por_img.values()))
        diag.update({
            "pares_img_cat_acima_de_100": sum(n > 100 for n in por_img_cat.values()),
            "deteccoes_alvo_por_imagem_media": f"{v.mean():.1f}",
            "deteccoes_alvo_por_imagem_p99": int(np.percentile(v, 99)),
            "deteccoes_alvo_por_imagem_max": int(v.max()),
            "teto_do_modelo": TETO_MODELO[nome],
            "imagens_no_teto_so_com_alvo": int((v >= TETO_MODELO[nome]).sum()),
            "imagens_com_90pct_do_teto": int((v >= 0.9 * TETO_MODELO[nome]).sum()),
        })
        for k2, v2 in diag.items():
            w_dg.writerow([nome, k2, v2])
            print(nome, k2, v2)

    f_ap.close(); f_dg.close()
    print("\nConcluído. Confira AP50_all contra a Tabela 4 antes de usar.")


if __name__ == "__main__":
    main()
