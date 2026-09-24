import argparse
import csv
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image


BASE_DIR   = Path.home() / "Documentos" / "Projects" / "TCC"
COCO_DIR   = BASE_DIR / "coco" / "val2017"
ANNOT_FILE = BASE_DIR / "coco" / "annotations" / "instances_val2017.json"
OUTPUT_DIR = BASE_DIR / "resultados"

YOLO_PESOS = "yolov8m.pt"

CLASSES_ALVO_COCO = {
    1: "person",
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}

GRUPO_PESSOA  = [1]
GRUPO_VEICULO = [3, 4, 6, 8]

N_WARMUP = 20
N_REPETICOES_FPS = 5
N_PILOTO = 100
CONF_THRESHOLD = 0.001
N_AMOSTRA_DECODE = 500
LOG_A_CADA = 1000


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


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


def sincronizar(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


_COCO_GT = None


def obter_coco_gt():
    global _COCO_GT
    if _COCO_GT is None:
        from pycocotools.coco import COCO
        log("Carregando ground truth do COCO (uma unica vez)...")
        _COCO_GT = COCO(str(ANNOT_FILE))
    return _COCO_GT


def contar_anotacoes() -> None:
    coco = obter_coco_gt()
    log("Volume de anotacoes por classe-alvo no val2017:")
    total_anot = 0
    for cat_id, nome in CLASSES_ALVO_COCO.items():
        n_anot = len(coco.getAnnIds(catIds=[cat_id]))
        n_img  = len(coco.getImgIds(catIds=[cat_id]))
        total_anot += n_anot
        print(f"    {nome:<12} id={cat_id:<3} {n_anot:>6} anotacoes  {n_img:>5} imagens",
              flush=True)
    print(f"    {'TOTAL':<12} {'':<6} {total_anot:>6} anotacoes", flush=True)


def relatar_parametros_efetivos(nome: str, modelo) -> None:
    def buscar(*caminhos):
        for caminho in caminhos:
            alvo = modelo
            try:
                for parte in caminho.split("."):
                    alvo = getattr(alvo, parte)
                return alvo
            except AttributeError:
                continue
        return "n/d"

    campos = {
        "limiar de pontuacao":   buscar("score_thresh", "roi_heads.score_thresh"),
        "limiar de NMS":         buscar("nms_thresh", "roi_heads.nms_thresh"),
        "max. deteccoes/imagem": buscar("detections_per_img", "roi_heads.detections_per_img"),
        "resolucao (min_size)":  buscar("transform.min_size"),
        "resolucao (max_size)":  buscar("transform.max_size"),
        "resolucao (fixed)":     buscar("transform.fixed_size"),
    }
    try:
        n_param = sum(p.numel() for p in modelo.parameters())
        campos["parametros treinaveis"] = f"{n_param:,}".replace(",", ".")
    except Exception:
        pass

    log(f"[{nome}] Parametros de inferencia efetivos:")
    for k, v in campos.items():
        if v != "n/d":
            print(f"    {k:<24} {v}", flush=True)


def aquecer_cache_disco(imagens: list[Path]) -> None:
    log(f"Pre-carregando {len(imagens)} arquivos no cache do sistema...")
    t0 = time.perf_counter()
    total = 0
    for path in imagens:
        total += len(path.read_bytes())
    seg = time.perf_counter() - t0
    log(f"Cache aquecido: {total / 1024 / 1024:.0f} MB em {seg:.1f}s")


def medir_custo_decodificacao(imagens: list[Path]) -> float:
    amostra = imagens[:min(N_AMOSTRA_DECODE, len(imagens))]
    log(f"Estimando custo de decodificacao sobre {len(amostra)} imagens...")
    t0 = time.perf_counter()
    for path in amostra:
        Image.open(path).convert("RGB")
    ms = (time.perf_counter() - t0) / len(amostra) * 1000
    log(f"Custo medio de decodificacao: {ms:.2f} ms/imagem")
    return ms


def avaliar_map(predicoes: list[dict], cat_ids: list[int]) -> dict:
    from pycocotools.cocoeval import COCOeval

    coco_gt = obter_coco_gt()

    if not predicoes:
        log("  AVISO: nenhuma predicao gerada, metricas zeradas.")
        return {"mAP_05": 0.0, "precision": 0.0, "recall": 0.0, "por_classe": {}}

    coco_dt = coco_gt.loadRes(predicoes)
    avaliador = COCOeval(coco_gt, coco_dt, "bbox")
    avaliador.params.catIds = sorted(cat_ids)
    avaliador.params.imgIds = sorted({p["image_id"] for p in predicoes})

    avaliador.evaluate()
    avaliador.accumulate()
    avaliador.summarize()

    mAP = float(avaliador.stats[1])

    por_classe: dict[str, dict] = {}
    precisoes, recalls = [], []
    for cat_id in sorted(cat_ids):
        nome = CLASSES_ALVO_COCO[cat_id]
        ap   = _extrair_ap_classe(avaliador, cat_id)
        p, r = _extrair_pr_maximo_f1(avaliador, cat_id)
        por_classe[nome] = {"ap_05": round(ap, 4),
                            "precision": round(p, 4),
                            "recall": round(r, 4)}
        precisoes.append(p)
        recalls.append(r)

    return {
        "mAP_05":     round(mAP, 4),
        "precision":  round(sum(precisoes) / len(precisoes), 4),
        "recall":     round(sum(recalls)   / len(recalls),   4),
        "por_classe": por_classe,
    }


def _indice_classe(avaliador, cat_id: int) -> int | None:
    cat_ids_ordenados = list(avaliador.params.catIds)
    if cat_id not in cat_ids_ordenados:
        return None
    return cat_ids_ordenados.index(cat_id)


def _extrair_ap_classe(avaliador, cat_id: int) -> float:
    import numpy as np

    k_idx = _indice_classe(avaliador, cat_id)
    if k_idx is None:
        return 0.0

    curva_p = avaliador.eval["precision"][0, :, k_idx, 0, 2]
    validos = curva_p > -1
    if not validos.any():
        return 0.0
    return float(np.mean(curva_p[validos]))


def _extrair_pr_maximo_f1(avaliador, cat_id: int) -> tuple[float, float]:
    import numpy as np

    k_idx = _indice_classe(avaliador, cat_id)
    if k_idx is None:
        return 0.0, 0.0

    curva_p = avaliador.eval["precision"][0, :, k_idx, 0, 2]
    curva_r = np.array(avaliador.params.recThrs)

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
    log(f"  Predicoes salvas: {caminho} ({len(predicoes)} itens)")


def medir_fps(nome: str, inferir, imagens: list[Path], device: torch.device,
              repeticoes: int) -> dict:
    log(f"[{nome}] Aquecimento: {N_WARMUP} imagens...")
    for path in imagens[:N_WARMUP]:
        inferir(path)
    sincronizar(device)

    medicoes: list[float] = []
    for rep in range(1, repeticoes + 1):
        log(f"[{nome}] Passagem de FPS {rep}/{repeticoes} sobre {len(imagens)} imagens...")
        sincronizar(device)
        t0 = time.perf_counter()

        for i, path in enumerate(imagens, 1):
            inferir(path)
            if i % LOG_A_CADA == 0:
                parcial = i / (time.perf_counter() - t0)
                log(f"[{nome}]   {i}/{len(imagens)} ({parcial:.2f} FPS parcial)")

        sincronizar(device)
        fps = len(imagens) / (time.perf_counter() - t0)
        medicoes.append(fps)
        log(f"[{nome}] Repeticao {rep}: {fps:.2f} FPS")

    media = statistics.mean(medicoes)
    desvio = statistics.stdev(medicoes) if len(medicoes) > 1 else 0.0
    mediana = statistics.median(medicoes)
    log(f"[{nome}] FPS: {media:.2f} +/- {desvio:.2f} "
        f"(min {min(medicoes):.2f} | mediana {mediana:.2f} | max {max(medicoes):.2f}; "
        f"{repeticoes} execucoes)")
    return {"fps": media, "fps_desvio": desvio, "fps_mediana": mediana,
            "fps_min": min(medicoes), "fps_max": max(medicoes),
            "fps_medicoes": medicoes}


def coletar_predicoes(nome: str, extrair, imagens: list[Path]) -> list[dict]:
    log(f"[{nome}] Passagem de coleta de predicoes...")
    predicoes: list[dict] = []
    for i, path in enumerate(imagens, 1):
        predicoes.extend(extrair(path))
        if i % LOG_A_CADA == 0:
            log(f"[{nome}]   {i}/{len(imagens)} ({len(predicoes)} predicoes)")
    log(f"[{nome}] Predicoes coletadas: {len(predicoes)}")
    return predicoes


YOLO_PARA_COCO = {0: 1, 2: 3, 3: 4, 5: 6, 7: 8}


def benchmark_yolo(imagens: list[Path], id_map: dict, device: torch.device,
                   repeticoes: int) -> dict:
    from ultralytics import YOLO

    log("[YOLOv8] Carregando modelo...")
    modelo = YOLO(YOLO_PESOS)
    if device.type == "cuda":
        modelo.to(device.index or 0)

    def inferir(img_path: Path):
        img = Image.open(img_path).convert("RGB")
        return modelo(img, verbose=False, conf=CONF_THRESHOLD, iou=0.7)

    def extrair(img_path: Path) -> list[dict]:
        image_id = id_map.get(img_path.name)
        if image_id is None:
            return []
        saida = []
        for r in inferir(img_path):
            for box in r.boxes:
                yolo_cls = int(box.cls)
                if yolo_cls not in YOLO_PARA_COCO:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                saida.append({
                    "image_id":    image_id,
                    "category_id": YOLO_PARA_COCO[yolo_cls],
                    "bbox":        coco_bbox(x1, y1, x2, y2),
                    "score":       round(float(box.conf), 6),
                })
        return saida

    tempo = medir_fps("YOLOv8", inferir, imagens, device, repeticoes)
    predicoes = coletar_predicoes("YOLOv8", extrair, imagens)
    return {"modelo": "YOLOv8m", "predicoes": predicoes, **tempo}


def _benchmark_torchvision(
    nome: str,
    rotulo: str,
    construtor,
    pesos,
    kwargs_limiar: dict,
    imagens: list[Path],
    id_map: dict,
    device: torch.device,
    repeticoes: int,
) -> dict:
    log(f"[{nome}] Carregando modelo (limiar interno: {kwargs_limiar})...")
    modelo = construtor(weights=pesos, **kwargs_limiar).to(device)
    modelo.eval()
    transforms = pesos.transforms()
    log(f"[{nome}] Modelo carregado.")
    relatar_parametros_efetivos(nome, modelo)

    def inferir(img_path: Path):
        img = Image.open(img_path).convert("RGB")
        tensor = transforms(img).unsqueeze(0).to(device)
        with torch.no_grad():
            return modelo(tensor)[0]

    def extrair(img_path: Path) -> list[dict]:
        image_id = id_map.get(img_path.name)
        if image_id is None:
            return []
        s = inferir(img_path)
        saida = []
        for box, score, label in zip(s["boxes"].cpu(), s["scores"].cpu(), s["labels"].cpu()):
            score_val = float(score)
            label_id  = int(label)
            if score_val < CONF_THRESHOLD:
                continue
            if label_id not in CLASSES_ALVO_COCO:
                continue
            x1, y1, x2, y2 = box.tolist()
            saida.append({
                "image_id":    image_id,
                "category_id": label_id,
                "bbox":        coco_bbox(x1, y1, x2, y2),
                "score":       round(score_val, 6),
            })
        return saida

    tempo = medir_fps(nome, inferir, imagens, device, repeticoes)
    predicoes = coletar_predicoes(nome, extrair, imagens)
    return {"modelo": rotulo, "predicoes": predicoes, **tempo}


def benchmark_ssd(imagens: list[Path], id_map: dict, device: torch.device,
                  repeticoes: int) -> dict:
    from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
    return _benchmark_torchvision(
        "SSD", "SSD300+VGG-16",
        ssd300_vgg16, SSD300_VGG16_Weights.COCO_V1,
        {"score_thresh": CONF_THRESHOLD},
        imagens, id_map, device, repeticoes,
    )


def benchmark_faster_rcnn(imagens: list[Path], id_map: dict, device: torch.device,
                          repeticoes: int) -> dict:
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn_v2,
        FasterRCNN_ResNet50_FPN_V2_Weights,
    )
    return _benchmark_torchvision(
        "Faster R-CNN", "Faster R-CNN ResNet-50+FPN V2",
        fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
        {"box_score_thresh": CONF_THRESHOLD},
        imagens, id_map, device, repeticoes,
    )


def media_grupo(por_classe: dict, grupo: list[int], chave: str) -> float:
    nomes = [CLASSES_ALVO_COCO[c] for c in grupo if CLASSES_ALVO_COCO[c] in por_classe]
    if not nomes:
        return 0.0
    return round(sum(por_classe[n][chave] for n in nomes) / len(nomes), 4)


def fps_inferencia_estimado(fps: float, ms_decode: float) -> float:
    ms_total = 1000.0 / fps
    ms_puro  = ms_total - ms_decode
    return 1000.0 / ms_puro if ms_puro > 0 else 0.0


def imprimir_relatorio(resultados: list[dict], ms_decode: float) -> None:
    print("\n" + "=" * 84, flush=True)
    print("RESULTADO CONSOLIDADO", flush=True)
    print("=" * 84, flush=True)
    print(f"{'Modelo':<32} {'FPS':>15} {'mAP@0.5':>9} {'Precision':>10} {'Recall':>8}",
          flush=True)
    print("-" * 84, flush=True)
    for r in resultados:
        fps_txt = f"{r['fps']:.2f} +/- {r['fps_desvio']:.2f}"
        print(f"{r['modelo']:<32} {fps_txt:>15} {r['mAP_05']:>9.4f} "
              f"{r['precision']:>10.4f} {r['recall']:>8.4f}", flush=True)

    print("\n" + "-" * 84, flush=True)
    print(f"FPS FIM A FIM x FPS DE INFERENCIA ESTIMADO "
          f"(decodificacao: {ms_decode:.2f} ms/imagem)", flush=True)
    print("-" * 84, flush=True)
    print(f"{'Modelo':<32} {'fim a fim':>12} {'inferencia (est.)':>20}", flush=True)
    print("-" * 84, flush=True)
    for r in resultados:
        est = fps_inferencia_estimado(r["fps"], ms_decode)
        print(f"{r['modelo']:<32} {r['fps']:>12.2f} {est:>20.2f}", flush=True)

    print("\n" + "-" * 84, flush=True)
    print("AP@0.5 POR CLASSE", flush=True)
    print("-" * 84, flush=True)
    nomes = list(CLASSES_ALVO_COCO.values())
    print(f"{'Modelo':<32}" + "".join(f"{n:>11}" for n in nomes), flush=True)
    print("-" * 84, flush=True)
    for r in resultados:
        linha = f"{r['modelo']:<32}"
        for n in nomes:
            linha += f"{r['por_classe'].get(n, {}).get('ap_05', 0.0):>11.4f}"
        print(linha, flush=True)

    print("\n" + "-" * 84, flush=True)
    print("AP@0.5 AGREGADO POR GRUPO SEMANTICO", flush=True)
    print("-" * 84, flush=True)
    print(f"{'Modelo':<32} {'pessoa':>11} {'veiculo':>11}", flush=True)
    print("-" * 84, flush=True)
    for r in resultados:
        p = media_grupo(r["por_classe"], GRUPO_PESSOA, "ap_05")
        v = media_grupo(r["por_classe"], GRUPO_VEICULO, "ap_05")
        print(f"{r['modelo']:<32} {p:>11.4f} {v:>11.4f}", flush=True)
    print("=" * 84, flush=True)


def salvar_csvs(resultados: list[dict], sufixo: str, ms_decode: float) -> None:
    csv_path = OUTPUT_DIR / f"resultados_{sufixo}.csv"
    campos = ["modelo", "n_imagens", "fps", "fps_desvio", "fps_mediana",
              "fps_min", "fps_max", "fps_execucoes",
              "ms_decode", "fps_inferencia_est",
              "mAP_05", "precision", "recall", "ap_pessoa", "ap_veiculo"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for r in resultados:
            writer.writerow({
                "modelo":             r["modelo"],
                "n_imagens":          r["n_imagens"],
                "fps":                round(r["fps"], 2),
                "fps_desvio":         round(r["fps_desvio"], 2),
                "fps_mediana":        round(r["fps_mediana"], 2),
                "fps_min":            round(r["fps_min"], 2),
                "fps_max":            round(r["fps_max"], 2),
                "fps_execucoes":      "; ".join(f"{v:.2f}" for v in r["fps_medicoes"]),
                "ms_decode":          round(ms_decode, 2),
                "fps_inferencia_est": round(fps_inferencia_estimado(r["fps"], ms_decode), 2),
                "mAP_05":             r["mAP_05"],
                "precision":          r["precision"],
                "recall":             r["recall"],
                "ap_pessoa":          media_grupo(r["por_classe"], GRUPO_PESSOA, "ap_05"),
                "ap_veiculo":         media_grupo(r["por_classe"], GRUPO_VEICULO, "ap_05"),
            })
    log(f"CSV consolidado salvo: {csv_path}")

    csv_classe = OUTPUT_DIR / f"resultados_por_classe_{sufixo}.csv"
    with open(csv_classe, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["modelo", "classe", "ap_05",
                                               "precision", "recall"])
        writer.writeheader()
        for r in resultados:
            for classe, m in r["por_classe"].items():
                writer.writerow({"modelo": r["modelo"], "classe": classe, **m})
    log(f"CSV por classe salvo: {csv_classe}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--completo", action="store_true",
                        help="Roda sobre o dataset completo (5000 imagens)")
    parser.add_argument("--modelo", choices=["yolo", "ssd", "faster", "todos"],
                        default="todos",
                        help="Modelo a executar (padrao: todos)")
    parser.add_argument("--repeticoes", type=int, default=N_REPETICOES_FPS,
                        help=f"Repeticoes da passagem de FPS (padrao: {N_REPETICOES_FPS})")
    args = parser.parse_args()

    t_inicio_geral = time.perf_counter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Dispositivo: {device}")
    if device.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    limite  = None if args.completo else N_PILOTO
    imagens = listar_imagens(limite)
    id_map  = carregar_id_map()
    cat_ids = sorted(CLASSES_ALVO_COCO.keys())
    sufixo  = "completo" if args.completo else f"piloto{len(imagens)}"

    log(f"Imagens: {len(imagens)} | Sufixo: {sufixo}")
    log(f"Classes-alvo: {', '.join(CLASSES_ALVO_COCO.values())}")
    log(f"Limiar de pontuacao unificado: {CONF_THRESHOLD}")
    log(f"Repeticoes de FPS por modelo: {args.repeticoes}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    contar_anotacoes()
    aquecer_cache_disco(imagens)
    ms_decode = medir_custo_decodificacao(imagens)

    funcoes = {
        "yolo":   ("yolo",        benchmark_yolo),
        "ssd":    ("ssd",         benchmark_ssd),
        "faster": ("faster_rcnn", benchmark_faster_rcnn),
    }

    resultados: list[dict] = []

    for chave, (prefixo_arquivo, funcao) in funcoes.items():
        if args.modelo not in (chave, "todos"):
            continue
        print("\n" + "=" * 84, flush=True)
        r = funcao(imagens, id_map, device, args.repeticoes)
        salvar_predicoes(r["predicoes"],
                         OUTPUT_DIR / f"{prefixo_arquivo}_predicoes_{sufixo}.json")
        metricas = avaliar_map(r["predicoes"], cat_ids)
        resultados.append({
            "modelo":       r["modelo"],
            "n_imagens":    len(imagens),
            "fps":          r["fps"],
            "fps_desvio":   r["fps_desvio"],
            "fps_mediana":  r["fps_mediana"],
            "fps_min":      r["fps_min"],
            "fps_max":      r["fps_max"],
            "fps_medicoes": r["fps_medicoes"],
            **metricas,
        })

    if resultados:
        imprimir_relatorio(resultados, ms_decode)
        salvar_csvs(resultados, sufixo, ms_decode)

    total_min = (time.perf_counter() - t_inicio_geral) / 60
    log(f"Execucao concluida em {total_min:.1f} minutos.")


if __name__ == "__main__":
    main()
