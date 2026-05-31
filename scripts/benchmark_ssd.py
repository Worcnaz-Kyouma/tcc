"""
benchmark_ssd.py
Nicolas Almeida Prado, UEPG:2026

Tips:
    python benchmark_ssd.py
    python benchmark_ssd.py --completo
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

BASE_DIR    = Path.home() / "Documentos" / "Projects" / "TCC"
COCO_DIR    = BASE_DIR / "coco" / "val2017"
ANNOT_FILE  = BASE_DIR / "coco" / "annotations" / "instances_val2017.json"
OUTPUT_DIR  = BASE_DIR / "resultados"

# Classes-alvo: mapeamento ID COCO oficial → nome
CLASSES_ALVO_COCO = {1: "person", 3: "car"}

# Aquecimento da GPU antes de medir FPS (imagens, não contam no tempo)
N_WARMUP = 20

# Subconjunto piloto para testes rápidos
N_PILOTO = 100

# Limiar de confiança para salvar predições no JSON
# Baixo para preservar a curva Precision-Recall completa (mesmo critério do YOLOv8)
CONF_THRESHOLD = 0.001

# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def listar_imagens(limite: int | None) -> list[Path]:
    imagens = sorted(COCO_DIR.glob("*.jpg"))
    if not imagens:
        raise RuntimeError(f"Nenhuma imagem encontrada em {COCO_DIR}")
    return imagens if limite is None else imagens[:limite]


def carregar_id_map(annot_file: Path) -> dict[str, int]:
    with open(annot_file) as f:
        dados = json.load(f)
    return {img["file_name"]: img["id"] for img in dados["images"]}


def coco_bbox(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    return [x1, y1, x2 - x1, y2 - y1]


# ---------------------------------------------------------------------------
# Inferência
# ---------------------------------------------------------------------------

def carregar_modelo(device: torch.device):
    print("Carregando SSD300+VGG-16 com pesos COCO_V1...")
    pesos = SSD300_VGG16_Weights.COCO_V1
    modelo = ssd300_vgg16(weights=pesos)
    modelo.to(device)
    modelo.eval()
    transforms = pesos.transforms()
    print(f"Modelo carregado. Classes conhecidas: {len(pesos.meta['categories'])}")
    return modelo, transforms


def inferir_imagem(modelo, transforms, imagem_pil: Image.Image, device: torch.device) -> dict:
    tensor = transforms(imagem_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        saida = modelo(tensor)
    return {
        "boxes":  saida[0]["boxes"].cpu(),
        "scores": saida[0]["scores"].cpu(),
        "labels": saida[0]["labels"].cpu(),
    }


def rodar_benchmark(
    modelo,
    transforms,
    imagens: list[Path],
    id_map: dict[str, int],
    device: torch.device,
) -> tuple[list[dict], float]:
    predicoes: list[dict] = []

    # Aquecimento
    print(f"Aquecimento: {N_WARMUP} imagens...")
    for path in imagens[:N_WARMUP]:
        img = Image.open(path).convert("RGB")
        inferir_imagem(modelo, transforms, img, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("Aquecimento concluído.\n")

    # Inferência
    print(f"Inferência: {len(imagens)} imagens...")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_inicio = time.perf_counter()

    for path in imagens:
        img = Image.open(path).convert("RGB")
        saida = inferir_imagem(modelo, transforms, img, device)

        nome_arquivo = path.name
        image_id = id_map.get(nome_arquivo)
        if image_id is None:
            continue

        for box, score, label in zip(saida["boxes"], saida["scores"], saida["labels"]):
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
    t_fim = time.perf_counter()

    tempo_total = t_fim - t_inicio
    fps = len(imagens) / tempo_total

    print(f"Concluído. Tempo total: {tempo_total:.1f}s | FPS: {fps:.2f}")
    print(f"Predições geradas (person+car, conf>{CONF_THRESHOLD}): {len(predicoes)}")

    return predicoes, fps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--completo", action="store_true",
                        help="Roda sobre o dataset completo (5000 imagens)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    limite = None if args.completo else N_PILOTO
    imagens = listar_imagens(limite)
    print(f"Imagens selecionadas: {len(imagens)}")

    id_map = carregar_id_map(ANNOT_FILE)
    modelo, transforms = carregar_modelo(device)
    predicoes, fps = rodar_benchmark(modelo, transforms, imagens, id_map, device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sufixo = "completo" if args.completo else f"piloto{len(imagens)}"
    json_path = OUTPUT_DIR / f"ssd_predicoes_{sufixo}.json"
    with open(json_path, "w") as f:
        json.dump(predicoes, f)
    print(f"\nPredições salvas em: {json_path}")

    fps_path = OUTPUT_DIR / f"ssd_fps_{sufixo}.txt"
    fps_path.write_text(f"{fps:.4f}\n")
    print(f"FPS salvo em: {fps_path}")


if __name__ == "__main__":
    main()
