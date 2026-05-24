"""
Nicolas Almeida Prado — UEPG — 2026
"""

import os
import time
from pathlib import Path

from ultralytics import YOLO

# Diretório do dataset COCO val2017
COCO_DIR = Path.home() / "Documentos" / "Projects" / "TCC" / "coco" / "val2017"

# Modelo YOLOv8 a ser usado.
MODELO = "yolov8m.pt"

N_IMAGENS = 10

# Classes alvo do experimento.
CLASSES_ALVO_YOLO = {0: "person", 2: "car"}

# Retorna os primeiros `limite` arquivos .jpg do diretório, ordenados.
def listar_imagens(diretorio: Path, limite: int) -> list[Path]:
    if not diretorio.exists():
        raise FileNotFoundError(f"Diretório do dataset não encontrado: {diretorio}")

    imagens = sorted(diretorio.glob("*.jpg"))
    if not imagens:
        raise RuntimeError(f"Nenhuma imagem .jpg encontrada em {diretorio}")

    return imagens[:limite]

# Carrega o modelo YOLOv8 com pesos pré-treinados no COCO.
def carregar_yolov8(caminho_pesos: str) -> YOLO:
    print(f"Carregando modelo: {caminho_pesos}")
    modelo = YOLO(caminho_pesos)
    print(f"Modelo carregado. Classes conhecidas: {len(modelo.names)}")
    return modelo


def rodar_inferencia(modelo: YOLO, imagens: list[Path]) -> dict:
    sumario = {
        "n_imagens": len(imagens),
        "total_deteccoes": 0,
        "por_classe_alvo": {nome: 0 for nome in CLASSES_ALVO_YOLO.values()},
        "tempo_total_s": 0.0,
    }

    print(f"\nProcessando {len(imagens)} imagens...\n")

    inicio = time.perf_counter()

    resultados = modelo(imagens, verbose=False)

    fim = time.perf_counter()
    sumario["tempo_total_s"] = fim - inicio

    for resultado in resultados:
        n_box = len(resultado.boxes)
        sumario["total_deteccoes"] += n_box

        for box in resultado.boxes:
            cls_id = int(box.cls)
            if cls_id in CLASSES_ALVO_YOLO:
                nome = CLASSES_ALVO_YOLO[cls_id]
                sumario["por_classe_alvo"][nome] += 1

        alvos_na_imagem = [
            CLASSES_ALVO_YOLO[int(b.cls)]
            for b in resultado.boxes
            if int(b.cls) in CLASSES_ALVO_YOLO
        ]
        nome_arq = Path(resultado.path).name
        print(
            f"  {nome_arq}: {n_box} deteções totais, "
            f"alvos = {alvos_na_imagem if alvos_na_imagem else '(nenhum)'}"
        )

    return sumario


def imprimir_sumario(sumario: dict) -> None:
    print("\n" + "=" * 60)
    print("SUMÁRIO DA EXECUÇÃO")
    print("=" * 60)
    print(f"Imagens processadas:       {sumario['n_imagens']}")
    print(f"Detecções totais:          {sumario['total_deteccoes']}")
    print(f"Detecções por classe-alvo do TCC:")
    for nome, n in sumario["por_classe_alvo"].items():
        print(f"  - {nome:8s}: {n}")
    print(f"Tempo total de inferência: {sumario['tempo_total_s']:.2f} s")
    print(
        f"Tempo médio por imagem:    "
        f"{sumario['tempo_total_s'] / sumario['n_imagens'] * 1000:.1f} ms"
    )
    print("=" * 60)

def main():
    imagens = listar_imagens(COCO_DIR, N_IMAGENS)
    modelo = carregar_yolov8(MODELO)
    sumario = rodar_inferencia(modelo, imagens)
    imprimir_sumario(sumario)

if __name__ == "__main__":
    main()