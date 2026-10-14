"""Extração e reconstrução da imagem (etapas 1 e 4 do projeto)."""
from pathlib import Path

import numpy as np
from PIL import Image


def carregar_imagem(caminho: str | Path, max_lado: int | None = None) -> np.ndarray:
    """Carrega a imagem em RGB (uint8, shape H x W x 3).

    max_lado: se informado, reduz a imagem mantendo a proporção (deixa o treino mais rápido).
    """
    img = Image.open(caminho).convert("RGB")
    if max_lado is not None and max(img.size) > max_lado:
        img.thumbnail((max_lado, max_lado), Image.LANCZOS)
    return np.asarray(img)


def extrair_pixels(img: np.ndarray) -> np.ndarray:
    """Transforma a imagem em um conjunto de dados 3D: N amostras x 3 (R, G, B) normalizadas em [0, 1]."""
    return img.reshape(-1, 3).astype(np.float64) / 255.0


def reconstruir_imagem(pixels: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Volta os pixels [0, 1] (N x 3) para a matriz da imagem uint8 (H x W x 3)."""
    return np.clip(np.rint(pixels * 255.0), 0, 255).astype(np.uint8).reshape(shape)


def salvar_imagem(img: np.ndarray, caminho: str | Path) -> None:
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(caminho)
