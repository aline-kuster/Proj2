"""Busca do neurônio vencedor (BMU), quantização e métricas de avaliação."""
import numpy as np


def dist2(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Distância euclidiana ao quadrado entre cada amostra de X (N x 3) e cada peso de W (K x 3)."""
    return (X**2).sum(1)[:, None] - 2.0 * X @ W.T + (W**2).sum(1)[None, :]


def dois_melhores(X: np.ndarray, W: np.ndarray, bloco: int = 50_000):
    """Retorna (bmu, segundo_bmu, distancia_bmu) para cada amostra, processando em blocos."""
    n = len(X)
    b1 = np.empty(n, dtype=np.int64)
    b2 = np.empty(n, dtype=np.int64)
    d1 = np.empty(n)
    for i in range(0, n, bloco):
        d = np.maximum(dist2(X[i : i + bloco], W), 0.0)
        idx = np.argpartition(d, 1, axis=1)[:, :2]
        dd = np.take_along_axis(d, idx, axis=1)
        ordem = np.argsort(dd, axis=1)
        idx = np.take_along_axis(idx, ordem, axis=1)
        b1[i : i + bloco], b2[i : i + bloco] = idx[:, 0], idx[:, 1]
        d1[i : i + bloco] = np.sqrt(np.take_along_axis(dd, ordem, axis=1)[:, 0])
    return b1, b2, d1


def quantizar(X: np.ndarray, W: np.ndarray):
    """Inferência: troca a cor de cada pixel pela cor (pesos) do seu neurônio vencedor."""
    bmu, _, _ = dois_melhores(X, W)
    return W[bmu], bmu


def erro_quantizacao(X: np.ndarray, W: np.ndarray) -> float:
    """Erro de quantização: distância média entre cada pixel e o peso do seu BMU."""
    _, _, d1 = dois_melhores(X, W)
    return float(d1.mean())


def erro_topologico(X: np.ndarray, W: np.ndarray, vizinhos) -> float:
    """Erro topológico: fração de pixels cujo 1º e 2º BMU NÃO são vizinhos na rede.

    vizinhos(b1, b2) -> array booleano indicando se os pares são vizinhos
    (no SOM: adjacentes no grid; no GNG: ligados por uma aresta).
    """
    b1, b2, _ = dois_melhores(X, W)
    return float(1.0 - vizinhos(b1, b2).mean())


def psnr(original: np.ndarray, reconstruida: np.ndarray) -> float:
    """PSNR (dB) entre a imagem original e a quantizada (uint8). Métrica extra de qualidade."""
    mse = np.mean((original.astype(np.float64) - reconstruida.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))
