"""Figuras: comparação original x quantizada, paletas, mapas do SOM, rede do GNG e curvas."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _salvar(fig, caminho):
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _paleta(ax, W, titulo="Paleta (pesos dos neurônios)"):
    """Desenha a paleta ordenada por luminância."""
    ordem = np.argsort(W @ np.array([0.299, 0.587, 0.114]))
    k = len(W)
    linhas = 1 if k <= 32 else int(np.ceil(k / 32))
    cols = min(k, 32)
    pal = np.ones((linhas * cols, 3))
    pal[:k] = W[ordem]
    ax.imshow(pal.reshape(linhas, cols, 3), aspect="auto", interpolation="nearest")
    ax.set_title(titulo, fontsize=8, loc="left")
    ax.axis("off")


def comparacao(original, quantizada, W, titulo, caminho):
    """Original x quantizada com a paleta extraída embaixo (igual ao exemplo do enunciado)."""
    fig = plt.figure(figsize=(8, 4.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[6, 1])
    for i, (img, t) in enumerate(((original, "Original"), (quantizada, titulo))):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    _paleta(fig.add_subplot(gs[1, :]), W)
    _salvar(fig, caminho)


def _paleta_grid(W):
    """Organiza os pesos num quadrado (16 -> 4x4, 64 -> 8x8, 256 -> 16x16) para exibir a paleta."""
    lado = int(np.ceil(np.sqrt(len(W))))
    pal = np.ones((lado * lado, 3))
    pal[: len(W)] = np.clip(W, 0, 1)
    return pal.reshape(lado, lado, 3)


def figura_grade(original, itens, titulo):
    """Original + imagem quantizada de cada tamanho de rede, com a paleta embaixo.

    itens = [(rótulo, img_quantizada, pesos_em_grid, psnr), ...]
    pesos_em_grid: SOM -> pesos no formato do grid (mostra a ordenação topológica);
                   GNG -> pesos na ordem de inserção.
    """
    n = len(itens) + 1
    fig, axs = plt.subplots(2, n, figsize=(3.2 * n, 6.4), gridspec_kw=dict(height_ratios=[1, 0.75]))
    n_orig = len(np.unique(original.reshape(-1, 3), axis=0))
    axs[0, 0].imshow(original)
    axs[0, 0].set_title(f"Original\n{n_orig} cores", fontsize=10)
    for k, (rot, img, pal, p) in enumerate(itens, start=1):
        n_cores = len(np.unique(img.reshape(-1, 3), axis=0))
        axs[0, k].imshow(img)
        axs[0, k].set_title(f"{rot}\nPSNR {p:.2f} dB", fontsize=10)
        axs[1, k].imshow(pal, interpolation="nearest")
        axs[1, k].set_title(f"Paleta {rot}\n{n_cores} cores", fontsize=9)
    for ax in axs.ravel():
        ax.axis("off")
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.35)
    return fig


def grade_quantizacoes(original, itens, titulo, caminho):
    _salvar(figura_grade(original, itens, titulo), caminho)


def mapa_som(som, caminho):
    """Mapa de cores do SOM (pesos no grid) e U-Matrix."""
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(np.clip(som.grid_cores(), 0, 1), interpolation="nearest")
    axs[0].set_title(f"Pesos no grid {som.linhas}x{som.colunas}")
    im = axs[1].imshow(som.umatrix(), cmap="viridis", interpolation="nearest")
    axs[1].set_title("U-Matrix")
    fig.colorbar(im, ax=axs[1], fraction=0.046)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    _salvar(fig, caminho)


def rede_gng(gng, X, caminho, n_amostras=3_000, seed=0):
    """Rede do GNG no espaço RGB: pixels (fundo), neurônios (pontos) e arestas."""
    rng = np.random.default_rng(seed)
    A = X[rng.choice(len(X), min(n_amostras, len(X)), replace=False)]
    fig = plt.figure(figsize=(6, 5.5))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*A.T, c=A, s=2, alpha=0.15)
    for i, j in gng.arestas():
        ax.plot(*np.stack([gng.W[i], gng.W[j]]).T, color="k", lw=0.5, alpha=0.6)
    ax.scatter(*gng.W.T, c=np.clip(gng.W, 0, 1), s=40, edgecolors="k", linewidths=0.6)
    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")
    ax.set_title(f"GNG com {len(gng.W)} neurônios no espaço RGB")
    _salvar(fig, caminho)


def mapa_som_rgb(som, X, caminho, n_amostras=3_000, seed=0):
    """Grid do SOM 'dobrado' no espaço RGB (mostra a preservação topológica)."""
    rng = np.random.default_rng(seed)
    A = X[rng.choice(len(X), min(n_amostras, len(X)), replace=False)]
    G = som.grid_cores()
    fig = plt.figure(figsize=(6, 5.5))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*A.T, c=A, s=2, alpha=0.15)
    for i in range(som.linhas):
        ax.plot(*G[i].T, color="k", lw=0.5)
    for j in range(som.colunas):
        ax.plot(*G[:, j].T, color="k", lw=0.5)
    ax.scatter(*som.W.T, c=np.clip(som.W, 0, 1), s=30, edgecolors="k", linewidths=0.5)
    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")
    ax.set_title(f"SOM {som.linhas}x{som.colunas} no espaço RGB")
    _salvar(fig, caminho)


def curvas_treino(curvas, titulo, caminho):
    """curvas = {rótulo: [(iteração, erro), ...]}"""
    fig, ax = plt.subplots(figsize=(7, 4))
    for rot, hist in curvas.items():
        h = np.array(hist)
        ax.plot(h[:, 0], h[:, -1], label=rot)
    ax.set_xlabel("Amostras apresentadas (iterações)")
    ax.set_ylabel("Erro de quantização")
    ax.set_title(titulo)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    _salvar(fig, caminho)


def barras_erros(resultados, caminho):
    """Erro de quantização e topológico da melhor configuração de cada rede."""
    rot = [r["rede"] for r in resultados]
    cores = ["#c0392b" if r["modelo"] == "SOM" else "#2471a3" for r in resultados]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, chave, t in ((axs[0], "erro_quantizacao", "Erro de quantização"),
                         (axs[1], "erro_topologico", "Erro topológico")):
        ax.bar(rot, [r[chave] for r in resultados], color=cores)
        ax.set_title(t)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
    _salvar(fig, caminho)
