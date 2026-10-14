"""Projeto 2 — Aprendizado Não Supervisionado: Quantização de Cores com SOM e GNG.

Pipeline:
  1. Extração     : carrega a imagem, extrai os pixels e normaliza RGB em [0, 1]
  2. Treinamento  : SOM (4x4, 8x8, 16x16) e GNG (16, 64, 256) com variações de hiperparâmetros
  3. Inferência   : para cada pixel, encontra o BMU e troca a cor pelos pesos do neurônio
  4. Reconstrução : remonta a matriz da imagem com as novas cores e salva/exibe
  Avaliação       : erro de quantização e erro topológico (+ PSNR como métrica extra)

Uso:
  uv run main.py                                   # roda as 4 imagens padrão (por_do_sol, campo, praia, tintas)
                                                   # ao final, exibe as imagens reconstruídas
  uv run main.py --imagem data/input/praia.jpg     # só uma imagem
  uv run main.py --modelo som                      # só SOM (ou gng)
  uv run main.py --som-impl propria                # SOM com a implementação própria em vez da XPySom
  uv run main.py --rapido                          # menos iterações (para testar)
  uv run main.py --max-lado 0                      # usa todos os pixels da imagem original
  uv run main.py --sem-mostrar                     # não abre as janelas (só salva os arquivos)
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from common import graficos
from common.imagem import carregar_imagem, extrair_pixels, reconstruir_imagem, salvar_imagem
from common.metricas import erro_quantizacao, erro_topologico, psnr, quantizar
from gng import GNG
from som import SOM, SOMXPySom

RAIZ = Path(__file__).parent

# ---------------------------------------------------------------------------------------------
# Experimentos: tamanhos exigidos no enunciado + ajustes de hiperparâmetros (taxas)
# ---------------------------------------------------------------------------------------------
SOM_GRIDS = [4, 8, 16]
SOM_CONFIGS = {
    # nome: função(lado do grid) -> hiperparâmetros
    "A (lr=0.5, σ0=L/2)": lambda L: dict(lr_inicial=0.5, sigma_inicial=L / 2),
    "B (lr=0.1, σ0=L/2)": lambda L: dict(lr_inicial=0.1, sigma_inicial=L / 2),
    "C (lr=0.5, σ0=L/4)": lambda L: dict(lr_inicial=0.5, sigma_inicial=max(1.0, L / 4)),
}
# XPySom treina em lote (batch SOM): a taxa de aprendizado se cancela na atualização
# (numerador / denominador), então o ajuste é feito na largura da vizinhança σ (início e fim).
SOM_CONFIGS_XPYSOM = {
    "A (σ0=L/2, σN=1.0)": lambda L: dict(sigma_inicial=L / 2, sigma_final=1.0),
    "B (σ0=L/2, σN=0.5)": lambda L: dict(sigma_inicial=L / 2, sigma_final=0.5),
    "C (σ0=L/4, σN=1.0)": lambda L: dict(sigma_inicial=max(1.0, L / 4), sigma_final=1.0),
}
SOM_ITER = {4: 10_000, 8: 20_000, 16: 40_000}   # implementação própria (online, 1 pixel por iteração)
SOM_EPOCAS = 100                                # XPySom: nº de passagens por todos os pixels

GNG_TAMANHOS = [16, 64, 256]
GNG_EPOCAS = 2  # cada época = todos os pixels da imagem (em ordem aleatória)
GNG_CONFIGS = {
    "A (εb=0.2, εn=0.006, λ=100)": dict(eps_b=0.2, eps_n=0.006, lam=100, idade_max=50),
    "B (εb=0.05, εn=0.0006, λ=100)": dict(eps_b=0.05, eps_n=0.0006, lam=100, idade_max=50),
    "C (εb=0.2, εn=0.006, λ=300)": dict(eps_b=0.2, eps_n=0.006, lam=300, idade_max=100),
}


def slug(txt: str) -> str:
    return txt.split(" ")[0]


def avaliar(modelo, X, img, nome_modelo, rede, config, pasta_saida, tempo):
    """Inferência + reconstrução + métricas de uma rede treinada."""
    novas_cores, bmu = quantizar(X, modelo.W)                      # 3. inferência
    img_q = reconstruir_imagem(novas_cores, img.shape)             # 4. reconstrução
    caminho = pasta_saida / nome_modelo.lower() / f"{rede}_{slug(config)}.png"
    salvar_imagem(img_q, caminho)
    return dict(
        modelo=nome_modelo, rede=rede, config=config,
        neuronios=len(modelo.W),
        cores_usadas=int(len(np.unique(bmu))),
        erro_quantizacao=erro_quantizacao(X, modelo.W),
        erro_topologico=erro_topologico(X, modelo.W, modelo.vizinhos),
        mse=float(np.mean((X - novas_cores) ** 2)),
        psnr=psnr(img, img_q),
        tempo_s=tempo,
        imagem=str(caminho.relative_to(RAIZ)),
    ), img_q


def rodar_som(X, img, pasta_saida, pasta_fig, rapido, impl="xpysom"):
    resultados, melhores, curvas_por_grid = [], [], {}
    for L in SOM_GRIDS:
        rede = f"{L}x{L}"
        candidatos = []
        configs = SOM_CONFIGS_XPYSOM if impl == "xpysom" else SOM_CONFIGS
        for nome_cfg, cfg in configs.items():
            t0 = time.perf_counter()
            if impl == "xpysom":
                som = SOMXPySom(L, L, n_epocas=5 if rapido else SOM_EPOCAS, **cfg(L)).treinar(X)
            else:
                n_iter = SOM_ITER[L] // (5 if rapido else 1)
                som = SOM(L, L, n_iter=n_iter, **cfg(L)).treinar(X)
            tempo = time.perf_counter() - t0
            res, img_q = avaliar(som, X, img, "SOM", rede, nome_cfg, pasta_saida, tempo)
            print(f"  SOM {rede:6s} {nome_cfg:24s} QE={res['erro_quantizacao']:.4f} "
                  f"TE={res['erro_topologico']:.4f} PSNR={res['psnr']:.2f}dB ({tempo:.1f}s)")
            resultados.append(res)
            candidatos.append((res, som, img_q))
            curvas_por_grid.setdefault(rede, {})[nome_cfg] = som.historico
        res, som, img_q = min(candidatos, key=lambda c: c[0]["erro_quantizacao"])
        melhores.append((res, som, img_q))
        graficos.comparacao(img, img_q, som.W, f"SOM {rede} — config {slug(res['config'])}",
                            pasta_fig / f"som_{rede}_comparacao.png")
        graficos.mapa_som(som, pasta_fig / f"som_{rede}_mapa.png")
        graficos.mapa_som_rgb(som, X, pasta_fig / f"som_{rede}_rgb.png")
        graficos.curvas_treino(curvas_por_grid[rede], f"SOM {rede} — erro de quantização no treino",
                               pasta_fig / f"som_{rede}_curvas.png")
    graficos.grade_quantizacoes(img, [(f"SOM {r['rede']}", q, np.clip(s.grid_cores(), 0, 1), r["psnr"])
                                      for r, s, q in melhores],
                                "SOM — melhor configuração por grid", pasta_fig / "som_grade.png")
    return resultados, melhores


def rodar_gng(X, img, pasta_saida, pasta_fig, rapido):
    resultados, melhores, curvas_por_rede = [], [], {}
    for N in GNG_TAMANHOS:
        rede = f"{N} neur."
        nome_arq = f"{N}n"
        candidatos = []
        for nome_cfg, cfg in GNG_CONFIGS.items():
            # crescimento até N neurônios (N·λ iterações) + ajuste fino; no mínimo GNG_EPOCAS
            # passagens pelos pixels, para comparar de forma justa com o SOM (100 épocas)
            n_iter = max(int(N * cfg["lam"] * 1.5), GNG_EPOCAS * len(X))
            if rapido:
                n_iter = int(N * cfg["lam"] * 1.5) // 5
            if rapido:  # no modo rápido, λ menor para a rede ainda crescer até N
                cfg = {**cfg, "lam": max(10, cfg["lam"] // 5)}
            t0 = time.perf_counter()
            gng = GNG(N, n_iter=n_iter, **cfg).treinar(X)
            tempo = time.perf_counter() - t0
            res, img_q = avaliar(gng, X, img, "GNG", nome_arq, nome_cfg, pasta_saida, tempo)
            res["rede"] = rede
            print(f"  GNG {rede:9s} {nome_cfg:30s} QE={res['erro_quantizacao']:.4f} "
                  f"TE={res['erro_topologico']:.4f} PSNR={res['psnr']:.2f}dB "
                  f"neur={res['neuronios']} ({tempo:.1f}s)")
            resultados.append(res)
            candidatos.append((res, gng, img_q))
            curvas_por_rede.setdefault(nome_arq, {})[nome_cfg] = gng.historico
        res, gng, img_q = min(candidatos, key=lambda c: c[0]["erro_quantizacao"])
        melhores.append((res, gng, img_q))
        graficos.comparacao(img, img_q, gng.W, f"GNG {rede} — config {slug(res['config'])}",
                            pasta_fig / f"gng_{nome_arq}_comparacao.png")
        graficos.rede_gng(gng, X, pasta_fig / f"gng_{nome_arq}_rgb.png")
        graficos.curvas_treino(curvas_por_rede[nome_arq], f"GNG {rede} — erro de quantização no treino",
                               pasta_fig / f"gng_{nome_arq}_curvas.png")
    graficos.grade_quantizacoes(img, [(f"GNG {r['rede']}", q, graficos._paleta_grid(g.W), r["psnr"])
                                      for r, g, q in melhores],
                                "GNG — melhor configuração por tamanho", pasta_fig / "gng_grade.png")
    return resultados, melhores


def tabela_latex(resultados, caminho):
    """Gera a tabela de resultados para o relatório (report/tabelas/)."""
    linhas = [r"\begin{tabular}{llccccccc}", r"\toprule",
              r"Modelo & Rede & Config. & Cores & EQ & ET & MSE & PSNR (dB) & Tempo (s)\\", r"\midrule"]
    ultimo = None
    for r in resultados:
        if ultimo and (r["modelo"], r["rede"]) != ultimo:
            linhas.append(r"\midrule")
        ultimo = (r["modelo"], r["rede"])
        linhas.append(f"{r['modelo']} & {r['rede']} & {slug(r['config'])} & {r['cores_usadas']} & "
                      f"{r['erro_quantizacao']:.4f} & {r['erro_topologico']:.4f} & {r['mse']:.2e} & {r['psnr']:.2f} & "
                      f"{r['tempo_s']:.1f}\\\\")
    linhas += [r"\bottomrule", r"\end{tabular}"]
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    Path(caminho).write_text("\n".join(linhas) + "\n", encoding="utf-8")


def tabela_comparativa(melhores, caminho):
    """Tabela lado a lado SOM x GNG (melhor configuração) com mesmo número de neurônios."""
    som = [r for r, _, _ in melhores if r["modelo"] == "SOM"]
    gng = [r for r, _, _ in melhores if r["modelo"] == "GNG"]
    if not som or not gng:
        return
    linhas = [r"\begin{tabular}{lcccccc}", r"\toprule",
              r"Neurônios & EQ SOM & EQ GNG & ET SOM & ET GNG & PSNR SOM & PSNR GNG\\", r"\midrule"]
    for a, b in zip(som, gng):
        linhas.append(f"{a['neuronios']} & {a['erro_quantizacao']:.4f} & {b['erro_quantizacao']:.4f} & "
                      f"{a['erro_topologico']:.4f} & {b['erro_topologico']:.4f} & "
                      f"{a['psnr']:.2f} & {b['psnr']:.2f}\\\\")
    linhas += [r"\bottomrule", r"\end{tabular}"]
    Path(caminho).write_text("\n".join(linhas) + "\n", encoding="utf-8")


IMAGENS_PADRAO = [
    "data/input/por_do_sol.jpg",
    "data/input/campo.jpg",
    "data/input/praia.jpg",
    "data/input/tintas.jpg",
]


def processar_imagem(caminho, args):
    """Roda o pipeline completo (extração → treino → inferência → reconstrução) em uma imagem."""
    # 1. Extração
    caminho_img = RAIZ / caminho
    img = carregar_imagem(caminho_img, args.max_lado or None)
    X = extrair_pixels(img)
    nome = caminho_img.stem
    pasta_saida = RAIZ / "data" / "output" / nome
    pasta_fig = RAIZ / "report" / "figures" / nome
    print(f"\n=== Imagem {caminho_img.name}: {img.shape[1]}x{img.shape[0]} = {len(X)} pixels, "
          f"{len(np.unique(img.reshape(-1, 3), axis=0))} cores distintas ===")

    # 2-4. Treinamento, inferência e reconstrução
    resultados, melhores = [], []
    if args.modelo in ("som", "todos"):
        print(f"Treinando SOM ({args.som_impl})...")
        r, m = rodar_som(X, img, pasta_saida, pasta_fig, args.rapido, args.som_impl)
        resultados += r
        melhores += m
    if args.modelo in ("gng", "todos"):
        print("Treinando GNG...")
        r, m = rodar_gng(X, img, pasta_saida, pasta_fig, args.rapido)
        resultados += r
        melhores += m

    # Avaliação: salva tudo
    pasta_saida.mkdir(parents=True, exist_ok=True)
    with open(pasta_saida / "resultados.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(resultados[0].keys()))
        w.writeheader()
        w.writerows(resultados)
    (pasta_saida / "resultados.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
    tabela_latex(resultados, RAIZ / "report" / "tabelas" / f"{nome}_resultados.tex")
    tabela_comparativa(melhores, RAIZ / "report" / "tabelas" / f"{nome}_comparacao.tex")
    graficos.barras_erros([r for r, _, _ in melhores], pasta_fig / "erros_melhores.png")
    print(f"Resultados em {pasta_saida.relative_to(RAIZ)} | figuras em {pasta_fig.relative_to(RAIZ)}")

    # Figuras das imagens reconstruídas, para exibir na tela no final (etapa 4 do enunciado)
    if not args.sem_mostrar:
        for modelo in ("SOM", "GNG"):
            itens = []
            for r, rede, q in melhores:
                if r["modelo"] != modelo:
                    continue
                pal = (np.clip(rede.grid_cores(), 0, 1) if modelo == "SOM"
                       else graficos._paleta_grid(rede.W))
                itens.append((f"{modelo} {r['rede']}", q, pal, r["psnr"]))
            if itens:
                graficos.figura_grade(img, itens, f"{modelo} — imagens reconstruídas ({nome})")


def main():
    p = argparse.ArgumentParser(description="Quantização de cores com SOM e GNG")
    p.add_argument("--imagem", nargs="+", default=IMAGENS_PADRAO,
                   help="uma ou mais imagens (padrão: as 4 imagens de data/input)")
    p.add_argument("--modelo", choices=["som", "gng", "todos"], default="todos")
    p.add_argument("--som-impl", choices=["xpysom", "propria"], default="xpysom",
                   help="treina o SOM com a biblioteca XPySom (padrão) ou com a implementação própria")
    p.add_argument("--max-lado", type=int, default=450,
                   help="reduz a imagem para no máx. N pixels de lado (0 = tamanho original)")
    p.add_argument("--rapido", action="store_true", help="menos iterações, para testar o pipeline")
    p.add_argument("--sem-mostrar", action="store_true",
                   help="não exibe as imagens reconstruídas na tela (só salva os arquivos)")
    args = p.parse_args()

    for caminho in args.imagem:
        processar_imagem(caminho, args)

    if not args.sem_mostrar:
        import matplotlib.pyplot as plt
        if plt.get_backend().lower() == "agg":
            print("(ambiente sem tela: as figuras foram salvas em report/figures/)")
        else:
            plt.show()  # abre todas as janelas de uma vez, no final


if __name__ == "__main__":
    main()
