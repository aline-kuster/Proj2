"""Projeto 2 — Aprendizado Não Supervisionado: Quantização de Cores com SOM e GNG.

Ideia geral: cada pixel da imagem é um ponto (R, G, B). A rede aprende poucas cores
"representantes" (os pesos dos neurônios) e depois cada pixel é trocado pela cor do
neurônio mais parecido com ele. Resultado: a mesma imagem com uma paleta reduzida.

Etapas (as mesmas do enunciado):
  1. Extração     : carrega a imagem, extrai os pixels e normaliza RGB em [0, 1]
  2. Treinamento  : SOM (4x4, 8x8, 16x16) e GNG (16, 64, 256), testando 3 configurações cada
  3. Inferência   : para cada pixel, acha o BMU (neurônio mais próximo) e pega a cor dele
  4. Reconstrução : remonta a imagem com as novas cores, salva e exibe
  Avaliação       : erro de quantização e erro topológico (+ PSNR como métrica extra)

Uso:
  uv run main.py                                   # roda as 5 imagens padrão
  uv run main.py --imagem data/input/arara.jpg     # só uma imagem
  uv run main.py --modelo som                      # só SOM (ou gng)
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

from common import graficos                                   # figuras
from common.imagem import carregar_imagem, extrair_pixels, reconstruir_imagem, salvar_imagem
from common.metricas import erro_quantizacao, erro_topologico, psnr, quantizar
from gng import GNG                                           # Growing Neural Gas (implementação própria)
from som import SOMXPySom                                     # SOM com a biblioteca XPySom

RAIZ = Path(__file__).parent   # pasta do projeto


# =============================================================================================
# 1) CONFIGURAÇÕES DOS EXPERIMENTOS
#    Aqui só DEFINIMOS o que vai ser testado. O treino acontece em rodar_som / rodar_gng.
# =============================================================================================

# Imagens que rodam quando não se passa --imagem (para adicionar uma imagem: só uma linha aqui)
IMAGENS_PADRAO = [
    "data/input/por_do_sol.jpg",
    "data/input/campo.jpg",
    "data/input/praia.jpg",
    "data/input/tintas.jpg",
    "data/input/arara.jpg",
]

# ---------------------------------- SOM ----------------------------------
SOM_GRIDS = [4, 8, 16]   # grades pedidas: 4x4 (16), 8x8 (64) e 16x16 (256 neurônios)
SOM_EPOCAS = 100         # quantas vezes a rede vê TODOS os pixels da imagem

# σ = largura da vizinhança: quantos vizinhos do vencedor também são puxados para o pixel.
# Começa grande (σ0, organiza o mapa) e diminui até σN (refina cada neurônio).
# A XPySom treina em lote (batch SOM): a taxa de aprendizado se cancela na conta,
# por isso o ajuste de hiperparâmetros do SOM é feito no σ.   (L = lado da grade)
SOM_CONFIGS = {
    "A (σ0=L/2, σN=1.0)": lambda L: dict(sigma_inicial=L / 2, sigma_final=1.0),            # padrão
    "B (σ0=L/2, σN=0.5)": lambda L: dict(sigma_inicial=L / 2, sigma_final=0.5),            # termina mais estreito
    "C (σ0=L/4, σN=1.0)": lambda L: dict(sigma_inicial=max(1.0, L / 4), sigma_final=1.0),  # começa mais estreito
}

# ---------------------------------- GNG ----------------------------------
GNG_TAMANHOS = [16, 64, 256]   # nº de neurônios pedidos
GNG_EPOCAS = 2                 # mínimo de passagens por todos os pixels (comparação justa com o SOM)

# εb  = taxa do vencedor (quanto o neurônio vencedor anda até o pixel)
# εn  = taxa dos vizinhos do vencedor (bem menor)
# λ   = a cada λ pixels, a rede ganha 1 neurônio novo (onde o erro é maior)
# idade_max = aresta que fica muito tempo sem ser usada é removida
GNG_CONFIGS = {
    "A (εb=0.2, εn=0.006, λ=100)": dict(eps_b=0.2, eps_n=0.006, lam=100, idade_max=50),     # valores do artigo
    "B (εb=0.05, εn=0.0006, λ=100)": dict(eps_b=0.05, eps_n=0.0006, lam=100, idade_max=50),  # taxas menores
    "C (εb=0.2, εn=0.006, λ=300)": dict(eps_b=0.2, eps_n=0.006, lam=300, idade_max=100),    # cresce mais devagar
}


# =============================================================================================
# 2) INFERÊNCIA + RECONSTRUÇÃO + MÉTRICAS  (etapas 3, 4 e avaliação)
#    Chamada depois de cada rede treinada, tanto do SOM quanto do GNG.
# =============================================================================================

def slug(txt: str) -> str:
    """'A (σ0=L/2, σN=1.0)' -> 'A'  (só a letra da configuração, para nomes de arquivo)."""
    return txt.split(" ")[0]


def avaliar(modelo, X, img, nome_modelo, rede, config, pasta_saida, tempo):
    # 3. Inferência: para cada pixel acha o BMU e pega a cor (pesos) dele
    novas_cores, bmu = quantizar(X, modelo.W)

    # 4. Reconstrução: volta a lista de pixels para o formato da imagem (altura x largura x 3)
    img_q = reconstruir_imagem(novas_cores, img.shape)
    caminho = pasta_saida / nome_modelo.lower() / f"{rede}_{slug(config)}.png"
    salvar_imagem(img_q, caminho)

    # Avaliação: as métricas desta rede (vira uma linha do resultados.csv)
    metricas = dict(
        modelo=nome_modelo, rede=rede, config=config,
        neuronios=len(modelo.W),
        cores_usadas=int(len(np.unique(bmu))),                           # neurônios que viraram BMU
        erro_quantizacao=erro_quantizacao(X, modelo.W),                  # distância média pixel → BMU
        erro_topologico=erro_topologico(X, modelo.W, modelo.vizinhos),   # 1º e 2º BMU não vizinhos
        mse=float(np.mean((X - novas_cores) ** 2)),
        psnr=psnr(img, img_q),                                           # qualidade da imagem (dB)
        tempo_s=tempo,
        imagem=str(caminho.relative_to(RAIZ)),
    )
    return metricas, img_q


# =============================================================================================
# 3) TREINO DO SOM  (etapa 2)
#    Para cada grade (4x4, 8x8, 16x16): treina as 3 configs, avalia e fica com a melhor.
# =============================================================================================

def rodar_som(X, img, pasta_saida, pasta_fig, rapido):
    resultados, melhores, curvas_por_grid = [], [], {}

    for L in SOM_GRIDS:                                  # laço 1: tamanhos da grade
        rede = f"{L}x{L}"
        candidatos = []

        for nome_cfg, cfg in SOM_CONFIGS.items():       # laço 2: configurações A, B, C
            # --- treino ---
            t0 = time.perf_counter()
            som = SOMXPySom(L, L, n_epocas=5 if rapido else SOM_EPOCAS, **cfg(L)).treinar(X)
            tempo = time.perf_counter() - t0

            # --- inferência, reconstrução e métricas ---
            res, img_q = avaliar(som, X, img, "SOM", rede, nome_cfg, pasta_saida, tempo)
            print(f"  SOM {rede:6s} {nome_cfg:24s} QE={res['erro_quantizacao']:.4f} "
                  f"TE={res['erro_topologico']:.4f} PSNR={res['psnr']:.2f}dB ({tempo:.1f}s)")
            resultados.append(res)
            candidatos.append((res, som, img_q))
            curvas_por_grid.setdefault(rede, {})[nome_cfg] = som.historico

        # --- escolhe a MELHOR configuração: a de menor erro de quantização ---
        res, som, img_q = min(candidatos, key=lambda c: c[0]["erro_quantizacao"])
        melhores.append((res, som, img_q))

        # --- figuras da melhor configuração desta grade ---
        graficos.comparacao(img, img_q, som.W, f"SOM {rede} — config {slug(res['config'])}",
                            pasta_fig / f"som_{rede}_comparacao.png")
        graficos.mapa_som(som, pasta_fig / f"som_{rede}_mapa.png")        # pesos no grid + U-Matrix
        graficos.mapa_som_rgb(som, X, pasta_fig / f"som_{rede}_rgb.png")  # grade no espaço RGB
        graficos.curvas_treino(curvas_por_grid[rede], f"SOM {rede} — erro de quantização no treino",
                               pasta_fig / f"som_{rede}_curvas.png")

    # figura com as 3 grades lado a lado (som_grade.png, usada no relatório)
    graficos.grade_quantizacoes(img, [(f"SOM {r['rede']}", q, np.clip(s.grid_cores(), 0, 1), r["psnr"])
                                      for r, s, q in melhores],
                                "SOM — melhor configuração por grid", pasta_fig / "som_grade.png")
    return resultados, melhores


# =============================================================================================
# 4) TREINO DO GNG  (etapa 2)
#    Mesma estrutura do SOM: para cada tamanho (16, 64, 256), treina as 3 configs e fica com a melhor.
# =============================================================================================

def rodar_gng(X, img, pasta_saida, pasta_fig, rapido):
    resultados, melhores, curvas_por_rede = [], [], {}

    for N in GNG_TAMANHOS:                               # laço 1: nº de neurônios
        rede = f"{N} neur."
        nome_arq = f"{N}n"
        candidatos = []

        for nome_cfg, cfg in GNG_CONFIGS.items():       # laço 2: configurações A, B, C
            # nº de iterações (pixels apresentados): tempo para crescer até N neurônios (N·λ)
            # + ajuste fino, e no mínimo GNG_EPOCAS passagens por todos os pixels
            if rapido:
                n_iter = int(N * cfg["lam"] * 1.5) // 5
                cfg = {**cfg, "lam": max(10, cfg["lam"] // 5)}   # λ menor para ainda crescer até N
            else:
                n_iter = max(int(N * cfg["lam"] * 1.5), GNG_EPOCAS * len(X))

            # --- treino ---
            t0 = time.perf_counter()
            gng = GNG(N, n_iter=n_iter, **cfg).treinar(X)
            tempo = time.perf_counter() - t0

            # --- inferência, reconstrução e métricas ---
            res, img_q = avaliar(gng, X, img, "GNG", nome_arq, nome_cfg, pasta_saida, tempo)
            res["rede"] = rede
            print(f"  GNG {rede:9s} {nome_cfg:30s} QE={res['erro_quantizacao']:.4f} "
                  f"TE={res['erro_topologico']:.4f} PSNR={res['psnr']:.2f}dB "
                  f"neur={res['neuronios']} ({tempo:.1f}s)")
            resultados.append(res)
            candidatos.append((res, gng, img_q))
            curvas_por_rede.setdefault(nome_arq, {})[nome_cfg] = gng.historico

        # --- escolhe a MELHOR configuração: a de menor erro de quantização ---
        res, gng, img_q = min(candidatos, key=lambda c: c[0]["erro_quantizacao"])
        melhores.append((res, gng, img_q))

        # --- figuras da melhor configuração deste tamanho ---
        graficos.comparacao(img, img_q, gng.W, f"GNG {rede} — config {slug(res['config'])}",
                            pasta_fig / f"gng_{nome_arq}_comparacao.png")
        graficos.rede_gng(gng, X, pasta_fig / f"gng_{nome_arq}_rgb.png")  # neurônios + arestas no RGB
        graficos.curvas_treino(curvas_por_rede[nome_arq], f"GNG {rede} — erro de quantização no treino",
                               pasta_fig / f"gng_{nome_arq}_curvas.png")

    # figura com os 3 tamanhos lado a lado (gng_grade.png, usada no relatório)
    graficos.grade_quantizacoes(img, [(f"GNG {r['rede']}", q, graficos._paleta_grid(g.W), r["psnr"])
                                      for r, g, q in melhores],
                                "GNG — melhor configuração por tamanho", pasta_fig / "gng_grade.png")
    return resultados, melhores


# =============================================================================================
# 5) TABELAS EM LaTeX (vão para report/tabelas/)
# =============================================================================================

def tabela_latex(resultados, caminho):
    """Todas as redes e configurações de uma imagem."""
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
    """SOM x GNG lado a lado (melhor configuração), com o mesmo número de neurônios."""
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


# =============================================================================================
# 6) PIPELINE COMPLETO DE UMA IMAGEM  (junta tudo: etapas 1 → 4 + avaliação)
# =============================================================================================

def processar_imagem(caminho, args):
    # ---------- Etapa 1: extração ----------
    caminho_img = RAIZ / caminho
    img = carregar_imagem(caminho_img, args.max_lado or None)   # abre em RGB (reduz p/ máx. 450 px)
    X = extrair_pixels(img)                                      # lista de pixels (N x 3) em [0, 1]
    nome = caminho_img.stem
    pasta_saida = RAIZ / "data" / "output" / nome                # imagens quantizadas + resultados.csv
    pasta_fig = RAIZ / "report" / "figures" / nome               # figuras do relatório
    print(f"\n=== Imagem {caminho_img.name}: {img.shape[1]}x{img.shape[0]} = {len(X)} pixels, "
          f"{len(np.unique(img.reshape(-1, 3), axis=0))} cores distintas ===")

    # ---------- Etapas 2, 3 e 4: treino, inferência e reconstrução ----------
    resultados, melhores = [], []
    if args.modelo in ("som", "todos"):
        print("Treinando SOM (XPySom)...")
        r, m = rodar_som(X, img, pasta_saida, pasta_fig, args.rapido)
        resultados += r
        melhores += m
    if args.modelo in ("gng", "todos"):
        print("Treinando GNG...")
        r, m = rodar_gng(X, img, pasta_saida, pasta_fig, args.rapido)
        resultados += r
        melhores += m

    # ---------- Avaliação: salva as métricas (CSV/JSON), tabelas e gráfico de erros ----------
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

    # ---------- Etapa 4 (exibir): prepara as janelas com as imagens reconstruídas ----------
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


# =============================================================================================
# 7) MAIN: lê as opções da linha de comando e roda cada imagem
# =============================================================================================

def main():
    p = argparse.ArgumentParser(description="Quantização de cores com SOM e GNG")
    p.add_argument("--imagem", nargs="+", default=IMAGENS_PADRAO,
                   help="uma ou mais imagens (padrão: as 5 imagens de data/input)")
    p.add_argument("--modelo", choices=["som", "gng", "todos"], default="todos")
    p.add_argument("--max-lado", type=int, default=450,
                   help="reduz a imagem para no máx. N pixels de lado (0 = tamanho original)")
    p.add_argument("--rapido", action="store_true", help="menos iterações, para testar o pipeline")
    p.add_argument("--sem-mostrar", action="store_true",
                   help="não exibe as imagens reconstruídas na tela (só salva os arquivos)")
    args = p.parse_args()

    for caminho in args.imagem:          # roda o pipeline completo em cada imagem
        processar_imagem(caminho, args)

    if not args.sem_mostrar:             # no final, abre todas as janelas de uma vez
        import matplotlib.pyplot as plt
        if plt.get_backend().lower() == "agg":
            print("(ambiente sem tela: as figuras foram salvas em report/figures/)")
        else:
            plt.show()


if __name__ == "__main__":
    main()
