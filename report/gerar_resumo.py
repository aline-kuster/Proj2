"""Gera as tabelas-resumo e os gráficos comparativos do relatório a partir dos resultados.

Lê data/output/<imagem>/resultados.csv (gerados pelo main.py) e cria:
  report/tabelas/som_resumo.tex        melhor config. de cada grade do SOM, por imagem + média
  report/tabelas/gng_resumo.tex        melhor config. de cada rede do GNG, por imagem + média
  report/tabelas/hiperparametros.tex   efeito das configurações A/B/C (médias)
  report/tabelas/comparacao.tex        PSNR e erro topológico SOM x GNG
  report/figures/som_curvas.png        PSNR, EQ, ET e cores usadas x tamanho da grade
  report/figures/gng_curvas.png        idem para o GNG

Uso (depois de rodar o main.py):  uv run report/gerar_resumo.py
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
IMAGENS = ["por_do_sol", "campo", "praia", "tintas", "arara"]
TAM = {"SOM": ["4x4", "8x8", "16x16"], "GNG": ["16 neur.", "64 neur.", "256 neur."]}
ROT = {"SOM": ["4x4", "8x8", "16x16"], "GNG": ["16", "64", "256"]}
TABELAS = RAIZ / "report" / "tabelas"
FIGURAS = RAIZ / "report" / "figures"


def tex_nome(n):
    return r"\texttt{" + n.replace("_", r"\_") + "}"


def num(x, casas=4):
    return f"{x:.{casas}f}".replace(".", "{,}")


def carregar():
    dados = {}
    for img in IMAGENS:
        with open(RAIZ / "data" / "output" / img / "resultados.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                for k in ("erro_quantizacao", "erro_topologico", "psnr", "mse"):
                    r[k] = float(r[k])
                r["cores_usadas"] = int(r["cores_usadas"])
                r["cfg"] = r["config"][0]
                dados.setdefault((img, r["modelo"], r["rede"]), []).append(r)
    return dados


def melhor(dados, img, modelo, rede):
    return min(dados[(img, modelo, rede)], key=lambda r: r["erro_quantizacao"])


def tabela_resumo(dados, modelo):
    col = "Grade" if modelo == "SOM" else "Neurônios"
    L = [r"\begin{tabular}{llcccccc}", r"\toprule",
         rf"Imagem & {col} & Config. & EQ & ET & PSNR (dB) & Cores \\", r"\midrule"]
    medias = {t: [] for t in TAM[modelo]}
    for img in IMAGENS:
        for t, rot in zip(TAM[modelo], ROT[modelo]):
            r = melhor(dados, img, modelo, t)
            medias[t].append(r)
            L.append(f"{tex_nome(img)} & {rot} & {r['cfg']} & {num(r['erro_quantizacao'])} & "
                     f"{num(r['erro_topologico'])} & {num(r['psnr'], 2)} & {r['cores_usadas']} \\\\")
        L.append(r"\midrule")
    for t, rot in zip(TAM[modelo], ROT[modelo]):
        rs = medias[t]
        L.append(rf"\textbf{{Média}} & {rot} & -- & {num(np.mean([r['erro_quantizacao'] for r in rs]))} & "
                 f"{num(np.mean([r['erro_topologico'] for r in rs]))} & "
                 f"{num(np.mean([r['psnr'] for r in rs]), 2)} & "
                 f"{num(np.mean([r['cores_usadas'] for r in rs]), 1)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TABELAS / f"{modelo.lower()}_resumo.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def tabela_hiperparametros(dados):
    """Para cada modelo e configuração: média do EQ e do ET relativos à melhor config. de cada caso,
    e em quantos dos 15 casos (5 imagens x 3 tamanhos) a config. teve o menor EQ / menor ET."""
    L = [r"\begin{tabular}{llcccc}", r"\toprule",
         r"Modelo & Config. & EQ médio & ET médio & Menor EQ & Menor ET \\", r"\midrule"]
    for modelo in ("SOM", "GNG"):
        casos = [dados[(img, modelo, t)] for img in IMAGENS for t in TAM[modelo]]
        for cfg in "ABC":
            eq = [next(r for r in c if r["cfg"] == cfg)["erro_quantizacao"] for c in casos]
            et = [next(r for r in c if r["cfg"] == cfg)["erro_topologico"] for c in casos]
            v_eq = sum(min(c, key=lambda r: r["erro_quantizacao"])["cfg"] == cfg for c in casos)
            v_et = sum(min(c, key=lambda r: r["erro_topologico"])["cfg"] == cfg for c in casos)
            L.append(f"{modelo} & {cfg} & {num(np.mean(eq))} & {num(np.mean(et))} & "
                     f"{v_eq}/{len(casos)} & {v_et}/{len(casos)} \\\\")
        if modelo == "SOM":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TABELAS / "hiperparametros.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def tabela_comparacao(dados):
    L = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r" & \multicolumn{2}{c}{16 protótipos} & \multicolumn{2}{c}{64 protótipos} & \multicolumn{2}{c}{256 protótipos} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Imagem & SOM & GNG & SOM & GNG & SOM & GNG \\", r"\midrule",
         r"\multicolumn{7}{l}{\textit{PSNR (dB) -- quanto maior, melhor}} \\"]
    for chave, titulo, casas in (("psnr", None, 2),
                                 ("erro_topologico", r"\textit{Erro topológico -- quanto menor, melhor}", 4)):
        if titulo:
            L += [r"\midrule", rf"\multicolumn{{7}}{{l}}{{{titulo}}} \\"]
        medias = []
        for img in IMAGENS:
            vals = []
            for i in range(3):
                vals += [melhor(dados, img, "SOM", TAM["SOM"][i])[chave],
                         melhor(dados, img, "GNG", TAM["GNG"][i])[chave]]
            medias.append(vals)
            L.append(tex_nome(img) + " & " + " & ".join(num(v, casas) for v in vals) + r" \\")
        m = np.mean(medias, axis=0)
        L.append(r"\textbf{Média} & " + " & ".join(rf"\textbf{{{num(v, casas)}}}" for v in m) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TABELAS / "comparacao.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def grafico_curvas(dados, modelo):
    metricas = [("psnr", "PSNR (dB)"), ("erro_quantizacao", "Erro de quantização"),
                ("erro_topologico", "Erro topológico"), ("cores_usadas", "Cores usadas")]
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    x = [16, 64, 256]
    for ax, (chave, titulo) in zip(axs.ravel(), metricas):
        for img in IMAGENS:
            y = [melhor(dados, img, modelo, t)[chave] for t in TAM[modelo]]
            ax.plot(x, y, marker="o", label=img)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.set_xticklabels(ROT[modelo])
        ax.set_title(titulo)
        ax.set_xlabel("Grade" if modelo == "SOM" else "Neurônios")
        ax.grid(alpha=0.3)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGURAS / f"{modelo.lower()}_curvas.png", dpi=150)
    plt.close(fig)


def main():
    TABELAS.mkdir(parents=True, exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)
    dados = carregar()
    for modelo in ("SOM", "GNG"):
        tabela_resumo(dados, modelo)
        grafico_curvas(dados, modelo)
    tabela_hiperparametros(dados)
    tabela_comparacao(dados)
    print("Tabelas em report/tabelas/ e gráficos em report/figures/")


if __name__ == "__main__":
    main()
