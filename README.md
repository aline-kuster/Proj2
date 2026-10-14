# Projeto 2 — Quantização de Cores com SOM e GNG

Aprendizado não supervisionado: reduz a paleta de cores de uma imagem tratando cada pixel
como uma amostra 3D (R, G, B) e treinando um **Self-Organizing Map** e um **Growing Neural Gas**.

## Estrutura

```
quantizacao-cores/
├── main.py                 # pipeline: extração → treino → inferência → reconstrução → avaliação
├── som/som.py              # SOM: classe SOMXPySom (biblioteca XPySom, padrão) e SOM (implementação própria)
├── gng/gng.py              # implementação do GNG (Fritzke, 1995)
├── common/
│   ├── imagem.py           # carregar, extrair/normalizar pixels [0,1], reconstruir a matriz
│   ├── metricas.py         # BMU, quantização, erro de quantização, erro topológico, PSNR
│   └── graficos.py         # comparações, paletas, mapa do SOM, rede do GNG, curvas
├── data/
│   ├── input/              # imagens de entrada (por_do_sol, campo, praia, tintas)
│   └── output/<imagem>/    # imagens quantizadas (som/, gng/) + resultados.csv/.json
├── report/
│   ├── relatorio.tex       # relatório (LaTeX)
│   ├── relatorio.pdf
│   ├── figures/<imagem>/   # figuras geradas pelo main.py
│   └── tabelas/            # tabelas .tex geradas pelo main.py
├── pyproject.toml
├── uv.lock
└── .python-version
```

## Como rodar

```bash
uv sync
uv run main.py                                  # roda as 4 imagens (por_do_sol, campo, praia, tintas)
                                                # e ao final abre as imagens reconstruídas
uv run main.py --imagem data/input/praia.jpg    # só uma imagem
uv run main.py --modelo som                     # só SOM (ou --modelo gng)
uv run main.py --som-impl propria               # SOM com a implementação própria em vez da XPySom
uv run main.py --rapido                         # menos iterações (teste rápido)
uv run main.py --max-lado 0                     # usa todos os pixels da imagem original
uv run main.py --sem-mostrar                    # não abre as janelas (só salva os arquivos)
```

Para gerar o PDF do relatório: `cd report && latexmk -pdf relatorio.tex`.

## O que é executado

| Etapa | Descrição |
|---|---|
| Extração | carrega a imagem em RGB (redimensionada para no máx. 450 px de lado), extrai todos os pixels e normaliza em [0, 1] |
| Treinamento | SOM 4x4, 8x8, 16x16 (XPySom, 100 épocas) · GNG 16, 64, 256 neurônios (2 épocas) · 3 configurações cada |
| Inferência | para cada pixel encontra o BMU e troca a cor pelos pesos do neurônio |
| Reconstrução | remonta a matriz H×W×3, salva em `data/output/` e exibe na tela |
| Avaliação | erro de quantização, erro topológico, MSE, PSNR e cores usadas |

**Erro topológico:** fração de pixels cujo 1º e 2º BMU não são vizinhos
(SOM: adjacentes no grid, vizinhança-8; GNG: ligados por aresta).

### Configurações de hiperparâmetros

| Config | SOM | GNG |
|---|---|---|
| A | σ0=L/2, σN=1.0 | εb=0.2, εn=0.006, λ=100, idade_max=50 |
| B | σ0=L/2, σN=0.5 | εb=0.05, εn=0.0006, λ=100, idade_max=50 |
| C | σ0=L/4, σN=1.0 | εb=0.2, εn=0.006, λ=300, idade_max=100 |

A XPySom treina em lote (batch SOM): a taxa de aprendizado se cancela na atualização, então no SOM
o ajuste é feito na largura da vizinhança (σ). Para testar outras configurações, edite
`SOM_CONFIGS_XPYSOM` / `GNG_CONFIGS` no `main.py` (`SOM_CONFIGS` vale para a implementação própria).

## Resultados

Depois de rodar, os números ficam em `data/output/<imagem>/resultados.csv` e as tabelas
para o relatório em `report/tabelas/`.
