"""Self-Organizing Map (Kohonen) para quantização de cores."""
import numpy as np

from common.metricas import dois_melhores, erro_quantizacao


class SOM:
    """SOM com grid retangular, vizinhança gaussiana e decaimento exponencial.

    Parâmetros
    ----------
    linhas, colunas : tamanho do grid (ex.: 4x4, 8x8, 16x16)
    lr_inicial, lr_final : taxa de aprendizado no início e no fim do treino
    sigma_inicial, sigma_final : raio da vizinhança no início e no fim
        (padrão: sigma_inicial = metade do maior lado do grid)
    n_iter : número de iterações (uma amostra/pixel aleatório por iteração)
    """

    def __init__(self, linhas, colunas, lr_inicial=0.5, lr_final=0.01,
                 sigma_inicial=None, sigma_final=0.5, n_iter=20_000, seed=42):
        self.linhas, self.colunas = linhas, colunas
        self.n_neuronios = linhas * colunas
        self.lr_inicial, self.lr_final = lr_inicial, lr_final
        self.sigma_inicial = sigma_inicial if sigma_inicial is not None else max(linhas, colunas) / 2
        self.sigma_final = sigma_final
        self.n_iter = n_iter
        self.rng = np.random.default_rng(seed)

        # posição (linha, coluna) de cada neurônio no grid e distâncias² entre eles no grid
        r, c = np.divmod(np.arange(self.n_neuronios), colunas)
        self.coords = np.stack([r, c], axis=1).astype(np.float64)
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        self.dist2_grid = (diff**2).sum(-1)
        self.W = None
        self.historico = []  # (iteração, erro de quantização) durante o treino

    def treinar(self, X, n_avaliacoes=20, amostra_avaliacao=5_000):
        n = len(X)
        # inicialização: pesos = pixels aleatórios da imagem
        self.W = X[self.rng.choice(n, self.n_neuronios, replace=False)].copy()
        Xval = X[self.rng.choice(n, min(amostra_avaliacao, n), replace=False)]
        passo_log = max(1, self.n_iter // n_avaliacoes)
        indices = self.rng.integers(0, n, self.n_iter)

        for t in range(self.n_iter):
            frac = t / self.n_iter
            lr = self.lr_inicial * (self.lr_final / self.lr_inicial) ** frac
            sigma = self.sigma_inicial * (self.sigma_final / self.sigma_inicial) ** frac

            x = X[indices[t]]
            bmu = np.argmin(((self.W - x) ** 2).sum(1))           # competição
            h = np.exp(-self.dist2_grid[bmu] / (2 * sigma**2))     # cooperação
            self.W += lr * h[:, None] * (x - self.W)               # adaptação

            if t % passo_log == 0 or t == self.n_iter - 1:
                self.historico.append((t, erro_quantizacao(Xval, self.W)))
        return self

    def vizinhos(self, b1, b2):
        """Dois neurônios são vizinhos se forem adjacentes no grid (vizinhança-8)."""
        d = np.abs(self.coords[b1] - self.coords[b2]).max(1)
        return d <= 1

    def grid_cores(self):
        """Pesos no formato do grid (linhas x colunas x 3) — é o 'mapa de cores' aprendido."""
        return self.W.reshape(self.linhas, self.colunas, 3)

    def umatrix(self):
        """U-Matrix: distância média de cada neurônio aos vizinhos-4 no grid."""
        G = self.grid_cores()
        U = np.zeros((self.linhas, self.colunas))
        for i in range(self.linhas):
            for j in range(self.colunas):
                ds = [np.linalg.norm(G[i, j] - G[i + di, j + dj])
                      for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1))
                      if 0 <= i + di < self.linhas and 0 <= j + dj < self.colunas]
                U[i, j] = np.mean(ds)
        return U

    def bmus(self, X):
        return dois_melhores(X, self.W)[0]


class SOMXPySom(SOM):
    """SOM treinado com a biblioteca XPySom (treino em lote vetorizado, bem mais rápido).

    Usa os mesmos hiperparâmetros (lr_inicial/final, sigma_inicial/final) e a mesma interface
    da classe SOM acima, então inferência, métricas e gráficos funcionam igual.
    n_epocas : quantas vezes TODOS os pixels da imagem são apresentados à rede.
    """

    def __init__(self, linhas, colunas, n_epocas=100, seed=42, **kwargs):
        kwargs.pop("n_iter", None)
        super().__init__(linhas, colunas, seed=seed, **kwargs)
        self.n_epocas = n_epocas
        self.seed = seed

    def treinar(self, X, n_avaliacoes=20, amostra_avaliacao=5_000):
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from xpysom import XPySom  # importa em silêncio (aviso de CuPy/GPU ausente)

        som = XPySom(self.linhas, self.colunas, 3,
                     sigma=self.sigma_inicial, sigmaN=self.sigma_final,
                     learning_rate=self.lr_inicial, learning_rateN=self.lr_final,
                     decay_function="exponential", neighborhood_function="gaussian",
                     topology="rectangular", random_seed=self.seed)
        som.random_weights_init(X)  # pesos iniciais = pixels aleatórios da imagem

        Xval = X[self.rng.choice(len(X), min(amostra_avaliacao, len(X)), replace=False)]
        passo_log = max(1, self.n_epocas // n_avaliacoes)
        for e in range(self.n_epocas):  # uma época por vez, para registrar a curva de erro
            som.train(X, self.n_epocas, iter_beg=e, iter_end=e + 1)
            if e % passo_log == 0 or e == self.n_epocas - 1:
                self.W = np.asarray(som.get_weights(), dtype=np.float64).reshape(-1, 3)
                self.historico.append(((e + 1) * len(X), erro_quantizacao(Xval, self.W)))
        self.W = np.asarray(som.get_weights(), dtype=np.float64).reshape(-1, 3)
        return self
