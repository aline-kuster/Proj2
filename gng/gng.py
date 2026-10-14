"""Growing Neural Gas (Fritzke, 1995) para quantização de cores."""
import numpy as np

from common.metricas import dois_melhores, erro_quantizacao


class GNG:
    """Growing Neural Gas.

    Parâmetros
    ----------
    max_neuronios : tamanho final da rede (ex.: 16, 64, 256)
    eps_b : taxa de aprendizado do vencedor
    eps_n : taxa de aprendizado dos vizinhos topológicos do vencedor
    idade_max : idade máxima de uma aresta antes de ser removida
    lam (λ) : a cada λ iterações um novo neurônio é inserido
    alpha : fator de redução do erro dos neurônios q e f na inserção
    d : decaimento global do erro acumulado a cada iteração
    n_iter : total de iterações (padrão: tempo para crescer até max_neuronios + 50% de ajuste fino)
    """

    def __init__(self, max_neuronios, eps_b=0.2, eps_n=0.006, idade_max=50, lam=100,
                 alpha=0.5, d=0.995, n_iter=None, seed=42):
        self.max_neuronios = max_neuronios
        self.eps_b, self.eps_n = eps_b, eps_n
        self.idade_max, self.lam = idade_max, lam
        self.alpha, self.d = alpha, d
        self.n_iter = n_iter if n_iter is not None else int(max_neuronios * lam * 1.5)
        self.rng = np.random.default_rng(seed)
        self.historico = []  # (iteração, nº de neurônios, erro de quantização)

    def treinar(self, X, n_avaliacoes=20, amostra_avaliacao=5_000):
        N = self.max_neuronios
        n = len(X)
        self.pesos = np.zeros((N, 3))
        self.ativo = np.zeros(N, dtype=bool)
        self.erro = np.zeros(N)
        self.idade = -np.ones((N, N), dtype=np.int64)  # -1 = sem aresta

        # começa com 2 neurônios em posições aleatórias (pixels da imagem)
        self.pesos[:2] = X[self.rng.choice(n, 2, replace=False)]
        self.ativo[:2] = True

        Xval = X[self.rng.choice(n, min(amostra_avaliacao, n), replace=False)]
        passo_log = max(1, self.n_iter // n_avaliacoes)
        indices = self.rng.integers(0, n, self.n_iter)

        for t in range(1, self.n_iter + 1):
            x = X[indices[t - 1]]
            ids = np.flatnonzero(self.ativo)
            dist = ((self.pesos[ids] - x) ** 2).sum(1)
            o = np.argpartition(dist, 1)[:2]
            o = o[np.argsort(dist[o])]
            s1, s2 = ids[o[0]], ids[o[1]]

            # acumula erro e move vencedor + vizinhos topológicos
            self.erro[s1] += dist[o[0]]
            self.pesos[s1] += self.eps_b * (x - self.pesos[s1])
            viz = np.flatnonzero(self.idade[s1] >= 0)
            self.pesos[viz] += self.eps_n * (x - self.pesos[viz])

            # envelhece arestas de s1 e cria/zera a aresta s1-s2
            self.idade[s1, viz] += 1
            self.idade[viz, s1] += 1
            self.idade[s1, s2] = self.idade[s2, s1] = 0

            # remove arestas velhas e neurônios que ficaram isolados
            velhas = np.flatnonzero(self.idade[s1] > self.idade_max)
            if len(velhas):
                self.idade[s1, velhas] = self.idade[velhas, s1] = -1
                for u in np.append(velhas, s1):
                    if self.ativo.sum() > 2 and not (self.idade[u] >= 0).any():
                        self._remover(u)

            # inserção de um novo neurônio a cada λ iterações
            if t % self.lam == 0 and self.ativo.sum() < N:
                self._inserir()

            self.erro[self.ativo] *= self.d

            if t % passo_log == 0 or t == self.n_iter:
                self.historico.append((t, int(self.ativo.sum()),
                                       erro_quantizacao(Xval, self.pesos[self.ativo])))

        # compacta: mantém só os neurônios ativos
        ids = np.flatnonzero(self.ativo)
        self.W = self.pesos[ids].copy()
        self.adj = self.idade[np.ix_(ids, ids)] >= 0
        return self

    def _remover(self, u):
        self.ativo[u] = False
        self.erro[u] = 0
        self.idade[u, :] = self.idade[:, u] = -1

    def _inserir(self):
        ids = np.flatnonzero(self.ativo)
        q = ids[np.argmax(self.erro[ids])]               # neurônio com maior erro
        viz = np.flatnonzero(self.idade[q] >= 0)
        if len(viz) == 0:
            return
        f = viz[np.argmax(self.erro[viz])]               # vizinho de q com maior erro
        r = np.flatnonzero(~self.ativo)[0]               # posição livre para o novo neurônio
        self.pesos[r] = 0.5 * (self.pesos[q] + self.pesos[f])
        self.ativo[r] = True
        self.idade[q, f] = self.idade[f, q] = -1
        self.idade[q, r] = self.idade[r, q] = 0
        self.idade[f, r] = self.idade[r, f] = 0
        self.erro[q] *= self.alpha
        self.erro[f] *= self.alpha
        self.erro[r] = self.erro[q]

    def vizinhos(self, b1, b2):
        """Dois neurônios são vizinhos se existir uma aresta entre eles."""
        return self.adj[b1, b2]

    def arestas(self):
        i, j = np.nonzero(np.triu(self.adj))
        return np.stack([i, j], axis=1)

    def bmus(self, X):
        return dois_melhores(X, self.W)[0]
