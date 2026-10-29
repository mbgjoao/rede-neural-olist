
"""Rede neural do zero em NumPy para classificação binária.

A estrutura segue a da playlist de referência ("Rede neural do zero em
Python", canal Machine Learning para humanos): uma classe que guarda os
pesos num dicionário (model_dict), com métodos de inicialização,
propagação direta (forward), perda (loss), retropropagação
(backpropagation) e treinamento (fit).

Diferenças em relação à referência (ver README.md):
  * ativação oculta: Leaky ReLU no lugar da tangente hiperbólica;
  * saída: 1 neurônio sigmoide + entropia cruzada binária no lugar de
    2 neurônios softmax + entropia cruzada categórica;
  * arquitetura com número arbitrário de camadas ocultas (a final tem uma,
    com 16 neurônios);
  * inicialização He (sqrt(2 / n_entradas)), adequada à família ReLU;
  * minilotes, gradiente médio por lote e parada antecipada pela
    perda de validação.
"""

from pathlib import Path
import json

import numpy as np


# ---------------------------------------------------------------------------
# Funções de ativação e suas derivadas
# ---------------------------------------------------------------------------

def leaky_relu(z, alfa):
    return np.where(z > 0, z, alfa * z)


def derivada_leaky_relu(z, alfa):
    return np.where(z > 0, 1.0, alfa)


def tanh(z, alfa=None):
    return np.tanh(z)


def derivada_tanh(z, alfa=None):
    return 1.0 - np.power(np.tanh(z), 2)


def sigmoide(z):
    # Forma estável: evita overflow de exp para |z| grande.
    saida = np.empty_like(z, dtype=float)
    positivo = z >= 0
    saida[positivo] = 1.0 / (1.0 + np.exp(-z[positivo]))
    exp_z = np.exp(z[~positivo])
    saida[~positivo] = exp_z / (1.0 + exp_z)
    return saida


def derivada_sigmoide(z):
    # Máximo de 0,25 em z = 0 (playlist 1, vídeo #19). A retropropagação não
    # a chama diretamente: com entropia cruzada binária, a regra da cadeia
    # dL/dp * sigmoide'(z) se simplifica para p - y (ver os testes).
    s = sigmoide(z)
    return s * (1.0 - s)


ATIVACOES = {
    "leaky_relu": (leaky_relu, derivada_leaky_relu),
    # A tangente hiperbólica é a ativação da referência; fica disponível
    # apenas para o experimento de comparação.
    "tanh": (tanh, derivada_tanh),
}


# ---------------------------------------------------------------------------
# Rede neural
# ---------------------------------------------------------------------------

class RedeNeural:
    """Perceptron multicamadas para classificação binária.

    camadas: lista com o número de neurônios de cada camada, da entrada à
        saída, por exemplo [39, 32, 16, 1]. A última deve ser 1.
    """

    def __init__(self, camadas, ativacao="leaky_relu", alfa=0.01, semente=42):
        if len(camadas) < 2 or camadas[-1] != 1:
            raise ValueError("A última camada deve ter 1 neurônio.")
        if ativacao not in ATIVACOES:
            raise ValueError(f"Ativação desconhecida: {ativacao}")

        self.camadas = list(camadas)
        self.ativacao = ativacao
        self.alfa = alfa
        self.semente = semente
        self.f, self.df = ATIVACOES[ativacao]
        self.gerador = np.random.default_rng(semente)

        # model_dict guarda W1, b1, W2, b2, ... como na referência.
        # W_l tem formato (neurônios da camada anterior, neurônios da camada)
        # para que Z = A_anterior . W + b, com uma amostra por linha.
        self.model_dict = {}
        for l in range(1, len(camadas)):
            n_entrada, n_saida = camadas[l - 1], camadas[l]
            escala = np.sqrt(2.0 / n_entrada)  # inicialização He
            self.model_dict[f"W{l}"] = (
                self.gerador.standard_normal((n_entrada, n_saida)) * escala
            )
            self.model_dict[f"b{l}"] = np.zeros((1, n_saida))

    @property
    def n_camadas(self):
        """Número de camadas com pesos (ocultas + saída)."""
        return len(self.camadas) - 1

    # -- Propagação direta -------------------------------------------------

    def forward(self, X):
        """Calcula as probabilidades P(satisfeito = 1 | x).

        Guarda os Z (somas ponderadas) e os A (ativações) de cada camada,
        necessários para a retropropagação.
        """
        A = np.asarray(X, dtype=float)
        self.cache = {"A0": A}
        for l in range(1, self.n_camadas + 1):
            Z = A @ self.model_dict[f"W{l}"] + self.model_dict[f"b{l}"]
            if l == self.n_camadas:
                A = sigmoide(Z)
            else:
                A = self.f(Z, self.alfa)
            self.cache[f"Z{l}"] = Z
            self.cache[f"A{l}"] = A
        return A

    # -- Função de perda ---------------------------------------------------

    @staticmethod
    def loss(y, p, eps=1e-12):
        """Entropia cruzada binária média.

        L = -1/m * soma[ y*log(p) + (1-y)*log(1-p) ]
        """
        # Os dois viram coluna (m, 1): com p em formato (m,), o NumPy faria
        # broadcasting para uma matriz (m, m) e a média sairia errada.
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        p = np.clip(np.asarray(p, dtype=float).reshape(-1, 1), eps, 1.0 - eps)
        if y.shape != p.shape:
            raise ValueError(f"Formatos incompatíveis: {y.shape} e {p.shape}")
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    # -- Retropropagação ---------------------------------------------------

    def gradientes(self, y):
        """Gradientes da perda média em relação a cada W e b.

        Usa os valores guardados pelo último forward.
        """
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        m = y.shape[0]
        L = self.n_camadas
        grads = {}

        # Sigmoide + entropia cruzada binária: dL/dZ_L = (p - y) / m.
        # É o equivalente binário do "softmax - 1 na classe correta"
        # usado na referência.
        delta = (self.cache[f"A{L}"] - y) / m
        for l in range(L, 0, -1):
            A_anterior = self.cache[f"A{l - 1}"]
            grads[f"W{l}"] = A_anterior.T @ delta
            grads[f"b{l}"] = np.sum(delta, axis=0, keepdims=True)
            if l > 1:
                # Propaga o erro para a camada anterior e multiplica pela
                # derivada da ativação oculta.
                delta = (delta @ self.model_dict[f"W{l}"].T) * self.df(
                    self.cache[f"Z{l - 1}"], self.alfa
                )
        return grads

    def backpropagation(self, y, taxa_aprendizado):
        """Calcula os gradientes e atualiza os pesos (descida do gradiente)."""
        grads = self.gradientes(y)
        for nome, gradiente in grads.items():
            self.model_dict[nome] -= taxa_aprendizado * gradiente
        return grads

    # -- Treinamento -------------------------------------------------------

    def fit(
        self,
        X,
        y,
        X_val,
        y_val,
        epocas=200,
        taxa_aprendizado=0.05,
        tamanho_lote=256,
        paciencia=15,
        verbose=True,
    ):
        """Treina com minilotes e parada antecipada pela perda de validação.

        Conta como melhora uma queda de mais de 1e-6 na perda de validação.
        Ao final, restaura os pesos da época com menor perda de validação.
        Retorna o histórico de perdas.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        m = X.shape[0]
        historico = {"epoca": [], "perda_treino": [], "perda_validacao": []}
        melhor_perda = np.inf
        melhores_pesos = None
        sem_melhora = 0
        self.melhor_epoca = 0

        for epoca in range(1, epocas + 1):
            ordem = self.gerador.permutation(m)
            for inicio in range(0, m, tamanho_lote):
                lote = ordem[inicio:inicio + tamanho_lote]
                self.forward(X[lote])
                self.backpropagation(y[lote], taxa_aprendizado)

            perda_treino = self.loss(y, self.forward(X))
            perda_val = self.loss(y_val, self.forward(X_val))
            historico["epoca"].append(epoca)
            historico["perda_treino"].append(perda_treino)
            historico["perda_validacao"].append(perda_val)

            if not np.isfinite(perda_treino):
                raise FloatingPointError(f"Perda não finita na época {epoca}.")

            if perda_val < melhor_perda - 1e-6:
                melhor_perda = perda_val
                melhores_pesos = {k: v.copy() for k, v in self.model_dict.items()}
                self.melhor_epoca = epoca
                sem_melhora = 0
            else:
                sem_melhora += 1

            if verbose and (epoca == 1 or epoca % 10 == 0):
                print(
                    f"Época {epoca:4d}/{epocas}  perda treino {perda_treino:.4f}"
                    f"  perda validação {perda_val:.4f}"
                )
            if sem_melhora >= paciencia:
                if verbose:
                    print(
                        f"Parada antecipada na época {epoca}; melhor época: "
                        f"{self.melhor_epoca} (perda val. {melhor_perda:.4f})"
                    )
                break

        if melhores_pesos is not None:
            self.model_dict = melhores_pesos
        return historico

    # -- Previsão ----------------------------------------------------------

    def prever_proba(self, X):
        return self.forward(X).ravel()

    def prever(self, X, limiar=0.5):
        return (self.prever_proba(X) >= limiar).astype(int)

    # -- Persistência ------------------------------------------------------

    def salvar(self, caminho):
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            caminho,
            **self.model_dict,
            meta=json.dumps(
                {
                    "camadas": self.camadas,
                    "ativacao": self.ativacao,
                    "alfa": self.alfa,
                    "semente": self.semente,
                }
            ),
        )

    @classmethod
    def carregar(cls, caminho):
        # "with" fecha o arquivo .npz (no Windows, um arquivo aberto
        # não pode ser sobrescrito nem apagado).
        with np.load(caminho) as dados:
            meta = json.loads(str(dados["meta"]))
            rede = cls(
                meta["camadas"], meta["ativacao"], meta["alfa"], meta["semente"]
            )
            rede.model_dict = {k: dados[k].copy() for k in rede.model_dict}
        return rede
