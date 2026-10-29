
"""Testes da rede neural, da matemática e dos dados preparados.

Executar a partir da raiz do projeto:
    python -m unittest discover -s tests -v
"""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from rede_neural import (  # noqa: E402
    RedeNeural, derivada_leaky_relu, derivada_sigmoide, derivada_tanh,
    leaky_relu, sigmoide,
)
from rede_referencia import NnModel  # noqa: E402
from metricas import avaliar, escolher_limiar  # noqa: E402


def dados_sinteticos(m=200, n=5, semente=0):
    """Problema linearmente separável com rótulos 0/1."""
    g = np.random.default_rng(semente)
    X = g.standard_normal((m, n))
    w = g.standard_normal(n)
    y = (X @ w > 0).astype(int)
    return X, y


def gradiente_numerico(rede, X, y, nome, h=1e-6):
    """Diferenças finitas centrais: (L(p + h) - L(p - h)) / 2h."""
    parametro = rede.model_dict[nome]
    aproximado = np.zeros_like(parametro)
    for indice in np.ndindex(parametro.shape):
        original = parametro[indice]
        parametro[indice] = original + h
        perda_mais = rede.loss(y, rede.forward(X))
        parametro[indice] = original - h
        perda_menos = rede.loss(y, rede.forward(X))
        parametro[indice] = original
        aproximado[indice] = (perda_mais - perda_menos) / (2 * h)
    return aproximado


class TestFormatos(unittest.TestCase):
    def test_formatos_dos_parametros(self):
        rede = RedeNeural([39, 32, 16, 1])
        esperado = {
            "W1": (39, 32), "b1": (1, 32),
            "W2": (32, 16), "b2": (1, 16),
            "W3": (16, 1), "b3": (1, 1),
        }
        self.assertEqual(
            {k: v.shape for k, v in rede.model_dict.items()}, esperado
        )

    def test_vieses_iniciam_em_zero(self):
        rede = RedeNeural([39, 32, 16, 1])
        for l in (1, 2, 3):
            self.assertTrue(np.all(rede.model_dict[f"b{l}"] == 0))

    def test_formato_da_saida_e_probabilidades_validas(self):
        rede = RedeNeural([39, 32, 16, 1])
        X = np.random.default_rng(1).standard_normal((50, 39)) * 5
        p = rede.forward(X)
        self.assertEqual(p.shape, (50, 1))
        self.assertEqual(rede.prever_proba(X).shape, (50,))
        self.assertTrue(np.all((p > 0) & (p < 1)))
        self.assertTrue(set(np.unique(rede.prever(X))) <= {0, 1})

    def test_valores_finitos(self):
        rede = RedeNeural([39, 32, 16, 1])
        X = np.random.default_rng(2).standard_normal((100, 39)) * 10
        self.assertTrue(all(np.isfinite(v).all() for v in rede.model_dict.values()))
        self.assertTrue(np.isfinite(rede.forward(X)).all())

    def test_sigmoide_estavel_para_valores_extremos(self):
        z = np.array([[-1000.0], [-30.0], [0.0], [30.0], [1000.0]])
        with np.errstate(over="raise"):
            s = sigmoide(z)
        self.assertTrue(np.isfinite(s).all())
        self.assertAlmostEqual(float(s[2, 0]), 0.5)
        self.assertTrue(np.all(np.diff(s.ravel()) >= 0))

    def test_rejeita_saida_diferente_de_um_neuronio(self):
        with self.assertRaises(ValueError):
            RedeNeural([39, 16, 2])


class TestDerivadas(unittest.TestCase):
    """Derivadas das ativações conferidas por diferenças finitas."""

    # Pontos longe de 0, onde a Leaky ReLU tem um "bico" sem derivada.
    z = np.array([[-3.0, -1.2, -0.4, 0.3, 1.1, 2.5]])
    h = 1e-6

    def numerica(self, f):
        return (f(self.z + self.h) - f(self.z - self.h)) / (2 * self.h)

    def test_leaky_relu_valores_e_derivada(self):
        np.testing.assert_allclose(
            leaky_relu(np.array([2.0, 0.0, -2.0]), 0.01), [2.0, 0.0, -0.02]
        )
        np.testing.assert_allclose(
            derivada_leaky_relu(self.z, 0.01),
            self.numerica(lambda z: leaky_relu(z, 0.01)), atol=1e-8,
        )

    def test_derivada_tanh(self):
        np.testing.assert_allclose(
            derivada_tanh(self.z), self.numerica(np.tanh), atol=1e-8
        )

    def test_derivada_sigmoide_tem_maximo_025(self):
        np.testing.assert_allclose(
            derivada_sigmoide(self.z), self.numerica(sigmoide), atol=1e-8
        )
        grade = np.linspace(-6, 6, 1201).reshape(1, -1)
        valores = derivada_sigmoide(grade)
        self.assertAlmostEqual(float(valores.max()), 0.25)
        self.assertAlmostEqual(float(grade[0, valores.argmax()]), 0.0)

    def test_erro_da_saida_e_p_menos_y(self):
        # Regra da cadeia completa na saída: dL/dz = dL/dp * sigmoide'(z).
        # Com entropia cruzada binária, isso se reduz a (p - y) / m.
        g = np.random.default_rng(0)
        z = g.normal(0, 2, (10, 1))
        y = g.integers(0, 2, (10, 1)).astype(float)
        p, m = sigmoide(z), len(y)
        dl_dp = -(y / p - (1 - y) / (1 - p)) / m
        np.testing.assert_allclose(dl_dp * derivada_sigmoide(z), (p - y) / m)


class TestPerda(unittest.TestCase):
    def test_previsao_perfeita_tem_perda_quase_zero(self):
        y = np.array([0, 1, 1, 0])
        p = np.array([[1e-9], [1 - 1e-9], [1 - 1e-9], [1e-9]])
        self.assertLess(RedeNeural.loss(y, p), 1e-6)

    def test_probabilidade_meio_vale_log2(self):
        y = np.array([0, 1, 1, 0])
        p = np.full((4, 1), 0.5)
        self.assertAlmostEqual(RedeNeural.loss(y, p), np.log(2), places=12)

    def test_perda_cresce_quando_erra_mais(self):
        y = np.array([1])
        perdas = [RedeNeural.loss(y, np.array([[q]])) for q in (0.9, 0.6, 0.3, 0.1)]
        self.assertTrue(all(a < b for a, b in zip(perdas, perdas[1:])))

    def test_perda_igual_para_vetor_e_coluna(self):
        # Regressão: p com formato (m,) gerava broadcasting (m, m).
        y = np.array([0, 1, 1, 0, 1])
        p = np.array([0.2, 0.7, 0.9, 0.4, 0.6])
        esperado = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        self.assertAlmostEqual(RedeNeural.loss(y, p), esperado, places=12)
        self.assertAlmostEqual(RedeNeural.loss(y, p.reshape(-1, 1)), esperado,
                               places=12)

    def test_perda_rejeita_tamanhos_diferentes(self):
        with self.assertRaises(ValueError):
            RedeNeural.loss(np.array([0, 1, 1]), np.array([0.5, 0.5]))

    def test_perda_finita_com_probabilidades_extremas(self):
        y = np.array([1, 0])
        p = np.array([[0.0], [1.0]])
        self.assertTrue(np.isfinite(RedeNeural.loss(y, p)))


class TestGradientes(unittest.TestCase):
    def verificar(self, ativacao):
        X, y = dados_sinteticos(m=20, n=6, semente=3)
        rede = RedeNeural([6, 5, 4, 1], ativacao=ativacao, alfa=0.1, semente=7)
        # Vieses diferentes de zero para testar também o termo b.
        for nome in rede.model_dict:
            if nome.startswith("b"):
                rede.model_dict[nome] += rede.gerador.normal(
                    0, 0.1, rede.model_dict[nome].shape
                )
        rede.forward(X)
        analitico = rede.gradientes(y)
        for nome in rede.model_dict:
            numerico = gradiente_numerico(rede, X, y, nome)
            erro = np.linalg.norm(analitico[nome] - numerico) / max(
                1e-12, np.linalg.norm(analitico[nome]) + np.linalg.norm(numerico)
            )
            with self.subTest(ativacao=ativacao, parametro=nome):
                self.assertLess(erro, 1e-6)

    def test_gradientes_leaky_relu(self):
        self.verificar("leaky_relu")

    def test_gradientes_tanh(self):
        self.verificar("tanh")

    def test_gradientes_tem_o_formato_dos_parametros(self):
        X, y = dados_sinteticos(m=12, n=39)
        rede = RedeNeural([39, 32, 16, 1])
        rede.forward(X)
        grads = rede.gradientes(y)
        self.assertEqual(set(grads), set(rede.model_dict))
        for nome, gradiente in grads.items():
            with self.subTest(parametro=nome):
                self.assertEqual(gradiente.shape, rede.model_dict[nome].shape)
                self.assertTrue(np.isfinite(gradiente).all())

    def test_atualizacao_e_w_menos_taxa_vezes_gradiente(self):
        X, y = dados_sinteticos(m=20, n=5)
        rede = RedeNeural([5, 4, 1], semente=3)
        antes = {k: v.copy() for k, v in rede.model_dict.items()}
        rede.forward(X)
        grads = rede.backpropagation(y, taxa_aprendizado=0.1)
        for nome in antes:
            with self.subTest(parametro=nome):
                np.testing.assert_allclose(
                    rede.model_dict[nome], antes[nome] - 0.1 * grads[nome]
                )

    def test_gradiente_aponta_para_descida(self):
        X, y = dados_sinteticos(m=100, n=5)
        rede = RedeNeural([5, 8, 1], semente=1)
        perda_antes = rede.loss(y, rede.forward(X))
        rede.backpropagation(y, taxa_aprendizado=0.01)
        perda_depois = rede.loss(y, rede.forward(X))
        self.assertLess(perda_depois, perda_antes)


class TestTreinamento(unittest.TestCase):
    def test_treinamento_modifica_parametros(self):
        X, y = dados_sinteticos()
        rede = RedeNeural([5, 8, 4, 1], semente=0)
        antes = {k: v.copy() for k, v in rede.model_dict.items()}
        rede.fit(X, y, X, y, epocas=3, tamanho_lote=32, paciencia=100,
                 verbose=False)
        for nome in antes:
            with self.subTest(parametro=nome):
                self.assertFalse(np.allclose(antes[nome], rede.model_dict[nome]))

    def test_reprodutivel_com_mesma_semente(self):
        X, y = dados_sinteticos()
        resultados = []
        for _ in range(2):
            rede = RedeNeural([5, 8, 4, 1], semente=123)
            historico = rede.fit(X, y, X, y, epocas=5, tamanho_lote=16,
                                 verbose=False)
            resultados.append((rede.model_dict, historico))
        (p1, h1), (p2, h2) = resultados
        self.assertEqual(h1, h2)
        for nome in p1:
            np.testing.assert_array_equal(p1[nome], p2[nome])

    def test_sementes_diferentes_dao_pesos_diferentes(self):
        a = RedeNeural([5, 8, 1], semente=1).model_dict["W1"]
        b = RedeNeural([5, 8, 1], semente=2).model_dict["W1"]
        self.assertFalse(np.allclose(a, b))

    def test_lote_de_um_exemplo_e_amostra_reduzida(self):
        X, y = dados_sinteticos(m=30)
        rede = RedeNeural([5, 4, 1], semente=0)
        historico = rede.fit(X, y, X, y, epocas=5, tamanho_lote=1,
                             taxa_aprendizado=0.01, verbose=False)
        self.assertEqual(len(historico["perda_treino"]), 5)
        self.assertTrue(np.isfinite(historico["perda_treino"]).all())

    def test_lote_maior_que_a_amostra(self):
        X, y = dados_sinteticos(m=10)
        rede = RedeNeural([5, 4, 1], semente=0)
        historico = rede.fit(X, y, X, y, epocas=3, tamanho_lote=1000,
                             verbose=False)
        self.assertEqual(len(historico["epoca"]), 3)

    def test_rede_aprende_problema_separavel(self):
        X, y = dados_sinteticos(m=400, n=5, semente=4)
        rede = RedeNeural([5, 16, 8, 1], semente=0)
        historico = rede.fit(X[:300], y[:300], X[300:], y[300:], epocas=200,
                             taxa_aprendizado=0.1, tamanho_lote=32,
                             paciencia=30, verbose=False)
        self.assertLess(historico["perda_treino"][-1],
                        historico["perda_treino"][0])
        acuracia = (rede.prever(X[300:]) == y[300:]).mean()
        self.assertGreater(acuracia, 0.9)

    def test_rede_aprende_xor_nao_linear(self):
        # XOR não é separável por uma reta: exige a camada oculta.
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 0])
        rede = RedeNeural([2, 8, 1], semente=0)
        rede.fit(X, y, X, y, epocas=3000, taxa_aprendizado=0.5,
                 tamanho_lote=4, paciencia=3000, verbose=False)
        np.testing.assert_array_equal(rede.prever(X), y)

    def test_parada_antecipada_restaura_melhores_pesos(self):
        X, y = dados_sinteticos(m=60, n=5, semente=5)
        Xv, yv = dados_sinteticos(m=60, n=5, semente=6)
        rede = RedeNeural([5, 32, 1], semente=0)
        historico = rede.fit(X, y, Xv, yv, epocas=300, taxa_aprendizado=0.3,
                             tamanho_lote=8, paciencia=5, verbose=False)
        melhor = min(historico["perda_validacao"])
        self.assertAlmostEqual(rede.loss(yv, rede.forward(Xv)), melhor,
                               places=10)
        self.assertEqual(
            historico["perda_validacao"][rede.melhor_epoca - 1], melhor
        )

    def test_salvar_e_carregar(self):
        # Pesos treinados (diferentes dos iniciais), ativação e semente fora
        # do padrão: se o arquivo não fosse lido, o teste falharia.
        X, y = dados_sinteticos()
        rede = RedeNeural([5, 8, 4, 1], ativacao="tanh", alfa=0.02, semente=7)
        rede.fit(X, y, X, y, epocas=2, tamanho_lote=16, verbose=False)
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "modelo.npz"
            rede.salvar(caminho)
            carregada = RedeNeural.carregar(caminho)
        self.assertEqual(
            (carregada.camadas, carregada.ativacao, carregada.alfa,
             carregada.semente),
            ([5, 8, 4, 1], "tanh", 0.02, 7),
        )
        for nome, valor in rede.model_dict.items():
            np.testing.assert_array_equal(carregada.model_dict[nome], valor)
        np.testing.assert_allclose(carregada.prever_proba(X),
                                   rede.prever_proba(X))


class TestRedeReferencia(unittest.TestCase):
    """Rede base da playlist (tanh + softmax), passo 1 do enunciado."""

    def test_formatos_e_vieses(self):
        X, y = dados_sinteticos(m=30, n=4)
        modelo = NnModel(X, y, hidden_neurons=10, output_neurons=2)
        formatos = {k: v.shape for k, v in modelo.model_dict.items()}
        self.assertEqual(formatos, {"W1": (4, 10), "b1": (1, 10),
                                    "W2": (10, 2), "b2": (1, 2)})
        self.assertTrue(np.all(modelo.model_dict["b1"] == 0))

    def test_softmax_soma_um_por_linha(self):
        X, y = dados_sinteticos(m=30, n=4)
        probs = NnModel(X, y, 6, 3).forward(X)
        self.assertEqual(probs.shape, (30, 3))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)
        self.assertTrue(np.all((probs > 0) & (probs < 1)))

    def test_softmax_estavel_com_valores_grandes(self):
        X, y = dados_sinteticos(m=10, n=3)
        modelo = NnModel(X, y)
        modelo.W2 *= 1000  # z2 da ordem de centenas: exp(z2) estouraria
        modelo.model_dict["W2"] = modelo.W2
        probs = modelo.forward(X * 100)
        self.assertTrue(np.isfinite(probs).all())
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)

    def test_perda_com_probabilidades_iguais_vale_log2(self):
        X, y = dados_sinteticos(m=10, n=3)
        modelo = NnModel(X, y)
        self.assertAlmostEqual(modelo.loss(np.full((10, 2), 0.5)), np.log(2))

    def test_gradientes_conferem_com_diferencas_finitas(self):
        X, y = dados_sinteticos(m=15, n=3, semente=8)
        modelo = NnModel(X, y, hidden_neurons=4, output_neurons=2, seed=3)
        modelo.B1 += 0.1  # vieses diferentes de zero
        analitico = modelo.gradients(modelo.forward(X))
        m, h = len(y), 1e-6
        for nome, parametro in modelo.model_dict.items():
            numerico = np.zeros_like(parametro)
            for indice in np.ndindex(parametro.shape):
                original = parametro[indice]
                parametro[indice] = original + h
                mais = modelo.loss(modelo.forward(X))
                parametro[indice] = original - h
                menos = modelo.loss(modelo.forward(X))
                parametro[indice] = original
                numerico[indice] = (mais - menos) / (2 * h)
            # A perda é a média, mas os gradientes do vídeo são da soma: × m.
            erro = np.linalg.norm(analitico[nome] - m * numerico) / (
                np.linalg.norm(analitico[nome]) + np.linalg.norm(m * numerico)
            )
            with self.subTest(parametro=nome):
                self.assertLess(erro, 1e-6)

    def test_gradientes_tem_o_formato_dos_parametros(self):
        X, y = dados_sinteticos(m=15, n=3)
        modelo = NnModel(X, y, hidden_neurons=10, output_neurons=2)
        grads = modelo.gradients(modelo.forward(X))
        for nome, gradiente in grads.items():
            with self.subTest(parametro=nome):
                self.assertEqual(gradiente.shape, modelo.model_dict[nome].shape)

    def test_backpropagation_atualiza_model_dict(self):
        X, y = dados_sinteticos(m=30, n=4)
        modelo = NnModel(X, y)
        antes = modelo.model_dict["W1"].copy()
        modelo.backpropagation(modelo.forward(X), learning_rate=0.01)
        self.assertFalse(np.allclose(antes, modelo.model_dict["W1"]))
        self.assertIs(modelo.model_dict["W1"], modelo.W1)

    def test_aprende_make_moons(self):
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=500, noise=0.1, random_state=0)
        modelo = NnModel(X, y, hidden_neurons=10, output_neurons=2)
        previsao = modelo.fit(epochs=1000, lr=0.001, verbose=False)
        self.assertEqual(previsao.shape, (500,))
        self.assertGreater((modelo.forward(X).argmax(axis=1) == y).mean(), 0.95)


class TestMetricas(unittest.TestCase):
    def test_confere_com_scikit_learn(self):
        try:
            from sklearn import metrics
        except ImportError:
            self.skipTest("scikit-learn não instalado")
        g = np.random.default_rng(0)
        y = g.integers(0, 2, 500)
        y_pred = np.where(g.random(500) < 0.8, y, 1 - y)
        r = avaliar(y, y_pred)
        self.assertAlmostEqual(r["acuracia"], metrics.accuracy_score(y, y_pred))
        self.assertAlmostEqual(r["acuracia_balanceada"],
                               metrics.balanced_accuracy_score(y, y_pred))
        self.assertAlmostEqual(r["f1_macro"],
                               metrics.f1_score(y, y_pred, average="macro"))
        np.testing.assert_array_equal(r["matriz_confusao"],
                                      metrics.confusion_matrix(y, y_pred))
        for c in (0, 1):
            self.assertAlmostEqual(r["por_classe"][str(c)]["precisao"],
                                   metrics.precision_score(y, y_pred, pos_label=c))
            self.assertAlmostEqual(r["por_classe"][str(c)]["recall"],
                                   metrics.recall_score(y, y_pred, pos_label=c))

    def test_referencia_classe_majoritaria(self):
        y = np.array([1] * 8 + [0] * 2)
        r = avaliar(y, np.ones_like(y))
        self.assertAlmostEqual(r["acuracia"], 0.8)
        self.assertAlmostEqual(r["acuracia_balanceada"], 0.5)
        self.assertEqual(r["por_classe"]["0"]["recall"], 0.0)

    def test_escolher_limiar(self):
        y = np.array([0, 0, 1, 1])
        proba = np.array([0.1, 0.3, 0.35, 0.9])
        limiar, valor = escolher_limiar(y, proba)
        self.assertTrue(0.3 < limiar <= 0.35)
        self.assertEqual(valor, 1.0)


class TestDadosPreparados(unittest.TestCase):
    """Executa somente se os CSVs preparados existirem."""

    @classmethod
    def setUpClass(cls):
        pasta = RAIZ / "dados" / "dados_modelo"
        if not (pasta / "treino.csv").exists():
            raise unittest.SkipTest("dados preparados ausentes")
        import json
        import pandas as pd
        cls.config = json.loads(
            (pasta / "config_preprocessamento.json").read_text(encoding="utf-8")
        )
        cls.dados = {n: pd.read_csv(pasta / f"{n}.csv")
                     for n in ("treino", "validacao", "teste")}
        cls.ids = pd.read_csv(pasta / "ids_divisao.csv")

    def test_39_entradas_iguais_ao_config(self):
        for df in self.dados.values():
            colunas = [c for c in df.columns if c != "satisfeito"]
            self.assertEqual(colunas, self.config["entradas"])
            self.assertEqual(len(colunas), 39)

    def test_entradas_finitas(self):
        for df in self.dados.values():
            self.assertTrue(np.isfinite(df.to_numpy(float)).all())

    def test_clientes_nao_se_repetem_entre_conjuntos(self):
        grupos = self.ids.groupby("customer_unique_id")["conjunto"].nunique()
        self.assertEqual(int((grupos > 1).sum()), 0)

    def test_ids_alinhados_aos_csvs(self):
        # Linha a linha: o alvo de cada order_id na base deve ser o do CSV.
        import pandas as pd
        alvo = pd.read_csv(
            RAIZ / "dados" / "base_modelo_inicial.csv",
            usecols=["order_id", "satisfeito"],
        ).set_index("order_id")["satisfeito"]
        for nome, df in self.dados.items():
            pedidos = self.ids.loc[self.ids["conjunto"] == nome, "order_id"]
            self.assertEqual(len(pedidos), len(df))
            np.testing.assert_array_equal(
                alvo.loc[pedidos].to_numpy(), df["satisfeito"].to_numpy()
            )

    def test_conjuntos_compativeis_e_alvo_binario(self):
        colunas = [list(df.columns) for df in self.dados.values()]
        self.assertTrue(all(c == colunas[0] for c in colunas))
        for df in self.dados.values():
            self.assertTrue(set(df["satisfeito"].unique()) <= {0, 1})


class TestModeloSalvo(unittest.TestCase):
    """O modelo salvo reproduz as métricas registradas no treino."""

    def test_modelo_final_reproduz_metricas_registradas(self):
        import json
        import pandas as pd
        modelo = RAIZ / "resultados" / "modelo_final.npz"
        registro = RAIZ / "resultados" / "metricas.json"
        teste = RAIZ / "dados" / "dados_modelo" / "teste.csv"
        if not (modelo.exists() and registro.exists() and teste.exists()):
            self.skipTest("modelo ainda não treinado")
        rede = RedeNeural.carregar(modelo)
        metricas = json.loads(registro.read_text(encoding="utf-8"))
        df = pd.read_csv(teste)
        proba = rede.prever_proba(df.drop(columns="satisfeito").to_numpy(float))
        r = avaliar(df["satisfeito"].to_numpy(),
                    (proba >= metricas["limiar"]).astype(int))
        esperado = metricas["teste_limiar_escolhido"]
        self.assertEqual(r["matriz_confusao"], esperado["matriz_confusao"])
        self.assertAlmostEqual(r["f1_macro"], esperado["f1_macro"], places=12)


if __name__ == "__main__":
    unittest.main()
