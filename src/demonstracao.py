
"""Demonstração rápida para a apresentação (poucos segundos).

Mostra o projeto funcionando sem treinar tudo de novo: a rede final é
carregada de resultados/modelo_final.npz. As quatro partes seguem a ordem
dos slides:
  1. Passo 1: a rede da playlist (NnModel) aprende o make_moons;
  2. a retropropagação confere com as diferenças finitas;
  3. Passo 3: a rede final classifica pedidos reais do teste;
  4. as métricas no teste, comparadas com a classe majoritária.

Executar a partir da pasta do projeto:
    python src/demonstracao.py
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from rede_neural import RedeNeural  # noqa: E402
from rede_referencia import NnModel  # noqa: E402
from metricas import avaliar  # noqa: E402

DADOS = RAIZ / "dados"
MODELO = DADOS / "dados_modelo"
RESULTADOS = RAIZ / "resultados"
ALVO = "satisfeito"


def titulo(texto):
    print("\n" + "=" * 72)
    print(texto)
    print("=" * 72)


def parte_1_rede_da_playlist():
    from sklearn.datasets import make_moons

    titulo("1) PASSO 1 - Rede da playlist (NnModel): tanh + softmax no make_moons")
    X, y = make_moons(n_samples=500, noise=0.1, random_state=0)
    modelo = NnModel(X, y, hidden_neurons=10, output_neurons=2)
    modelo.fit(epochs=1000, lr=0.001)
    acuracia = (modelo.forward(X).argmax(axis=1) == y).mean()
    print(f"Acurácia final: {acuracia:.3f}  (2 entradas -> 10 tanh -> 2 softmax)")


def parte_2_gradientes(X, y):
    titulo("2) Retropropagação x diferenças finitas (rede 39-16-1, 20 pedidos)")
    rede = RedeNeural([X.shape[1], 16, 1], semente=0)
    rede.forward(X)
    analitico = rede.gradientes(y)
    h = 1e-6
    erros = []
    for nome, parametro in rede.model_dict.items():
        # (L(p + h) - L(p - h)) / 2h para cada peso, um de cada vez.
        numerico = np.zeros_like(parametro)
        for indice in np.ndindex(parametro.shape):
            original = parametro[indice]
            parametro[indice] = original + h
            mais = rede.loss(y, rede.forward(X))
            parametro[indice] = original - h
            menos = rede.loss(y, rede.forward(X))
            parametro[indice] = original
            numerico[indice] = (mais - menos) / (2 * h)
        erro = np.linalg.norm(analitico[nome] - numerico) / (
            np.linalg.norm(analitico[nome]) + np.linalg.norm(numerico)
        )
        erros.append(erro)
        print(f"{nome:3s} formato {str(parametro.shape):9s} erro relativo {erro:.1e}")
    if max(erros) < 1e-6:
        print("Todos abaixo de 1e-06: os gradientes da retropropagação estão certos.")
    else:
        print("ATENÇÃO: erro acima de 1e-06; confira a retropropagação.")


def parte_3_previsoes(rede, limiar):
    titulo("3) PASSO 3 - Rede final (Leaky ReLU + sigmoide) em pedidos reais do teste")
    n_parametros = sum(v.size for v in rede.model_dict.values())
    print(f"Camadas {rede.camadas} · ativação {rede.ativacao} · "
          f"{n_parametros} parâmetros · limiar {limiar}\n")

    teste = pd.read_csv(MODELO / "teste.csv")
    ids = pd.read_csv(MODELO / "ids_divisao.csv")
    ids = ids[ids["conjunto"] == "teste"].reset_index(drop=True)
    base = pd.read_csv(DADOS / "base_modelo_inicial.csv").set_index("order_id")

    # Os 3 primeiros pedidos de cada classe no teste, sem escolher a dedo.
    linhas = (list(np.flatnonzero(teste[ALVO] == 1)[:3])
              + list(np.flatnonzero(teste[ALVO] == 0)[:3]))
    X = teste.drop(columns=ALVO).to_numpy(float)
    proba = rede.prever_proba(X[linhas])

    print(f"{'entrega':>8s} {'atraso':>7s} {'preço':>10s} {'UF':>3s}"
          f" {'P(satisf.)':>11s} {'previsto':>9s} {'real':>5s}")
    for linha, p in zip(linhas, proba):
        bruto = base.loc[ids.loc[linha, "order_id"]]
        print(f"{bruto['tempo_entrega_dias']:7.1f}d {bruto['atraso_dias']:6.1f}d"
              f" R$ {bruto['preco_total']:7.2f} {bruto['customer_state']:>3s}"
              f" {p:11.3f} {int(p >= limiar):9d} {int(teste.loc[linha, ALVO]):5d}")
    return X, teste[ALVO].to_numpy(int)


def parte_4_metricas(rede, limiar, X, y, majoritaria):
    titulo(f"4) Avaliação no teste ({len(y):,} pedidos de clientes nunca vistos)"
           .replace(",", "."))
    resultados = [
        (f"Rede neural (limiar {limiar})",
         avaliar(y, (rede.prever_proba(X) >= limiar).astype(int))),
        ("Sempre 'satisfeito'", avaliar(y, np.full_like(y, majoritaria))),
    ]
    print(f"{'':28s}{'acurácia':>10s}{'acur. bal.':>12s}{'F1 macro':>10s}")
    for nome, r in resultados:
        print(f"{nome:28s}{r['acuracia']:10.4f}"
              f"{r['acuracia_balanceada']:12.4f}{r['f1_macro']:10.4f}")

    rede_r = resultados[0][1]
    m = rede_r["matriz_confusao"]
    print("\nMatriz de confusão       previsto 0   previsto 1")
    print(f"real 0 (insatisfeito)  {m[0][0]:11d}  {m[0][1]:11d}")
    print(f"real 1 (satisfeito)    {m[1][0]:11d}  {m[1][1]:11d}")
    classe_0 = rede_r["por_classe"]["0"]
    print(f"\nInsatisfeitos encontrados: {classe_0['recall']:.0%} (recall da classe 0)."
          f" Quando aponta insatisfação, acerta {classe_0['precisao']:.0%}.")


def main():
    arquivo_modelo = RESULTADOS / "modelo_final.npz"
    if not arquivo_modelo.exists():
        raise SystemExit("Modelo não encontrado. Execute antes: python src/treinar.py")
    rede = RedeNeural.carregar(arquivo_modelo)
    limiar = json.loads(
        (RESULTADOS / "metricas.json").read_text(encoding="utf-8")
    )["limiar"]
    treino = pd.read_csv(MODELO / "treino.csv")

    parte_1_rede_da_playlist()
    parte_2_gradientes(treino.drop(columns=ALVO).to_numpy(float)[:20],
                       treino[ALVO].to_numpy(int)[:20])
    X_teste, y_teste = parte_3_previsoes(rede, limiar)
    parte_4_metricas(rede, limiar, X_teste, y_teste,
                     int(treino[ALVO].mode().iloc[0]))

    print(f"\nGráfico da perda: {RESULTADOS / 'grafico_perda.svg'}")
    print("Para abrir no navegador (PowerShell): start resultados\\grafico_perda.svg")


if __name__ == "__main__":
    main()
