
"""Auditoria dos conjuntos preparados para a rede neural.

Verifica, a partir dos arquivos em disco, se os CSVs de treino, validação
e teste são consistentes com o pré-processamento documentado e se não há
vazamento entre os conjuntos. O resultado é salvo em
resultados/auditoria_dados.json.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados"
OLIST = DADOS / "olist"
MODELO = DADOS / "dados_modelo"
SAIDA = RAIZ / "resultados" / "auditoria_dados.json"

ALVO = "satisfeito"
CONJUNTOS = ["treino", "validacao", "teste"]

config = json.loads(
    (MODELO / "config_preprocessamento.json").read_text(encoding="utf-8")
)
conjuntos = {n: pd.read_csv(MODELO / f"{n}.csv") for n in CONJUNTOS}
ids = pd.read_csv(MODELO / "ids_divisao.csv")
base = pd.read_csv(DADOS / "base_modelo_inicial.csv")

resultado = {}
falhas = []


def verificar(nome, condicao, detalhe=""):
    resultado[nome] = {"ok": bool(condicao), "detalhe": detalhe}
    print(f"[{'OK' if condicao else 'FALHA'}] {nome} {detalhe}")
    if not condicao:
        falhas.append(nome)


# 1. Colunas das entradas correspondem ao arquivo de configuração.
entradas = config["entradas"]
for nome, df in conjuntos.items():
    colunas = [c for c in df.columns if c != ALVO]
    verificar(
        f"colunas_{nome}",
        colunas == entradas,
        f"{len(colunas)} entradas",
    )

# 2. Valores ausentes, infinitos, alvo binário e colunas constantes.
for nome, df in conjuntos.items():
    X = df[entradas].to_numpy(dtype=float)
    verificar(f"finitos_{nome}", np.isfinite(X).all())
    verificar(f"alvo_binario_{nome}", set(df[ALVO].unique()) <= {0, 1})

constantes = [c for c in entradas if conjuntos["treino"][c].nunique() <= 1]
verificar("sem_colunas_constantes_treino", not constantes, str(constantes))

# 3. Escala: depois do recorte por percentis, nenhum z-score absurdo.
max_z = float(
    conjuntos["treino"][config["colunas_numericas"]].abs().max().max()
)
verificar("z_maximo_treino_menor_que_10", max_z < 10, f"max |z| = {max_z:.2f}")

# 4. Estatísticas aprendidas no treino: média ~0 e desvio ~1 no treino.
num_treino = conjuntos["treino"][config["colunas_numericas"]]
verificar(
    "padronizacao_ajustada_no_treino",
    np.allclose(num_treino.mean(), 0, atol=1e-6)
    and np.allclose(num_treino.std(ddof=0), 1, atol=1e-6),
)

# 5. Os CSVs correspondem ao pré-processamento salvo (recalcula a partir
#    da base inicial e do JSON, sem reajustar nada).
base_idx = base.set_index("order_id")
for nome in CONJUNTOS:
    pedidos = ids.loc[ids["conjunto"] == nome, "order_id"]
    parte = base_idx.loc[pedidos].reset_index()
    num = parte[config["colunas_numericas"]].astype(float)
    num = num.fillna(pd.Series(config["medianas_treino"]))
    for coluna, (inf, sup) in config["limites_percentis_treino"].items():
        num[coluna] = num[coluna].clip(inf, sup)
    num = (num - pd.Series(config["medias_treino"])) / pd.Series(
        config["desvios_treino"]
    )
    estados = pd.get_dummies(
        parte["customer_state"].fillna("DESCONHECIDO").astype(str),
        prefix="customer_state",
        dtype=float,
    )
    X = pd.concat([num, estados], axis=1).reindex(
        columns=entradas, fill_value=0
    )
    igual = np.allclose(
        X.to_numpy(float), conjuntos[nome][entradas].to_numpy(float)
    ) and (parte[ALVO].to_numpy() == conjuntos[nome][ALVO].to_numpy()).all()
    verificar(f"csv_reproduz_preprocessamento_{nome}", igual)

# 6. Nenhum pedido ou cliente em mais de um conjunto.
por_conjunto = {n: ids[ids["conjunto"] == n] for n in CONJUNTOS}
for i, a in enumerate(CONJUNTOS):
    for b in CONJUNTOS[i + 1:]:
        pedidos = set(por_conjunto[a]["order_id"]) & set(
            por_conjunto[b]["order_id"]
        )
        clientes = set(por_conjunto[a]["customer_unique_id"]) & set(
            por_conjunto[b]["customer_unique_id"]
        )
        verificar(f"pedidos_disjuntos_{a}_{b}", not pedidos, str(len(pedidos)))
        verificar(
            f"clientes_disjuntos_{a}_{b}", not clientes, str(len(clientes))
        )

# 7. Avaliações (review_id) reaproveitadas pela Olist em vários pedidos
#    não podem cruzar conjuntos.
reviews = pd.read_csv(
    OLIST / "olist_order_reviews_dataset.csv",
    usecols=["review_id", "order_id"],
)
reviews = reviews.merge(ids[["order_id", "conjunto"]], on="order_id")
cruzam = int((reviews.groupby("review_id")["conjunto"].nunique() > 1).sum())
verificar("review_id_nao_cruza_conjuntos", cruzam == 0, str(cruzam))

# 8. Identificadores não estão entre as entradas.
proibidas = [
    c for c in entradas
    if "id" in c.split("_") or c in {"order_id", "customer_id", ALVO}
]
verificar("sem_identificadores_nas_entradas", not proibidas, str(proibidas))

# 9. Delimitação: nenhuma avaliação da base foi respondida antes da entrega.
respostas = pd.read_csv(
    DADOS / "avaliacoes_tratadas.csv",
    usecols=["order_id", "review_answer_timestamp"],
)
entregas = pd.read_csv(
    OLIST / "olist_orders_dataset.csv",
    usecols=["order_id", "order_delivered_customer_date"],
)
datas = base[["order_id"]].merge(respostas, on="order_id").merge(
    entregas, on="order_id"
)
antes = int(
    (pd.to_datetime(datas["review_answer_timestamp"], format="ISO8601")
     < pd.to_datetime(datas["order_delivered_customer_date"], format="ISO8601"))
    .sum()
)
verificar("sem_avaliacao_respondida_antes_da_entrega", antes == 0, str(antes))

# 10. Referência: prever sempre a classe majoritária do treino.
majoritaria = int(conjuntos["treino"][ALVO].mode().iloc[0])
resultado["referencia_classe_majoritaria"] = {
    "classe": majoritaria,
    **{
        f"acuracia_{n}": float((conjuntos[n][ALVO] == majoritaria).mean())
        for n in CONJUNTOS
    },
}
resultado["linhas"] = {n: len(df) for n, df in conjuntos.items()}
resultado["falhas"] = falhas

SAIDA.parent.mkdir(parents=True, exist_ok=True)
SAIDA.write_text(
    json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("\nReferência (classe majoritária):",
      resultado["referencia_classe_majoritaria"])
print(f"Falhas: {len(falhas)}  ->  {SAIDA}")
if falhas:
    raise SystemExit(1)
