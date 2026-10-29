
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parent.parent
ARQUIVO = RAIZ / "dados" / "base_modelo_inicial.csv"
OLIST = RAIZ / "dados" / "olist"
PASTA_SAIDA = RAIZ / "dados" / "dados_modelo"

SEMENTE = 42
ALVO = "satisfeito"
GRUPO = "customer_unique_id"

# 20 partes de ~5% cada: 3 para teste, 3 para validação e 14 para treino.
N_PARTES = 20
PARTES_TESTE = {0, 1, 2}
PARTES_VALIDACAO = {3, 4, 5}

COLUNAS_NUMERICAS = [
    "tempo_entrega_dias",
    "prazo_prometido_dias",
    "atraso_dias",
    "tempo_aprovacao_horas",
    "n_itens",
    "preco_total",
    "frete_total",
    "n_vendedores",
    "n_produtos",
    "valor_pago_total",
    "parcelas_max",
    "n_pagamentos",
]

COLUNAS_CATEGORICAS = ["customer_state"]

# Evitar valores infinitos e garantir a existência do arquivo.
if not ARQUIVO.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {ARQUIVO}")

df = pd.read_csv(ARQUIVO)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=[ALVO]).copy()
df[ALVO] = df[ALVO].astype(int)

if not set(df[ALVO].unique()).issubset({0, 1}):
    raise ValueError("A variável satisfeito deve conter somente 0 e 1.")

# O customer_id da Olist muda a cada pedido; a pessoa real é o
# customer_unique_id. Ele é usado somente para agrupar a divisão,
# nunca como entrada da rede.
pedidos = pd.read_csv(
    OLIST / "olist_orders_dataset.csv", usecols=["order_id", "customer_id"]
)
clientes = pd.read_csv(
    OLIST / "olist_customers_dataset.csv",
    usecols=["customer_id", GRUPO],
)
mapa = pedidos.merge(clientes, on="customer_id", validate="many_to_one")
df = df.merge(
    mapa[["order_id", GRUPO]], on="order_id", how="left",
    validate="one_to_one",
)
if df[GRUPO].isna().any():
    raise ValueError("Existem pedidos sem customer_unique_id.")

# Divisão agrupada por cliente e estratificada pelo alvo (~70/15/15).
# Motivo: clientes com vários pedidos tendem a repetir a nota, e a Olist
# reutiliza o mesmo review_id para pedidos diferentes do mesmo cliente.
# Numa divisão aleatória por pedido, a mesma avaliação podia aparecer no
# treino e no teste.
divisor = StratifiedGroupKFold(
    n_splits=N_PARTES, shuffle=True, random_state=SEMENTE
)
particao = np.empty(len(df), dtype=int)
for indice, (_, idx) in enumerate(
    divisor.split(df, df[ALVO], groups=df[GRUPO])
):
    particao[idx] = indice

teste = df[np.isin(particao, list(PARTES_TESTE))].copy()
validacao = df[np.isin(particao, list(PARTES_VALIDACAO))].copy()
treino = df[~np.isin(particao, list(PARTES_TESTE | PARTES_VALIDACAO))].copy()

for a, b in [(treino, validacao), (treino, teste), (validacao, teste)]:
    if set(a[GRUPO]) & set(b[GRUPO]):
        raise RuntimeError("Um cliente apareceu em mais de um conjunto.")

# A partir daqui, os parâmetros são ajustados somente no treino.
num_treino = treino[COLUNAS_NUMERICAS].apply(
    pd.to_numeric, errors="coerce"
)

# Imputação pela mediana calculada exclusivamente no treino.
medianas = num_treino.median()

if medianas.isna().any():
    raise ValueError(
        "Existe uma coluna numérica totalmente vazia no treino. "
        "Verifique os dados antes de continuar."
    )

num_treino = num_treino.fillna(medianas)

# Limitar valores extremos usando os percentis 1 e 99 do treino.
# Isso reduz a influência de valores extremos sem usar validação/teste.
# As contagens também são limitadas: sem isso, n_pagamentos chegava a
# z = 57 e n_itens a z = 35, valores que desestabilizam o gradiente.
limites = {}
for coluna in COLUNAS_NUMERICAS:
    limites[coluna] = [
        float(num_treino[coluna].quantile(0.01)),
        float(num_treino[coluna].quantile(0.99)),
    ]

for coluna, (inferior, superior) in limites.items():
    num_treino[coluna] = num_treino[coluna].clip(
        lower=inferior, upper=superior
    )

# Normalização Z-score ajustada apenas com o treino.
medias = num_treino.mean()
desvios = num_treino.std(ddof=0).replace(0, 1)

num_treino = (num_treino - medias) / desvios

# Codificar o estado como variáveis binárias.
def codificar_estado(parte):
    estado = parte[COLUNAS_CATEGORICAS].copy()
    estado["customer_state"] = (
        estado["customer_state"]
        .fillna("DESCONHECIDO")
        .astype(str)
    )
    return pd.get_dummies(
        estado, columns=COLUNAS_CATEGORICAS, dtype=float
    )

estado_treino = codificar_estado(treino).reset_index(drop=True)

X_treino = pd.concat(
    [num_treino.reset_index(drop=True), estado_treino], axis=1
)
colunas_modelo = X_treino.columns.tolist()

def transformar(parte):
    numericas = parte[COLUNAS_NUMERICAS].apply(
        pd.to_numeric, errors="coerce"
    )
    numericas = numericas.fillna(medianas)

    for coluna, (inferior, superior) in limites.items():
        numericas[coluna] = numericas[coluna].clip(
            lower=inferior, upper=superior
        )

    numericas = (numericas - medias) / desvios

    estados = codificar_estado(parte)
    X = pd.concat(
        [numericas.reset_index(drop=True),
         estados.reset_index(drop=True)],
        axis=1,
    )

    # Usar exatamente as colunas aprendidas no conjunto de treino.
    return X.reindex(columns=colunas_modelo, fill_value=0).astype(float)

X_validacao = transformar(validacao)
X_teste = transformar(teste)

# Salvar entradas e rótulos, sem IDs de pedidos entre as entradas.
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

def salvar(nome, X, parte):
    saida = X.reset_index(drop=True).copy()
    saida[ALVO] = parte[ALVO].reset_index(drop=True).astype(int)
    caminho = PASTA_SAIDA / f"{nome}.csv"
    saida.to_csv(caminho, index=False)
    print(f"{nome}.csv: {len(saida):,} linhas, {X.shape[1]} entradas")

salvar("treino", X_treino, treino)
salvar("validacao", X_validacao, validacao)
salvar("teste", X_teste, teste)

# Identificadores de cada linha, na mesma ordem dos CSVs, para auditoria.
# Este arquivo não é lido pela rede.
ids = pd.concat(
    [
        parte[["order_id", GRUPO]].reset_index(drop=True).assign(conjunto=nome)
        for nome, parte in [
            ("treino", treino), ("validacao", validacao), ("teste", teste)
        ]
    ],
    ignore_index=True,
)
ids.to_csv(PASTA_SAIDA / "ids_divisao.csv", index=False)

# Parâmetros para permitir reprodução do pré-processamento.
config = {
    "semente": SEMENTE,
    "alvo": ALVO,
    "divisao": {
        "metodo": "StratifiedGroupKFold agrupado por customer_unique_id",
        "n_partes": N_PARTES,
        "partes_teste": sorted(PARTES_TESTE),
        "partes_validacao": sorted(PARTES_VALIDACAO),
        "linhas": {
            "treino": len(treino),
            "validacao": len(validacao),
            "teste": len(teste),
        },
    },
    "entradas": colunas_modelo,
    "colunas_numericas": COLUNAS_NUMERICAS,
    "colunas_categoricas": COLUNAS_CATEGORICAS,
    "medianas_treino": {
        k: float(v) for k, v in medianas.items()
    },
    "limites_percentis_treino": limites,
    "medias_treino": {k: float(v) for k, v in medias.items()},
    "desvios_treino": {k: float(v) for k, v in desvios.items()},
}

with open(
    PASTA_SAIDA / "config_preprocessamento.json",
    "w",
    encoding="utf-8",
) as arquivo:
    json.dump(config, arquivo, ensure_ascii=False, indent=2)

# Baseline: prever sempre a classe mais frequente no treino.
classe_majoritaria = int(treino[ALVO].mode().iloc[0])
acuracia_baseline = (teste[ALVO] == classe_majoritaria).mean()
acuracia_baseline_validacao = (validacao[ALVO] == classe_majoritaria).mean()

print("\n=== RESUMO ===")
print(f"Total de registros: {len(df):,}")
print(f"Treino: {len(treino):,}")
print(f"Validação: {len(validacao):,}")
print(f"Teste: {len(teste):,}")
print(f"Entradas da rede: {len(colunas_modelo)}")
print(f"Clientes distintos: {df[GRUPO].nunique():,}")
print(f"Classe majoritária do treino: {classe_majoritaria}")
print(f"Acurácia baseline na validação: {acuracia_baseline_validacao:.4f}")
print(f"Acurácia baseline no teste: {acuracia_baseline:.4f}")

for nome, parte in [
    ("Treino", treino),
    ("Validação", validacao),
    ("Teste", teste),
]:
    print(f"\nDistribuição de classes — {nome}:")
    print(
        (parte[ALVO].value_counts(normalize=True).sort_index() * 100)
        .round(2)
        .to_string()
    )

print(f"\nArquivos salvos em: {PASTA_SAIDA}")
print("Preparação concluída.")
