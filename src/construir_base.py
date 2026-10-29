
from pathlib import Path
import json

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados"
OLIST = DADOS / "olist"

def ler(nome):
    return pd.read_csv(OLIST / nome)

# Avaliações já deduplicadas
avaliacoes = pd.read_csv(DADOS / "avaliacoes_tratadas.csv")

# Pedidos entregues com datas válidas
pedidos = ler("olist_orders_dataset.csv")
pedidos = pedidos[pedidos["order_status"] == "delivered"].copy()

datas = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "order_approved_at",
]
for coluna in datas:
    pedidos[coluna] = pd.to_datetime(
        pedidos[coluna], format="ISO8601", errors="coerce"
    )

pedidos = pedidos.dropna(
    subset=[
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
)

pedidos["tempo_entrega_dias"] = (
    pedidos["order_delivered_customer_date"]
    - pedidos["order_purchase_timestamp"]
).dt.total_seconds() / 86400

pedidos["prazo_prometido_dias"] = (
    pedidos["order_estimated_delivery_date"]
    - pedidos["order_purchase_timestamp"]
).dt.total_seconds() / 86400

# A data estimada não tem horário (é sempre 00:00): entregar em qualquer hora
# desse dia não é atraso. O atraso conta os dias inteiros após a data
# prometida, com mínimo 0 para entregas adiantadas.
pedidos["atraso_dias"] = (
    pedidos["order_delivered_customer_date"].dt.normalize()
    - pedidos["order_estimated_delivery_date"]
).dt.days.clip(lower=0)

pedidos["tempo_aprovacao_horas"] = (
    pedidos["order_approved_at"]
    - pedidos["order_purchase_timestamp"]
).dt.total_seconds() / 3600

# Agregação dos itens: uma linha por pedido
itens = ler("olist_order_items_dataset.csv")
itens_agg = itens.groupby("order_id").agg(
    n_itens=("order_item_id", "count"),
    preco_total=("price", "sum"),
    frete_total=("freight_value", "sum"),
    n_vendedores=("seller_id", "nunique"),
    n_produtos=("product_id", "nunique"),
).reset_index()

# Agregação dos pagamentos: uma linha por pedido
pagamentos = ler("olist_order_payments_dataset.csv")
pagamentos_agg = pagamentos.groupby("order_id").agg(
    valor_pago_total=("payment_value", "sum"),
    parcelas_max=("payment_installments", "max"),
    n_pagamentos=("payment_sequential", "count"),
).reset_index()

# Dados dos clientes
clientes = ler("olist_customers_dataset.csv")[
    ["customer_id", "customer_state"]
]

# Base central: avaliações + pedidos
base = avaliacoes.merge(
    pedidos,
    on="order_id",
    how="inner",
    validate="one_to_one",
)

# Agregações e dados do cliente
base = base.merge(
    itens_agg,
    on="order_id",
    how="left",
    validate="one_to_one",
)

base = base.merge(
    pagamentos_agg,
    on="order_id",
    how="left",
    validate="one_to_one",
)

base = base.merge(
    clientes,
    on="customer_id",
    how="left",
    validate="many_to_one",
)

# Identificar campos importantes ausentes
print("Linhas antes do tratamento de ausentes:", len(base))
print("\nValores ausentes:")
print(base.isna().sum()[base.isna().sum() > 0].to_string())

# Não substituir pagamentos ausentes por zero sem investigação.
# Excluir por enquanto linhas sem os dados essenciais.
essenciais = [
    "n_itens",
    "preco_total",
    "frete_total",
    "valor_pago_total",
    "parcelas_max",
    "n_pagamentos",
    "customer_state",
    "satisfeito",
]

linhas_juncao = len(base)
base = base.dropna(subset=essenciais).copy()
linhas_com_dados = len(base)

# Delimitação do problema: satisfação DEPOIS da entrega. A Olist envia a
# pesquisa quando o produto chega ou quando vence a data estimada. Se o
# cliente respondeu antes de receber, a nota não mede a satisfação com o
# pedido entregue, e as entradas (tempo de entrega, atraso) usariam uma data
# posterior à própria nota. Esses pedidos são excluídos.
resposta = pd.to_datetime(base["review_answer_timestamp"], format="ISO8601")
antes_da_entrega = resposta < base["order_delivered_customer_date"]
excluidos = base.loc[antes_da_entrega, "satisfeito"]
base = base[~antes_da_entrega].copy()

resumo = {
    "avaliacoes_tratadas": len(avaliacoes),
    "com_pedido_entregue_e_datas": linhas_juncao,
    "com_itens_pagamentos_e_estado": linhas_com_dados,
    "respondidas_antes_da_entrega": {
        "excluidas": int(antes_da_entrega.sum()),
        "classe_0": int((excluidos == 0).sum()),
        "classe_1": int((excluidos == 1).sum()),
    },
    "base_final": {
        "linhas": len(base),
        "classe_0": int((base["satisfeito"] == 0).sum()),
        "classe_1": int((base["satisfeito"] == 1).sum()),
    },
}
(DADOS / "resumo_base.json").write_text(
    json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("\nAvaliações respondidas antes da entrega (excluídas):",
      resumo["respondidas_antes_da_entrega"])

# Selecionar somente variáveis úteis para o modelo.
colunas = [
    "order_id",
    "satisfeito",
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
    "customer_state",
]

base = base[colunas]

saida = DADOS / "base_modelo_inicial.csv"
base.to_csv(saida, index=False)

print("\n=== BASE INICIAL GERADA ===")
print("Linhas:", len(base))
print("Colunas:", len(base.columns))
print("\nDistribuição da variável satisfeito:")
print(base["satisfeito"].value_counts().sort_index())
print("\nEstatísticas das variáveis numéricas:")
print(base.select_dtypes(include="number").describe().round(2).to_string())
print("\nArquivo salvo em:", saida)
