
from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados" / "olist"

def carregar(nome):
    caminho = DADOS / nome
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    return pd.read_csv(caminho)

avaliacoes = pd.read_csv(RAIZ / "dados" / "avaliacoes_tratadas.csv")
pedidos = carregar("olist_orders_dataset.csv")
itens = carregar("olist_order_items_dataset.csv")
pagamentos = carregar("olist_order_payments_dataset.csv")
clientes = carregar("olist_customers_dataset.csv")

print("=" * 60)
print("DIAGNÓSTICO DOS CRUZAMENTOS")
print("=" * 60)

# Chaves e duplicações
for nome, df, chave in [
    ("Avaliações tratadas", avaliacoes, "order_id"),
    ("Pedidos", pedidos, "order_id"),
    ("Itens", itens, "order_id"),
    ("Pagamentos", pagamentos, "order_id"),
    ("Clientes", clientes, "customer_id"),
]:
    print(f"\n{nome}:")
    print(f"  Linhas: {len(df):,}")
    print(f"  Chaves distintas: {df[chave].nunique():,}")
    print(f"  Chaves ausentes: {df[chave].isna().sum():,}")
    print(f"  Linhas com chave repetida: {df[chave].duplicated().sum():,}")

# Filtrar pedidos entregues
entregues = pedidos[pedidos["order_status"] == "delivered"].copy()

print("\n" + "=" * 60)
print("PEDIDOS ENTREGUES")
print("=" * 60)
print(f"Total de pedidos entregues: {len(entregues):,}")
print(
    "Entregues com data de entrega preenchida:",
    entregues["order_delivered_customer_date"].notna().sum()
)
print(
    "Entregues sem data de entrega:",
    entregues["order_delivered_customer_date"].isna().sum()
)

# Avaliações associadas a pedidos entregues
base = avaliacoes.merge(
    entregues[["order_id", "customer_id"]],
    on="order_id",
    how="inner",
    validate="one_to_one"
)

print("\nAvaliações ligadas a pedidos entregues:", f"{len(base):,}")
print(
    "Avaliações sem correspondência em pedido entregue:",
    len(avaliacoes) - len(base)
)

# Cobertura dos itens e pagamentos
ids_base = set(base["order_id"])

print("\nCobertura de itens:")
print(f"Pedidos da base com itens: {len(ids_base & set(itens['order_id'])):,}")
print(f"Pedidos da base sem itens: {len(ids_base - set(itens['order_id'])):,}")

print("\nCobertura de pagamentos:")
print(
    f"Pedidos da base com pagamentos: "
    f"{len(ids_base & set(pagamentos['order_id'])):,}"
)
print(
    f"Pedidos da base sem pagamentos: "
    f"{len(ids_base - set(pagamentos['order_id'])):,}"
)

# Datas ausentes na avaliação tratada
for coluna in ["review_creation_date", "review_answer_timestamp"]:
    if coluna in avaliacoes.columns:
        print(
            f"\nAvaliações sem {coluna}:",
            avaliacoes[coluna].isna().sum()
        )

print("\nDiagnóstico concluído.")
