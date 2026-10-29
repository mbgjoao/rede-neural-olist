
from pathlib import Path
import pandas as pd

# Raiz do projeto: pasta acima de src
RAIZ = Path(__file__).resolve().parent.parent
PASTA_DADOS = RAIZ / "dados" / "olist"

arquivos = {
    "pedidos": "olist_orders_dataset.csv",
    "avaliacoes": "olist_order_reviews_dataset.csv",
    "itens": "olist_order_items_dataset.csv",
    "pagamentos": "olist_order_payments_dataset.csv",
    "clientes": "olist_customers_dataset.csv",
    "produtos": "olist_products_dataset.csv",
    "vendedores": "olist_sellers_dataset.csv",
}

print("=" * 65)
print("INSPEÇÃO INICIAL DO DATASET OLIST")
print("=" * 65)
print(f"Pasta dos dados: {PASTA_DADOS}")

dados = {}

for nome, arquivo in arquivos.items():
    caminho = PASTA_DADOS / arquivo

    print(f"\n{'-' * 65}")
    print(f"ARQUIVO: {arquivo}")

    if not caminho.exists():
        print("NÃO ENCONTRADO")
        continue

    try:
        df = pd.read_csv(caminho)
        dados[nome] = df

        print(f"Linhas: {len(df):,}")
        print(f"Colunas: {len(df.columns)}")
        print("Nomes das colunas:")
        print(list(df.columns))

        print("\nTipos de dados:")
        print(df.dtypes.to_string())

        print("\nValores ausentes por coluna:")
        ausentes = df.isna().sum()
        ausentes = ausentes[ausentes > 0]
        print(ausentes.to_string() if not ausentes.empty else "Nenhum")

        print("\nPrimeiras 3 linhas:")
        print(df.head(3).to_string(index=False))

    except Exception as erro:
        print(f"ERRO AO LER: {erro}")

# Análise específica das avaliações
if "avaliacoes" in dados:
    reviews = dados["avaliacoes"]

    print("\n" + "=" * 65)
    print("ANÁLISE DAS AVALIAÇÕES")
    print("=" * 65)

    if "review_score" in reviews.columns:
        print("\nDistribuição das notas:")
        print(reviews["review_score"].value_counts(dropna=False).sort_index())

        print("\nPercentual por nota:")
        print(
            (reviews["review_score"].value_counts(
                normalize=True, dropna=False
            ).sort_index() * 100).round(2)
        )

        notas_validas = reviews["review_score"].dropna()
        if not notas_validas.empty:
            satisfeitos = notas_validas.isin([4, 5]).sum()
            insatisfeitos = notas_validas.isin([1, 2, 3]).sum()
            total = satisfeitos + insatisfeitos

            print(f"\nNotas 4 ou 5: {satisfeitos:,}")
            print(f"Notas 1, 2 ou 3: {insatisfeitos:,}")

            if total:
                print(
                    f"Percentual satisfeito: "
                    f"{satisfeitos / total * 100:.2f}%"
                )
                print(
                    f"Percentual não satisfeito: "
                    f"{insatisfeitos / total * 100:.2f}%"
                )

    if "order_id" in reviews.columns:
        print(
            "\nLinhas com order_id repetido nas avaliações:",
            int(reviews["order_id"].duplicated(keep=False).sum())
        )
        print(
            "Pedidos distintos avaliados:",
            reviews["order_id"].nunique()
        )

# Análise dos pedidos
if "pedidos" in dados:
    orders = dados["pedidos"]

    print("\n" + "=" * 65)
    print("ANÁLISE DOS PEDIDOS")
    print("=" * 65)

    if "order_status" in orders.columns:
        print("\nStatus dos pedidos:")
        print(orders["order_status"].value_counts(dropna=False).to_string())

    if "order_delivered_customer_date" in orders.columns:
        entregues = orders["order_delivered_customer_date"].notna().sum()
        print(f"\nPedidos com data de entrega preenchida: {entregues:,}")
        print(
            "Pedidos sem data de entrega:",
            int(orders["order_delivered_customer_date"].isna().sum())
        )

print("\n" + "=" * 65)
print("INSPEÇÃO CONCLUÍDA")
print("=" * 65)
