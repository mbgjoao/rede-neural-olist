
from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
PASTA = RAIZ / "dados" / "olist"

reviews = pd.read_csv(PASTA / "olist_order_reviews_dataset.csv")

print("Total de avaliações:", len(reviews))
print("IDs de pedidos distintos:", reviews["order_id"].nunique())
print("IDs de avaliação distintos:", reviews["review_id"].nunique())

duplicadas = reviews[
    reviews["order_id"].duplicated(keep=False)
].sort_values(["order_id", "review_answer_timestamp"])

print("\nQuantidade de pedidos com múltiplas avaliações:")
print(reviews.groupby("order_id").size().gt(1).sum())

print("\nPrimeiros exemplos de pedidos com avaliações repetidas:")
colunas = [
    "order_id",
    "review_id",
    "review_score",
    "review_creation_date",
    "review_answer_timestamp",
]
print(duplicadas[colunas].head(20).to_string(index=False))

print("\nDistribuição das notas nas avaliações repetidas:")
print(duplicadas["review_score"].value_counts().sort_index())

print("\nAnálise concluída.")
