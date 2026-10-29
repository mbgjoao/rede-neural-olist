
from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
PASTA = RAIZ / "dados" / "olist"
SAIDA = RAIZ / "dados" / "avaliacoes_tratadas.csv"

arquivo = PASTA / "olist_order_reviews_dataset.csv"
df = pd.read_csv(arquivo)

# Converter as datas para permitir ordenação cronológica.
for coluna in ["review_creation_date", "review_answer_timestamp"]:
    df[coluna] = pd.to_datetime(
        df[coluna], format="ISO8601", errors="coerce"
    )

total_original = len(df)
pedidos_original = df["order_id"].nunique()

# Verificar se existem avaliações sem nota ou sem pedido.
sem_nota = df["review_score"].isna().sum()
sem_pedido = df["order_id"].isna().sum()

# Não é possível definir o rótulo sem uma nota válida
# nem associar a avaliação a um pedido sem order_id.
df = df.dropna(subset=["order_id", "review_score"]).copy()
pedidos_repetidos = int(df["order_id"].value_counts().gt(1).sum())
linhas_antes_deduplicacao = len(df)
# Pedidos repetidos cujas avaliações discordam quanto à classe: para eles,
# o rótulo depende da regra de deduplicação.
discordantes = int(
    df.assign(s=df["review_score"].isin([4, 5]))
    .groupby("order_id")["s"].nunique().gt(1).sum()
)

# Ordenar da mais antiga para a mais recente.
# Na ausência das datas, os registros ficam antes dos datados.
# A ordenação do pandas por várias colunas é estável, e nos dados atuais não
# há empate exato entre as duas avaliações mais recentes de um pedido; por
# isso a escolha abaixo é determinística.
df = df.sort_values(
    ["review_creation_date", "review_answer_timestamp"],
    na_position="first",
)

# Manter a última avaliação de cada pedido.
df = df.drop_duplicates(subset="order_id", keep="last").copy()

df["satisfeito"] = (
    df["review_score"].isin([4, 5]).astype(int)
)

# Salvar somente as colunas necessárias nesta etapa.
df = df[
    [
        "order_id",
        "review_id",
        "review_score",
        "satisfeito",
        "review_creation_date",
        "review_answer_timestamp",
    ]
]

SAIDA.parent.mkdir(parents=True, exist_ok=True)
# Formato de data explícito: sem ele, o pandas grava cada bloco de linhas
# com um formato próprio ("2017-12-20" em uns, "2017-12-20 00:00:00" em
# outros), e a releitura com pd.to_datetime falha.
df.to_csv(SAIDA, index=False, date_format="%Y-%m-%d %H:%M:%S")

print("=== RESULTADO DA PREPARAÇÃO ===")
print(f"Linhas originais: {total_original:,}")
print(f"Pedidos distintos originais: {pedidos_original:,}")
print(f"Linhas removidas por falta de nota: {sem_nota:,}")
print(f"Linhas sem order_id: {sem_pedido:,}")
print(f"Pedidos com mais de uma avaliação: {pedidos_repetidos:,}")
print(f"  com avaliações que discordam na classe: {discordantes:,}")
print(
    "Linhas removidas na deduplicação: "
    f"{linhas_antes_deduplicacao - len(df):,}"
)
print(f"Avaliações após deduplicação: {len(df):,}")
print("\nDistribuição da variável satisfeito:")
print(df["satisfeito"].value_counts().sort_index())
print("\nPercentual por classe:")
print((df["satisfeito"].value_counts(normalize=True).sort_index() * 100).round(2))
print(f"\nArquivo salvo em: {SAIDA}")
print("\nPreparação concluída.")
