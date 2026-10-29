
"""Números da auditoria de vazamento, recalculados a partir dos dados.

Reproduz os números citados no RELATORIO (§2 e §3.2):
  1. avaliações (review_id) que a Olist reaproveita em vários pedidos;
  2. o que a divisão aleatória por pedido (o procedimento anterior) faria
     nesta base: clientes e avaliações em mais de um conjunto;
  3. o mesmo para a divisão atual, agrupada por cliente (deve dar zero);
  4. a correlação de cada característica numérica com o alvo;
  5. o maior escore z das contagens se elas não fossem recortadas.
Salva em resultados/diagnostico_vazamento.json. Não altera nenhum dado.
"""

from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados"
OLIST = DADOS / "olist"
SAIDA = RAIZ / "resultados" / "diagnostico_vazamento.json"
ALVO = "satisfeito"
CONTAGENS = ["n_itens", "n_vendedores", "n_produtos", "parcelas_max",
             "n_pagamentos"]

base = pd.read_csv(DADOS / "base_modelo_inicial.csv")
avaliacoes = pd.read_csv(
    DADOS / "avaliacoes_tratadas.csv", usecols=["order_id", "review_id"]
)
pedidos = pd.read_csv(
    OLIST / "olist_orders_dataset.csv", usecols=["order_id", "customer_id"]
)
clientes = pd.read_csv(
    OLIST / "olist_customers_dataset.csv",
    usecols=["customer_id", "customer_unique_id"],
)
ids = pd.read_csv(DADOS / "dados_modelo" / "ids_divisao.csv")

chaves = (base[["order_id", ALVO]]
          .merge(avaliacoes, on="order_id", validate="one_to_one")
          .merge(pedidos, on="order_id", validate="one_to_one")
          .merge(clientes, on="customer_id", validate="many_to_one"))
resultado = {"pedidos_na_base": len(base)}

# 1. Avaliações usadas como rótulo em mais de um pedido.
por_review = chaves.groupby("review_id")["order_id"].nunique()
compartilhadas = por_review[por_review > 1].index
resultado["review_ids_compartilhados"] = {
    "avaliacoes": int(len(compartilhadas)),
    "pedidos": int(chaves["review_id"].isin(compartilhadas).sum()),
    "pessoas_diferentes_na_mesma_avaliacao": int(
        (chaves[chaves["review_id"].isin(compartilhadas)]
         .groupby("review_id")["customer_unique_id"].nunique() > 1).sum()
    ),
}
por_pessoa = chaves["customer_unique_id"].value_counts()
multi = chaves[chaves["customer_unique_id"].map(por_pessoa) > 1]
resultado["pessoas_com_varios_pedidos"] = {
    "pessoas": int((por_pessoa > 1).sum()),
    "pedidos": int(len(multi)),
    "fracao_com_uma_so_classe": round(float(
        multi.groupby("customer_unique_id")[ALVO].nunique().eq(1).mean()), 3),
}


def vazamento(conjunto):
    """Clientes e avaliações que aparecem em mais de um conjunto."""
    d = chaves.assign(conjunto=conjunto)
    pares = {}
    for a, b in [("treino", "validacao"), ("treino", "teste"),
                 ("validacao", "teste")]:
        da, db = d[d["conjunto"] == a], d[d["conjunto"] == b]
        pares[f"{a}_{b}"] = {
            "clientes": len(set(da["customer_unique_id"])
                            & set(db["customer_unique_id"])),
            "avaliacoes": len(set(da["review_id"]) & set(db["review_id"])),
        }
    teste = d[d["conjunto"] == "teste"]
    treino_clientes = set(d.loc[d["conjunto"] == "treino", "customer_unique_id"])
    return {
        "pares": pares,
        "linhas_do_teste_com_cliente_do_treino": int(
            teste["customer_unique_id"].isin(treino_clientes).sum()),
    }


# 2. Divisão aleatória por pedido, como na versão anterior do projeto.
indices = pd.Series("treino", index=chaves.index)
resto, teste = train_test_split(chaves.index, test_size=0.15, random_state=42,
                                stratify=chaves[ALVO])
_, validacao = train_test_split(resto, test_size=0.1764705882,
                                random_state=42,
                                stratify=chaves.loc[resto, ALVO])
indices[teste] = "teste"
indices[validacao] = "validacao"
resultado["divisao_aleatoria_por_pedido"] = vazamento(indices.to_numpy())

# 3. Divisão atual, agrupada por cliente.
atual = chaves[["order_id"]].merge(ids[["order_id", "conjunto"]],
                                   on="order_id", how="left")
resultado["divisao_atual_por_cliente"] = vazamento(atual["conjunto"].to_numpy())

# 4. Correlação de Pearson com o alvo.
numericas = base.drop(columns=["order_id", "customer_state"])
resultado["correlacao_com_alvo"] = (
    numericas.corr()[ALVO].drop(ALVO).round(3).sort_values().to_dict()
)

# 5. Escore z das contagens sem o recorte (estatísticas do treino).
treino = base[base["order_id"].isin(ids.loc[ids["conjunto"] == "treino",
                                            "order_id"])]
z = (treino[CONTAGENS] - treino[CONTAGENS].mean()) / treino[CONTAGENS].std(ddof=0)
resultado["z_maximo_contagens_sem_recorte"] = z.abs().max().round(1).to_dict()

SAIDA.parent.mkdir(exist_ok=True)
SAIDA.write_text(json.dumps(resultado, ensure_ascii=False, indent=2),
                 encoding="utf-8")
print(json.dumps(resultado, ensure_ascii=False, indent=2))
print(f"\nSalvo em {SAIDA}")
