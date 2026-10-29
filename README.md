# Rede neural do zero: satisfação de clientes do e-commerce brasileiro (Olist)

Trabalho de Sistemas Inteligentes (2º bimestre). Uma rede neural escrita
**somente com NumPy**, a partir da estrutura da playlist de referência
"Rede neural do zero em Python", classifica se um cliente da Olist ficou
satisfeito (nota 4 ou 5) com o pedido **depois que ele foi entregue**.

A explicação completa (teoria, decisões, resultados e limitações) está em
[RELATORIO.md](RELATORIO.md).

## Resultado em poucas linhas

No teste (13.677 pedidos de clientes que não aparecem no treino), a rede
39-16-1 com Leaky ReLU, com o limiar escolhido na validação (0,76), tem
**acurácia balanceada de 0,5875** e **F1 macro de 0,5921**, contra 0,5000 e
0,4514 da referência que sempre prevê "satisfeito". Em **acurácia ela não
supera essa referência** (0,7752 contra 0,8230): ela encontra 30% dos
insatisfeitos, ao custo de errar mais satisfeitos. Depois de excluir as
avaliações respondidas antes da entrega, as informações do pedido têm pouca
relação com a nota (maior correlação: −0,13). Na validação, a rede base da
playlist e a variante com tanh empataram com a rede final.

## Requisitos do enunciado → onde estão atendidos

| Requisito | Onde |
|---|---|
| 1. Estrutura da playlist de referência | `src/rede_referencia.py`: a classe `NnModel` da playlist (tanh, softmax, entropia cruzada, lote completo), testada no `make_moons` ([RELATORIO §4](RELATORIO.md#4-os-materiais-do-professor)) |
| 2. Dados de origem brasileira | Olist: pedidos feitos em marketplaces do Brasil de 2016 a 2018, com clientes dos 27 estados ([RELATORIO §1](RELATORIO.md#1-problema-e-dados)) |
| 3a. Função de ativação alterada | `src/rede_neural.py`: tanh (referência) → **Leaky ReLU** nas ocultas ([RELATORIO §6](RELATORIO.md#6-funções-de-ativação-referência--projeto)) |
| 3b. Saída adequada ao problema binário | softmax com 2 neurônios → **sigmoide com 1 neurônio**, com entropia cruzada binária ([RELATORIO §6](RELATORIO.md#6-funções-de-ativação-referência--projeto)) |
| 3c. Arquitetura adaptada ao problema | 39 entradas, 1 saída binária, camadas ocultas escolhidas pela validação ([RELATORIO §5](RELATORIO.md#5-arquitetura-implementada)) |

As duas playlists do professor foram analisadas pelas transcrições: a de
implementação (8 vídeos) e a de conceitos (20 vídeos). As escolhas de
ativação, saída e perda se apoiam nesta última, nos vídeos #07 (ReLU e
PReLU), #08 e #11 (sigmoide e entropia cruzada binária para classificação
binária) e #19 (derivadas). O RELATORIO §4 traz a comparação item a item e
a tabela de rastreabilidade.

## Estrutura

```
Rede neural/
├── config_modelo.json          hiperparâmetros, arquiteturas candidatas e comparações
├── requirements.txt            versões exatas usadas
├── README.md / RELATORIO.md
├── src/
│   ├── inspecionar_olist.py     inspeção dos CSVs originais
│   ├── analisar_duplicatas.py   pedidos com mais de uma avaliação
│   ├── preparar_avaliacoes.py   deduplicação e alvo  -> dados/avaliacoes_tratadas.csv
│   ├── diagnosticar_juncoes.py  cobertura das junções entre tabelas
│   ├── construir_base.py        características por pedido -> dados/base_modelo_inicial.csv
│   ├── preparar_dados_modelo.py divisão por cliente, imputação, recorte, padronização
│   ├── auditar_dados.py         24 verificações de consistência e vazamento
│   ├── diagnosticar_vazamento.py números da auditoria de vazamento e correlações
│   ├── rede_referencia.py       PASSO 1: a rede da playlist (NnModel)
│   ├── rede_neural.py           PASSO 3: a rede customizada (NumPy)
│   ├── metricas.py              acurácia, precisão, recall, F1, matriz de confusão
│   ├── treinar.py               treino, seleção, limiar, comparações, avaliação final
│   └── demonstracao.py          demonstração ao vivo para a apresentação (segundos)
├── tests/test_rede_neural.py    47 testes (unittest), incluindo verificação de gradientes
├── dados/
│   ├── olist/                   CSVs originais do Kaggle (não modificados)
│   ├── avaliacoes_tratadas.csv, base_modelo_inicial.csv, resumo_base.json
│   ├── dados_modelo/            treino/validacao/teste.csv, ids_divisao.csv, config_preprocessamento.json
│   ├── dados_modelo_v1_divisao_aleatoria/                cópia da 1ª versão (divisão aleatória)
│   └── dados_modelo_v2_com_respostas_antes_da_entrega/   cópia da 2ª versão, com dados e resultados
└── resultados/
    ├── modelo_final.npz         pesos e vieses treinados (+ metadados)
    ├── metricas.json            todas as métricas (validação, comparações e teste)
    ├── historico_treinamento.csv, historico_tanh.csv
    ├── grafico_perda.svg        evolução da perda (abre no navegador)
    ├── log_treinamento.txt      saída completa do último treino
    ├── auditoria_dados.json, diagnostico_vazamento.json
```

As duas pastas de cópia servem só para comparação e podem ser apagadas sem
afetar a execução.

## Como reproduzir

No PowerShell ou no terminal, a partir da pasta do projeto:

```bash
cd "C:\Rede neural"
```

Instale as dependências, caso ainda não estejam instaladas:

```bash
python -m pip install -r requirements.txt
```

Se for partir do zero, os arquivos do Kaggle devem estar em `dados/olist/`.
Execute a preparação dos dados nesta ordem:

```bash
python src/preparar_avaliacoes.py
```

```bash
python src/construir_base.py
```

```bash
python src/preparar_dados_modelo.py
```

Depois, rode a auditoria, o diagnóstico de vazamento e os testes:

```bash
python src/auditar_dados.py
```

```bash
python src/diagnosticar_vazamento.py
```

```bash
python -m unittest discover -s tests -v
```

Para ver a rede base da playlist funcionando no `make_moons`, como nos
vídeos (alguns segundos):

```bash
python src/rede_referencia.py
```

Por fim, treine e avalie. Leva cerca de 2 a 3 minutos, a maior parte na
comparação com a rede base, que usa todos os exemplos a cada época:

```bash
python src/treinar.py
```

Se o console do Windows mostrar acentos trocados, rode antes
`$env:PYTHONIOENCODING="utf-8"` no PowerShell. Isso não afeta os arquivos.

Com a mesma semente (42) e as versões do `requirements.txt`, o treino
reproduz os mesmos números, exceto os campos de tempo (`segundos`).

### Demonstração rápida (para a apresentação)

Mostra o projeto funcionando em poucos segundos, sem treinar de novo:

```bash
python src/demonstracao.py
```

Em ordem: (1) a rede da playlist aprendendo o `make_moons`; (2) os
gradientes da retropropagação conferidos por diferenças finitas numa rede
39-16-1 com pedidos reais; (3) a rede final classificando 6 pedidos do
teste, com tempo de entrega, atraso, preço e estado de cada um; (4) as
métricas no teste e a matriz de confusão, comparadas com a classe
majoritária. O gráfico da perda abre no navegador com
`start resultados\grafico_perda.svg` (PowerShell).

### Usar o modelo treinado

```python
import sys; sys.path.insert(0, "src")
import json, pandas as pd
from rede_neural import RedeNeural

rede = RedeNeural.carregar("resultados/modelo_final.npz")
limiar = json.load(open("resultados/metricas.json", encoding="utf-8"))["limiar"]
X = pd.read_csv("dados/dados_modelo/teste.csv").drop(columns="satisfeito").to_numpy()
proba = rede.prever_proba(X)          # P(satisfeito = 1)
classe = (proba >= limiar).astype(int)
```

As entradas precisam passar pelo mesmo pré-processamento descrito em
`dados/dados_modelo/config_preprocessamento.json`.
