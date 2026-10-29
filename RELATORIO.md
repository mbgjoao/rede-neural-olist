# Relatório: rede neural do zero aplicada à satisfação de clientes da Olist

Este relatório separa quatro tipos de informação:

- **[Verificado]**: fato conferido nos dados, no código ou nas fontes;
- **[Decisão]**: escolha de projeto, com a justificativa;
- **[Interpretação]**: leitura dos resultados que não foi testada diretamente;
- **[Limitação]**: algo que não pôde ser verificado ou que restringe as conclusões.

**Origem dos números.** As métricas do modelo vêm de `resultados/metricas.json`
(gerado por `src/treinar.py`). Os números da preparação vêm das saídas de
`preparar_avaliacoes.py`, `diagnosticar_juncoes.py`, `construir_base.py`
(`dados/resumo_base.json`) e `preparar_dados_modelo.py`. Os números da
auditoria de vazamento vêm de `src/diagnosticar_vazamento.py`
(`resultados/diagnostico_vazamento.json`) e de `src/auditar_dados.py`
(`resultados/auditoria_dados.json`). Os números da versão anterior do projeto,
citados para comparação, estão guardados em
`dados/dados_modelo_v2_com_respostas_antes_da_entrega/`.

| Passo do enunciado | Onde está |
|---|---|
| 1. Estrutura da playlist de referência | `src/rede_referencia.py` (NnModel, §4) |
| 2. Dados de origem brasileira | Olist (§1–§3) |
| 3. Ativação alterada e arquitetura adaptada | `src/rede_neural.py` (§5–§10) |

---

## 1. Problema e dados

**Pergunta:** dado um pedido já entregue, o cliente ficou satisfeito com ele?

**[Verificado] Origem brasileira.** A descrição oficial do *Brazilian
E-Commerce Public Dataset by Olist* (Kaggle, `olistbr/brazilian-ecommerce`)
informa que se trata de uma base pública de pedidos feitos na Olist Store:
cerca de 100 mil pedidos de 2016 a 2018, em vários marketplaces do Brasil.
A mesma página apresenta a Olist como uma empresa que conecta pequenos
negócios de todo o Brasil a canais de venda. Os valores estão em reais, e
há clientes nos 27 estados; as 27 siglas viram colunas de entrada da rede.
A base final reúne compras de 03/10/2016 a 29/08/2018.

**[Verificado] Variável-alvo `satisfeito`:**

| `review_score` original | `satisfeito` | Pedidos na base final |
|---|---|---|
| 1, 2 ou 3 | 0 (não satisfeito) | 16.139 (17,70%) |
| 4 ou 5 | 1 (satisfeito) | 75.031 (82,30%) |

**[Decisão] Delimitação: satisfação depois da entrega.** Várias
características dependem da entrega: tempo de entrega e atraso. Por isso o
modelo **classifica a satisfação depois da entrega**. Ele **não** é uma
previsão feita no momento da compra. Para que o rótulo meça de fato a
satisfação com o pedido entregue, ficam de fora os pedidos cuja avaliação
foi respondida **antes** de o produto chegar (§3.2, problema 4).

## 2. Características usadas (39 entradas)

Há uma linha por pedido. As 12 características numéricas são calculadas em
`src/construir_base.py`:

| Característica | Significado |
|---|---|
| `tempo_entrega_dias` | entrega ao cliente − compra |
| `prazo_prometido_dias` | data estimada − compra |
| `atraso_dias` | dias inteiros entre a data estimada e o dia da entrega, com mínimo 0 (entregar no dia prometido é 0) |
| `tempo_aprovacao_horas` | aprovação do pagamento − compra (14 valores ausentes, imputados) |
| `n_itens`, `n_produtos`, `n_vendedores` | contagens dos itens do pedido; segundo a Olist, cada item pode vir de um vendedor diferente |
| `preco_total`, `frete_total` | soma de preço e de frete dos itens (R$) |
| `valor_pago_total`, `parcelas_max`, `n_pagamentos` | pagamentos do pedido |

As outras 27 entradas são o estado do cliente (`customer_state_AC` …
`customer_state_TO`) em codificação *one-hot*: cada estado vira uma coluna
com valor 0 ou 1.

**[Verificado]** As 39 colunas dos CSVs são exatamente as listadas em
`config_preprocessamento.json`. Nenhum identificador (`order_id`,
`customer_id`, `customer_unique_id`, `review_id`) é usado como entrada, nem
a nota, o comentário ou as datas da avaliação.

**[Verificado] Relação com o alvo** (correlação de Pearson,
`diagnosticar_vazamento.py`): as mais fortes são `n_vendedores` (−0,131),
`n_itens` (−0,130), `tempo_entrega_dias` (−0,127) e `n_produtos` (−0,119).
`atraso_dias` fica em −0,054: na base final só 1.907 pedidos (2%) chegaram
depois da data prometida. Entre eles, 64% estão satisfeitos, contra 83% nos
demais. Todas as correlações são fracas.

## 3. Preparação e divisão dos dados

### 3.1 Caminho dos dados

| Etapa | Registros | Script que imprime o número |
|---|---|---|
| Avaliações originais | 99.224 (98.673 pedidos distintos) | `inspecionar_olist.py` |
| Após a deduplicação (1 avaliação por pedido) | 98.673 | `preparar_avaliacoes.py` |
| Associadas a pedidos com status `delivered` | 95.832 | `diagnosticar_juncoes.py` |
| Com as três datas preenchidas (8 sem data de entrega) | 95.824 | `construir_base.py` |
| Com itens, pagamento e estado (1 sem pagamento) | 95.823 | `construir_base.py` |
| **Sem as avaliações respondidas antes da entrega (4.653)** | **91.170** | `construir_base.py` |

Itens e pagamentos são agregados por pedido **antes** das junções
(`groupby("order_id")`), e todas as junções usam `validate`
(`one_to_one` ou `many_to_one`), que interrompe o script se houver
duplicação inesperada.

**[Verificado] Reprodutibilidade.** Executar de novo a cadeia de scripts
gera os mesmos arquivos. Um revisor independente refez a preparação inteira
a partir de `dados/olist/` e obteve os 7 arquivos com o mesmo SHA-256.

**[Decisão] Deduplicação.** Há 547 pedidos com mais de uma avaliação, num
total de 551 linhas removidas. Para cada pedido foi mantida a avaliação
**mais recente**, ordenando por `review_creation_date` e depois por
`review_answer_timestamp`. A escolha é determinística: a ordenação do pandas
por várias colunas é estável, e não há empate exato entre as duas avaliações
mais recentes de um pedido. O `review_id` escolhido fica gravado em
`avaliacoes_tratadas.csv`.
**[Limitação]** Em 118 desses pedidos as avaliações discordam quanto à
classe, ou seja, o rótulo depende dessa regra.

### 3.2 Problemas encontrados na auditoria e correções

**Problema 1: vazamento entre treino e teste na divisão aleatória.** A
primeira versão dividia os pedidos ao acaso (`train_test_split`
estratificado; 67.075 / 14.374 / 14.374 linhas). Ocorre que o `customer_id`
da Olist muda a cada pedido, e a pessoa real é o `customer_unique_id`.
Nesta base:

- 2.465 pessoas têm mais de um pedido (5.208 pedidos). Em 81,9% delas,
  todos os pedidos têm a mesma classe;
- **369 avaliações (`review_id`) servem de rótulo para 743 pedidos.** A
  Olist reaproveita a mesma avaliação para pedidos diferentes da mesma
  pessoa (nenhuma avaliação liga duas pessoas).

Simulando a divisão aleatória por pedido nesta base, 518 clientes ficam ao
mesmo tempo no treino e no teste (536 linhas do teste), e a mesma avaliação
aparece em conjuntos diferentes 66 vezes entre treino e teste, 94 entre
treino e validação e 17 entre validação e teste.
Ou seja, a rede veria no treino a resposta de parte do teste.

**[Decisão] Correção:** divisão **agrupada por `customer_unique_id` e
estratificada pelo alvo**, com `StratifiedGroupKFold` de 20 partes: 3 para
teste, 3 para validação e 14 para treino. Todos os pedidos de uma pessoa
ficam num único conjunto, e o identificador não entra na rede. Na divisão
atual, zero clientes e zero avaliações aparecem em dois conjuntos.
Consequência: a avaliação mede clientes novos, e as proporções de classe se
mantêm.

| Conjunto | Linhas | % não satisfeito |
|---|---|---|
| Treino | 63.816 | 17,7% |
| Validação | 13.677 | 17,7% |
| Teste | 13.677 | 17,7% |

**Problema 2: escores z extremos.** As colunas de contagem não passavam
pelo recorte de valores extremos. Sem recorte, `n_pagamentos` chega a
z = 70 e `n_itens` a z = 35 no treino. Valores assim dominam a soma
ponderada e desestabilizam o gradiente.
**[Decisão] Correção:** recorte pelos percentis 1 e 99 **do treino** em
todas as 12 colunas numéricas (por exemplo, `n_itens` em [1, 4] e
`n_pagamentos` em [1, 2]). O maior |z| no treino passou a 8,5.

**Problema 3: datas gravadas em dois formatos.** Em
`avaliacoes_tratadas.csv`, parte das linhas de `review_creation_date` saía
como "2016-10-02 00:00:00" e parte como "2017-12-20", porque o pandas
formata cada bloco de linhas por conta própria; a releitura com
`pd.to_datetime` padrão falhava.
**[Decisão] Correção:** formato explícito na gravação (`date_format`) e
`format="ISO8601"` na leitura.

**Problema 4: avaliações respondidas antes da entrega.** A Olist envia a
pesquisa quando o cliente recebe o produto **ou quando vence a data
estimada** (descrição oficial no Kaggle). Em 4.653 pedidos da base o cliente
respondeu **antes** de receber: 4.048 da classe 0 e 605 da classe 1, ou seja,
20% de todos os insatisfeitos da versão anterior. Nesses casos a nota mede a
falta de entrega, não a satisfação com o pedido entregue, e as entradas
(tempo de entrega, atraso) usam uma data **posterior à própria nota**.
**[Decisão] Correção:** esses pedidos são excluídos em `construir_base.py`,
e `auditar_dados.py` confere que nenhum restou.
**[Verificado] Consequência:** era desses pedidos que vinha grande parte do
sinal. Na versão anterior, a correlação do atraso com o alvo era −0,23 e a
do tempo de entrega −0,29; agora são −0,05 e −0,13 (a definição do atraso
também mudou, problema 5). O efeito no desempenho está no §11.

**Problema 5: entrega no dia prometido contada como atraso.** A data
estimada não tem horário (é sempre 00:00). Um pedido entregue às 14h do dia
prometido contava 0,58 dia de atraso.
**[Decisão] Correção:** o atraso passou a contar os dias inteiros após a
data prometida.

### 3.3 O que foi verificado e estava correto

- Imputação pela mediana, limites de percentis, média e desvio-padrão são
  **calculados só no treino** e reaplicados à validação e ao teste. O
  `auditar_dados.py` recalcula os três CSVs a partir da base e do JSON e
  confere a igualdade.
- Não há valores ausentes ou infinitos nos CSVs, nem colunas constantes no
  treino.
- O *one-hot* dos estados usa as colunas do treino; os 27 estados aparecem
  nos três conjuntos.

**[Limitação] Divisão não temporal.** A divisão por cliente não respeita a
ordem do tempo. Uma divisão temporal (treinar em 2017 e testar em 2018)
mediria melhor o uso real do modelo.

### 3.4 Referência simples (classe majoritária)

A referência sempre prevê `satisfeito = 1`. Recalculada após as correções:
**acurácia de 0,8230 no teste**, acurácia balanceada de 0,5000 e F1 macro de
0,4514. Na versão anterior era 0,7892; ela subiu porque a maioria dos pedidos
excluídos era da classe 0.

## 4. Os materiais do professor

As duas playlists são do canal *Machine Learning para humanos*
(@MLparahumanos). **Como foram analisadas:** títulos e descrições de todos
os vídeos e, para o conteúdo, as **transcrições automáticas em português**,
lidas no próprio YouTube.

**[Limitação]** Transcrições automáticas podem conter erros, e o que aparece
apenas na tela (código, fórmulas em slides) não foi lido diretamente. Nenhuma
das descrições tem link para repositório ou notebook; a única referência
externa é o WolframAlpha, citado no vídeo #18 da playlist 1. Não procurei o
código em outros lugares.

### 4.1 Playlist 1: "Redes Neurais Artificiais" (conceitos, 20 vídeos)

Transcrições lidas por inteiro em 18 vídeos. Do **#17** só carregou um
trecho, e o **#10** abriu sem texto; para esses dois vale a descrição.

| Vídeo | Conceitos confirmados na transcrição |
|---|---|
| #01 Conceitos | Neurônio biológico e artificial; camadas de entrada, escondida (*hidden layer*) e saída; a função de ativação decide se o neurônio "dispara" |
| #02, #03 Perceptron | Um peso por entrada; viés (*bias*) constante somado; soma ponderada Σ wᵢxᵢ + b ("equação da reta") seguida da função de ativação |
| #04, #05 Ativação 1–2 | Sigmoide (função logística), σ(x) = 1/(1 + e⁻ˣ), para transformar a reta em curva em problemas não lineares |
| #06 Ativação 3 | Sigmoide e tangente hiperbólica (fórmulas); a tanh tende a ser melhor nas camadas escondidas, explicado depois pelas derivadas |
| #07 Ativação 4 | **ReLU** = max(0, x), "muito utilizada", com várias variações; **PReLU** = max(0, x) + a·min(0, x): deixa passar valores negativos em escala menor porque a ReLU zera metade dos valores; exemplo com a = 0,1, sendo a **treinado**; derivada descontínua em zero |
| #08 Ativação 5 | Softmax na última camada para várias classes; **para classificação binária, a sigmoide** |
| #09 Perda 1 | MAE (L1) e MSE, para regressão |
| #10 Perda 2 | (descrição) perda de Huber |
| #11 Perda 3 | **Entropia cruzada binária** para duas classes (0 ou 1), com −log da probabilidade prevista e o termo (1 − y); a sigmoide limita a única saída entre 0 e 1 |
| #12 Perda 4 | Entropia cruzada ligada à softmax (várias classes) |
| #13 Backpropagation | O aprendizado é achar o mínimo da perda em relação aos pesos (descida do gradiente) |
| #14 Learning rate | Animações com taxas 0,001 e 0,002: a taxa muda a velocidade e a estabilidade da convergência |
| #15 Derivada | Derivada como inclinação da reta tangente; derivada do erro em relação ao peso; η (eta) é a taxa de aprendizagem |
| #16 Regra da cadeia | Regra da cadeia com exemplo numérico |
| #17 Atualização de pesos | (trecho) regra da cadeia na derivada do erro quadrático; (descrição) previsão, perda e atualização "na mão" |
| #18 Multicamada | Atualização de pesos com função de ativação e várias camadas; a escolha da ativação influencia o aprendizado |
| #19 Ativações e derivadas | **A derivada da sigmoide vale no máximo 0,25**: em várias camadas os fatores se multiplicam, a derivada do erro fica perto de zero e a rede para de aprender; por isso a sigmoide fica geralmente na última camada, em classificação binária. A derivada da tanh tem valores "bem maiores" (máximo 1), e por isso a tanh é preferível à sigmoide nas camadas internas. **A ReLU pode ser usada nas camadas internas**, com a desvantagem de zerar muitos valores no início do treino; a PReLU é citada |
| #20 Descida do gradiente | **Em batch** (uma atualização por época, com todos os dados) e **estocástica** (uma atualização a cada observação); época = passar por todas as observações |

| Tópico pedido | Identificado? |
|---|---|
| Neurônios e camadas | Sim (#01–#03) |
| Pesos e vieses | Sim (#02–#05) |
| Propagação direta | Sim: soma ponderada mais ativação (#03–#05) |
| Funções de ativação | Sim: sigmoide, tanh, ReLU, PReLU, softmax (#04–#08) e derivadas (#19) |
| Função de perda | Sim: MAE, MSE, Huber, entropia cruzada binária e categórica (#09–#12) |
| Gradientes e retropropagação | Sim (#13–#18, #20) |
| Treinamento | Sim: taxa de aprendizagem (#14), épocas, batch e estocástica (#20) |
| Avaliação | Parcial: o #01 cita a acurácia. Não procurei sistematicamente outras métricas nesta playlist; a matriz de confusão aparece na playlist 2 (#08) |

### 4.2 Playlist 2: a referência da implementação (8 vídeos)

O enunciado chama esta playlist de "Rede neural do zero em Python", que é o
título de cada vídeo ("Rede neural do zero em Python #01" a "#08"); no
YouTube, o título da playlist é "Rede neural para classificação do zero".
Foram lidas as transcrições dos **8 vídeos** e as descrições de cada um.

| Elemento | Referência (verificado na transcrição) |
|---|---|
| Ambiente | Google Colab; NumPy para os cálculos, matplotlib e pandas para visualizar, pickle para salvar e `sklearn.datasets` para os dados (#01) |
| Organização | Classe `NnModel` com `__init__`, `forward`, `loss`, `backpropagation` e `fit`, com os pesos num dicionário `model_dict` (#01–#06) |
| Arquitetura | Entrada = nº de colunas de X → **1 camada oculta** (padrão de 10 neurônios) → **1 neurônio por classe** na saída (#02) |
| Inicialização | `W = randn(n_entrada, n_saida) / sqrt(n_entrada)`, `b = zeros((1, n))`, semente fixa (#02) |
| Propagação | `Z1 = X·W1 + b1`, `f1 = tanh(Z1)`, `Z2 = f1·W2 + b2`, **softmax** na saída (#03). No #03 também se aplicava `tanh` em Z2 antes da softmax; isso foi removido no #05 |
| Ativação oculta | **Tangente hiperbólica** (`np.tanh`) (#03) |
| Perda | Entropia cruzada: média de −log da probabilidade da classe correta (#04) |
| Retropropagação | `δ2 = softmax − 1` na classe correta; `dW2 = f1ᵀ·δ2`; `db2 = Σδ2`; `δ1 = δ2·W2ᵀ ⊙ (1 − tanh²(Z1))`; `dW1 = Xᵀ·δ1`. Na transcrição não aparece divisão pelo número de exemplos (#05) |
| Atualização | `W = W − taxa_aprendizado · dW` (descida do gradiente) (#05) |
| Treinamento | Laço de épocas fixo com todos os exemplos a cada passo; acurácia por `argmax`; impressão 10 vezes por treino; `fit` retorna as previsões; exemplo com 100 épocas, 10 neurônios e taxa 0,001 (#06) |
| Dados de exemplo | `make_moons(n_samples=500, noise=0)` (#01); dados gerados com 2 e depois 3 variáveis e 4 classes, rede com 8 ocultos e opção de gráfico durante o treino (#07); *breast cancer* do scikit-learn, com matriz de confusão (#08) |

### 4.3 Reimplementação da referência: `src/rede_referencia.py`

**[Decisão]** A classe `NnModel` reimplementa a rede descrita acima a partir
das transcrições. Os nomes do vídeo (`hidden_neurons`, `model_dict`,
`forward`, `loss`, `backpropagation`, `fit`) foram mantidos para facilitar a
comparação. Duas diferenças, ambas sem efeito na matemática: os gradientes
ficam num método `gradients` separado, que a `backpropagation` chama, para
poderem ser testados; e a softmax subtrai o maior valor de cada linha antes
do `exp`, o que não muda o resultado mas evita estouro numérico. Como a
transcrição não mostra divisão por m, os gradientes são somados nos
exemplos (inferência a partir da transcrição).

**[Verificado] Funcionamento.** No `make_moons` com 500 pontos (ruído 0,1),
10 neurônios e taxa 0,001, a acurácia é de 0,906 em 100 épocas e 0,998 em
1000 épocas (`python src/rede_referencia.py`). Oito testes automatizados
cobrem a classe.

**[Limitação]** Nomes internos, a semente e o formato das mensagens podem
diferir do código mostrado nos vídeos.

### 4.4 Comparação técnica: referência × projeto

| # | Item | Referência (playlist 2) | Projeto (`rede_neural.py`) | Situação |
|---|---|---|---|---|
| 1 | Camadas | entrada → 1 oculta (10) → 1 neurônio por classe | 39 → 16 → 1, oculta escolhida na validação | Confirmado; arquitetura adaptada (decisão) |
| 2 | Pesos e vieses | W (n_entrada, n_saída) = randn/√n; b = zeros (1, n) | mesmo formato; W = randn·√(2/n) (He); b = zeros | Confirmado; inicialização He (decisão) |
| 3 | Ativação oculta e derivada | tanh; 1 − tanh² | Leaky ReLU (a = 0,01); derivada 1 ou 0,01 | Confirmado; **alterada** (exigência) |
| 4 | Perda | entropia cruzada (média de −log p da classe correta) | entropia cruzada binária (playlist 1, #11) | Confirmado; caso binário da mesma perda |
| 5 | Gradientes | δ2 = softmax − 1 na classe correta; sem divisão por m na transcrição | δ = (p − y)/m; média do lote | Fórmulas confirmadas; a ausência de /m é inferida da transcrição |
| 6 | Retropropagação | camada a camada, de trás para frente, regra da cadeia | igual, num laço sobre L camadas | Confirmado |
| 7 | Atualização | W = W − taxa · dW | W = W − η · dW, η = 0,05 | Confirmado |
| 8 | Laço de treino | épocas fixas, todos os exemplos por passo | minilotes de 256 + parada antecipada pela validação | Confirmado; minilotes são um meio-termo entre o batch e o estocástico da playlist 1 (#20) (decisão) |
| 9 | Previsão | softmax → probabilidades; classe = `argmax` | sigmoide → P(satisfeito); limiar escolhido na validação | Confirmado; limiar (decisão) |
| 10 | Avaliação | acurácia no treino (#06); matriz de confusão (#08) | acurácia, precisão, recall e F1 por classe, F1 macro, acurácia balanceada, matriz de confusão; treino, validação e teste | Confirmado; não verifiquei se a referência separa treino e teste |

### 4.5 Rastreabilidade: requisito → evidência → implementação

| Requisito | Evidência nos materiais | Implementação | Alteração em relação à referência |
|---|---|---|---|
| 1. Estrutura da rede da playlist | P2 #01–#06: classe `NnModel`, `model_dict`, forward, loss, backpropagation, fit | `src/rede_referencia.py`; testes `TestRedeReferencia` | Reimplementada a partir das transcrições; detalhes do código da tela não verificados |
| 2. Dados de origem brasileira | Enunciado; página da Olist no Kaggle | `dados/olist/`, `src/construir_base.py` | — |
| 3a. Alterar a função de ativação | P2 #03: tanh nas ocultas; P1 #07: ReLU e PReLU; P1 #19: derivadas | `leaky_relu` e `derivada_leaky_relu` em `src/rede_neural.py`; `TestDerivadas` | **tanh → Leaky ReLU** (PReLU com a fixo) |
| 3b. Saída adequada ao problema binário | P1 #08 e #19: sigmoide na última camada para classificação binária; P1 #11: entropia cruzada binária | `sigmoide`, `loss`, `gradientes` | **softmax com 2 neurônios → sigmoide com 1** |
| 3c. Adequar a arquitetura | P2 #02 e #06: nº de entradas = colunas; neurônios ocultos ajustáveis | 39 → 16 → 1, regra de seleção em `src/treinar.py` | 1 oculta de 10 → 1 oculta de 16, comparada com 32 + 16 |
| Retropropagação e gradientes | P2 #05; P1 #13–#18 | `RedeNeural.gradientes` e `backpropagation`; verificação por diferenças finitas | Gradiente médio (divisão por m) |
| Treinamento | P2 #06 (épocas, taxa); P1 #14 (taxa), #20 (batch e estocástica) | `RedeNeural.fit` | Minilotes e parada antecipada |
| Avaliação | P2 #06 (acurácia), #08 (matriz de confusão) | `src/metricas.py`, `src/treinar.py` | Métricas por classe, F1 macro, acurácia balanceada e limiar escolhido na validação |

P1 = playlist 1 (conceitos); P2 = playlist 2 (implementação).

## 5. Arquitetura implementada

```
entrada (39) ──► oculta (16, Leaky ReLU) ──► saída (1, sigmoide) = P(satisfeito = 1)
```

**[Decisão] Por que esse formato:**

- **39 entradas:** uma por característica preparada (12 numéricas e 27
  estados).
- **1 neurônio de saída com sigmoide:** o problema tem duas classes, e um
  único número entre 0 e 1 basta para representar P(classe 1).
- **Camadas ocultas escolhidas pela validação.** Foram treinadas duas
  candidatas definidas em `config_modelo.json`, com tudo o mais igual:

| Candidata | Parâmetros | Melhor época | Perda (val.) | F1 macro (val.) | Acur. balanceada (val.) |
|---|---|---|---|---|---|
| 39-32-16-1 (proposta inicial) | 1.825 | 35 | 0,44787 | 0,5846 | 0,5858 |
| **39-16-1 (escolhida)** | **657** | 63 | 0,44805 | 0,5905 | 0,5868 |

A regra de escolha foi fixada **antes** de olhar o teste: menor perda de
validação, mas, se outra candidata estiver a menos de 0,001 da melhor, vence
a de menos parâmetros. A diferença foi de 0,0002, então ficou a rede com uma
camada oculta. A segunda camada não trouxe ganho mensurável, o que segue a
orientação de não aumentar a complexidade sem necessidade.

## 6. Funções de ativação: referência × projeto

| Posição | Referência (verificado) | Projeto |
|---|---|---|
| Camada oculta | tanh(z) = (eᶻ − e⁻ᶻ)/(eᶻ + e⁻ᶻ) | **Leaky ReLU**: f(z) = z se z > 0; 0,01·z caso contrário |
| Derivada oculta | 1 − tanh²(z) | f′(z) = 1 se z > 0; 0,01 caso contrário |
| Saída | softmax (2 neurônios) | **sigmoide** σ(z) = 1/(1 + e⁻ᶻ) (1 neurônio) |

**[Decisão] Por que Leaky ReLU, com base na playlist 1:**

1. Ela atende ao requisito de trocar a ativação: a referência usa tanh.
2. É a **PReLU do vídeo #07** com o parâmetro a **fixo** em 0,01, em vez de
   treinado: f(z) = max(0, z) + 0,01·min(0, z). O vídeo usa a = 0,1 no
   exemplo. Fixar o parâmetro mantém a retropropagação igual à das outras
   ativações e mais fácil de explicar.
3. O **vídeo #19** mostra que a derivada da sigmoide vale no máximo 0,25 e
   que, multiplicada camada a camada, faz a rede parar de aprender. A
   derivada da Leaky ReLU vale 1 para z > 0 e nunca é zero; já a derivada
   da tanh, embora chegue a 1, cai a zero quando |z| é grande (fato
   matemático, não atribuído ao vídeo).
4. Diferente da ReLU, a inclinação de 0,01 para z < 0 não zera os valores
   negativos, a desvantagem da ReLU apontada nos vídeos #07 e #19.

**[Limitação]** O nome "Leaky ReLU" não aparece nas transcrições lidas; o
que aparece é a PReLU, da qual ela é o caso de parâmetro fixo.

**[Decisão] Por que sigmoide em vez de softmax:** os vídeos #08, #11 e #19
da playlist 1 indicam a sigmoide na última camada para **classificação
binária**, com a entropia cruzada binária como perda. Com duas classes, a
softmax de dois neurônios é matematicamente igual a uma sigmoide aplicada
à diferença das duas saídas: softmax(z)₁ = e^z₁/(e^z₀ + e^z₁) = σ(z₁ − z₀).
Um neurônio só tem metade dos parâmetros na última camada e dá o mesmo
modelo. A implementação usa uma forma numericamente estável da sigmoide.

**[Decisão] Inicialização He.** A referência divide por √n_entrada. Aqui os
pesos são `N(0, 1)·√(2/n_entrada)`. O fator 2 compensa o fato de a família
ReLU "desligar" quase metade das ativações, mantendo a variância do sinal
estável entre as camadas. Os vieses começam em zero, como na referência.

**[Verificado] A troca de ativação não muda o resultado.** Ver §11.2: com
os mesmos dados, a tanh e a própria rede base da playlist ficaram empatadas
com a Leaky ReLU. A troca atende ao enunciado e é tecnicamente sólida, mas
**não deve ser apresentada como causa de melhora de desempenho**.

## 7. Função de perda

Foi usada a entropia cruzada binária média, apresentada no vídeo #11 da
playlist 1 e caso de 2 classes da entropia cruzada da referência:

  L = −(1/m) Σᵢ [ yᵢ·log(pᵢ) + (1 − yᵢ)·log(1 − pᵢ) ]

- Se y = 1, só conta −log(p): a perda é perto de 0 quando p ≈ 1 e cresce
  sem limite quando p → 0.
- p é limitado a [10⁻¹², 1 − 10⁻¹²] para evitar log(0). O recorte só atua
  com saídas muito saturadas (|z| > 27,6), o que não ocorre nas redes
  customizadas. Para a rede base com taxa 0,001, que satura, a perda de
  validação é calculada direto dos logits (`entropia_cruzada_logits`).
- **Referência para ler a perda:** prever sempre a proporção de satisfeitos
  do treino dá perda de 0,4668 na validação. A rede final chega a 0,4481.

## 8. Propagação direta (forward)

Com X de formato (m, 39), uma amostra por linha como na referência:

```
Z1 = X  · W1 + b1      W1: (39, 16)   b1: (1, 16)   Z1: (m, 16)
A1 = LeakyReLU(Z1)
Z2 = A1 · W2 + b2      W2: (16, 1)    b2: (1, 1)    Z2: (m, 1)
p  = sigmoide(Z2)      probabilidade de satisfeito = 1
```

O código (`RedeNeural.forward`) funciona para qualquer número de camadas e
guarda Z e A de cada camada, que serão usados na retropropagação. O viés
`b` de formato (1, n) é somado a todas as linhas por broadcasting.

## 9. Retropropagação e gradientes

A regra da cadeia (vídeos #16 e #18 da playlist 1) é aplicada da saída para
a entrada. Para a sigmoide com entropia cruzada binária, a derivada da perda
em relação a Z da saída fica simples:

  ∂L/∂Z_saída = ∂L/∂p · σ′(Z) = (p − y) / m

O teste `test_erro_da_saida_e_p_menos_y` confere essa simplificação
multiplicando as duas derivadas. Esse é o equivalente binário do
"softmax − 1 na classe correta" da referência. Depois, para cada camada l,
de trás para frente:

```
dW_l     = A_{l-1}ᵀ · δ_l                  (derivada de Z = A·W + b em relação a W)
db_l     = Σ_linhas δ_l                    (mantendo o formato (1, n))
δ_{l-1}  = (δ_l · W_lᵀ) ⊙ f′(Z_{l-1})      (passa o erro para trás e multiplica pela derivada da ativação)
```

**Diferença em relação à referência:** aqui o gradiente é dividido por m,
ou seja, é a média do lote. Na transcrição do vídeo #05 não aparece essa
divisão; a reimplementação da referência soma os gradientes. As duas formas
apontam na mesma direção; a soma equivale à média com uma taxa de
aprendizado m vezes maior. Por isso, na reimplementação, a taxa de 0,001 do
vídeo (usada com poucas centenas de exemplos) fica instável com 64 mil
(§11.2).

**[Verificado] Verificação numérica dos gradientes.** Para cada peso e viés,
o gradiente analítico foi comparado com diferenças finitas centrais,
(L(θ + h) − L(θ − h)) / 2h com h = 10⁻⁶. O erro relativo ficou abaixo de
10⁻⁶ para Leaky ReLU e para tanh, e também para a `NnModel`. As derivadas de
cada ativação também são conferidas separadamente. Como teste de mutação,
dois erros foram introduzidos de propósito numa cópia do código (derivada da
Leaky ReLU errada e falta da divisão por m), e o teste falhou nos dois casos.
Um revisor independente repetiu a verificação numa rede 39-16-1 com dados
reais (erro entre 5·10⁻¹¹ e 1,3·10⁻⁸).

## 10. Atualização dos pesos e treinamento

- **Atualização:** descida do gradiente, a mesma regra da referência:
  `W ← W − η·dW` e `b ← b − η·db`, com η = 0,05.
- **Minilotes** de 256 exemplos, embaralhados a cada época com a semente 42.
  O vídeo #20 da playlist 1 apresenta a descida em batch (uma atualização
  por época) e a estocástica (uma por exemplo); minilotes são o meio-termo
  entre as duas.
- **Parada antecipada:** a cada época é calculada a perda de validação; conta
  como melhora uma queda de mais de 10⁻⁶. Após 20 épocas sem melhora o treino
  para, e os pesos da **melhor época** são restaurados.
- **Limiar:** depois do treino, é escolhido o limiar entre 0,05 e 0,95 que
  maximiza o F1 macro **na validação**. O resultado foi **0,76**. Com o
  limiar 0,5 a rede quase só responde "satisfeito" (§11.1).
- **O teste não participou de nenhuma escolha:** ele só é avaliado no fim de
  `treinar.py`, depois da arquitetura, do limiar e das comparações.
  Reexecuções do script repetem essa mesma avaliação, sem ajustar nada.

**Evolução da perda** (`resultados/grafico_perda.svg`): as duas curvas caem
rápido nas primeiras 10 épocas e depois estabilizam. A perda de validação
fica em torno de 0,448, enquanto a de treino continua caindo devagar até
0,4435. A diferença final é pequena (cerca de 0,005), sem sobreajuste
relevante; a melhor época foi a 63, e o treino parou na 83. Não houve
instabilidade nem valores não finitos.

## 11. Resultados

### 11.1 Conjunto de teste (avaliação final)

Rede 39-16-1, Leaky ReLU; 13.677 pedidos de clientes que não aparecem no
treino, dos quais 2.421 são de não satisfeitos.

| Métrica | Rede (limiar 0,76) | Rede (limiar 0,5) | Classe majoritária |
|---|---|---|---|
| Acurácia | 0,7752 | 0,8251 | 0,8230 |
| Acurácia balanceada | **0,5875** | 0,5206 | 0,5000 |
| F1 macro | **0,5921** | 0,4969 | 0,4514 |
| Recall da classe 0 | 0,2970 | 0,0492 | 0 |

Por classe, com o limiar 0,76:

| Classe | Precisão | Recall | F1 | Suporte |
|---|---|---|---|---|
| 0: não satisfeito | 0,3439 | 0,2970 | 0,3187 | 2.421 |
| 1: satisfeito | 0,8531 | 0,8781 | 0,8654 | 11.256 |

**Matriz de confusão** (limiar 0,76; linhas = classe real, colunas = prevista):

|  | previsto 0 | previsto 1 |
|---|---|---|
| **real 0** | 719 | 1.702 |
| **real 1** | 1.372 | 9.884 |

**[Verificado] A rede não supera a referência em acurácia.** Com o limiar
0,5 ela empata com "sempre satisfeito" (0,8251 contra 0,8230), porque quase
nunca aponta insatisfação. Com o limiar 0,76, escolhido na validação, ela
encontra 30% dos insatisfeitos (719 de 2.421), acerta 34% das vezes em que
aponta insatisfação, e por isso perde acurácia (0,7752). As métricas que
consideram as duas classes melhoram: acurácia balanceada de 0,50 para 0,59 e
F1 macro de 0,45 para 0,59.

**[Interpretação] Por quê.** Depois de excluir as avaliações respondidas
antes da entrega, as informações do pedido (prazos, valores, quantidade de
itens e vendedores, estado) quase não se relacionam com a nota: a maior
correlação é −0,13, e a perda da rede (0,448) fica pouco abaixo da de quem
prevê sempre a proporção (0,467). Muitos clientes insatisfeitos receberam
pedidos "normais": no prazo e com valores típicos. A causa da insatisfação
provavelmente está no que a base não descreve aqui, como a qualidade do
produto ou o texto do comentário.

**Comparação com a versão anterior** (que mantinha as avaliações respondidas
antes da entrega): acurácia de 0,8101 contra referência de 0,7892,
acurácia balanceada de 0,6374 e F1 macro de 0,6577. **[Interpretação]** A
maior parte daquele ganho vinha de reconhecer reclamações sobre pedidos que
ainda não tinham chegado, um caso fora da delimitação do problema.

### 11.2 Comparação entre variantes (somente validação)

As comparações foram feitas na validação, cada uma com o limiar escolhido na
própria validação. Nenhuma delas alterou as escolhas do modelo final.

| Variante | Ativação | Otimização | Perda (val.) | F1 macro | Acur. bal. |
|---|---|---|---|---|---|
| Customizada 39-32-16-1 | Leaky ReLU + sigmoide | minilotes, parada antecipada | 0,44787 | 0,5846 | 0,5858 |
| **Final: 39-16-1** | **Leaky ReLU + sigmoide** | minilotes, parada antecipada | 0,44805 | 0,5905 | 0,5868 |
| Customizada 39-16-1 com tanh | tanh + sigmoide | minilotes, parada antecipada | 0,44853 | 0,5895 | 0,5861 |
| Rede base da playlist, 39-10-2 | tanh + softmax | lote completo, taxa 2 × 10⁻⁵, 1500 épocas | 0,44855 | 0,5883 | 0,5850 |
| Rede base com a taxa do vídeo | tanh + softmax | lote completo, taxa 0,001, 200 épocas | 61,30 | 0,4514 | 0,5000 |
| Prever sempre a proporção do treino | — | — | 0,46683 | — | — |

**[Verificado]** As quatro primeiras variantes ficam a menos de 0,001 uma da
outra na perda. **[Interpretação]** Com uma única semente por configuração,
diferenças desse tamanho provavelmente estão dentro da variação entre
sementes, que não foi medida; não há base para dizer que uma variante é
melhor. A rede base da playlist, sem customização, alcança o mesmo nível
quando a taxa é ajustada ao tamanho da base. Com a taxa do vídeo (0,001) e
gradientes somados, o treino satura: a rede passa a responder sempre a mesma
classe (acurácia balanceada 0,50) e a perda exata, calculada pelos logits,
vai a 61. Não foi comparada nenhuma técnica de fora das redes neurais.

## 12. Limitações e melhorias futuras

**Limitações:**

1. O modelo classifica a satisfação **depois da entrega** e não serve para
   prever na hora da compra.
2. Com as informações disponíveis o sinal é fraco: a rede não supera a
   referência em acurácia (§11.1).
3. As características são operacionais (prazo, valores, estado). Faltam o
   produto, a categoria, o vendedor e o texto do comentário.
4. O rótulo junta a nota 3 com as notas 1 e 2. Esse corte (≤ 3) é uma
   escolha, e outro corte mudaria o problema.
5. 118 rótulos dependem da regra de deduplicação, e 4.653 pedidos foram
   excluídos pela delimitação (§3.2).
6. A divisão não é temporal, e cada configuração foi treinada com uma única
   semente.
7. As playlists foram verificadas pelas transcrições automáticas, não pelo
   código-fonte original; do vídeo #17 da playlist 1 só carregou um trecho,
   e do #10 apenas a descrição (§4).

**Melhorias possíveis:**

- Adicionar a categoria do produto e a nota média histórica do vendedor
  (calculada só com dados anteriores, para não vazar informação);
- Usar entropia cruzada ponderada por classe para dar mais peso aos
  insatisfeitos;
- Treinar o parâmetro a da PReLU, como no vídeo #07, em vez de fixá-lo;
- Repetir o treino com várias sementes ou validação cruzada agrupada;
- Fazer uma divisão temporal e comparar com a regressão logística como
  segunda referência.

---

## Apêndice A: testes automatizados executados

`python -m unittest discover -s tests -v`: **47 testes, todos aprovados**.

| Grupo | O que verifica |
|---|---|
| Formatos | Formato de W e b em cada camada, vieses zerados, saída (m, 1), probabilidades em (0, 1), valores finitos, sigmoide estável em ±1000, rejeição de saída ≠ 1 neurônio |
| Derivadas | Leaky ReLU, tanh e sigmoide conferidas por diferenças finitas; o máximo da derivada da sigmoide é 0,25 e fica em z = 0; erro da saída igual a (p − y)/m pela regra da cadeia |
| Perda | ≈ 0 para previsões perfeitas, log 2 em p = 0,5, crescente com o erro, finita em p ∈ {0, 1}, mesmo valor para p em vetor ou coluna, erro com tamanhos diferentes |
| Gradientes | Diferenças finitas × retropropagação (Leaky ReLU e tanh, erro < 10⁻⁶); formato de cada gradiente igual ao do parâmetro; atualização exata W − η·dW; um passo de gradiente reduz a perda |
| Treinamento | Treino altera todos os parâmetros, reprodutibilidade com a mesma semente, sementes diferentes dão pesos diferentes, lote de 1 exemplo, lote maior que a amostra, aprende problema separável (> 90%), aprende XOR (não linear), parada antecipada restaura os melhores pesos, salvar e carregar uma rede treinada com ativação e semente fora do padrão |
| Rede de referência (NnModel) | Formatos dos pesos e gradientes, softmax somando 1 e estável com valores grandes, perda log 2 com probabilidades iguais, gradientes por diferenças finitas, `backpropagation` atualiza o `model_dict`, aprende o `make_moons` (> 95%) |
| Métricas | Acurácia, acurácia balanceada, F1 macro, precisão, recall e matriz de confusão iguais às do scikit-learn; referência majoritária; escolha do limiar |
| Dados | 39 entradas iguais ao config, valores finitos, nenhum cliente em dois conjuntos, ids alinhados aos CSVs linha a linha, mesmas colunas nos três conjuntos e alvo binário |
| Modelo salvo | `resultados/modelo_final.npz` reproduz a matriz de confusão e o F1 macro registrados em `metricas.json` |

`python src/auditar_dados.py`: **24 verificações, 0 falhas**
(`resultados/auditoria_dados.json`).

**Revisão independente.** Três revisores automáticos, sem permissão de
edição, revisaram separadamente a matemática, os dados e a coerência entre
documentação e código. Nenhum achado foi crítico ou alto; os achados médios
e baixos estão corrigidos (Apêndice B).

## Apêndice B: erros encontrados e corrigidos

| Onde | Problema | Correção |
|---|---|---|
| `preparar_dados_modelo.py` | Divisão aleatória por pedido com vazamento de clientes e avaliações (§3.2) | Divisão agrupada por cliente |
| `preparar_dados_modelo.py` | Contagens sem recorte de extremos (z até 70) | Recorte p1–p99 em todas as numéricas |
| `construir_base.py` | Avaliações respondidas antes da entrega dentro da base (§3.2, problema 4) | Excluídas; nova verificação na auditoria |
| `construir_base.py` | Entrega no dia prometido contada como fração de dia de atraso | Atraso em dias inteiros |
| `preparar_avaliacoes.py` | Datas gravadas em dois formatos no mesmo CSV | `date_format` explícito na gravação; `format="ISO8601"` na leitura |
| `preparar_avaliacoes.py` | Mensagem mostrava 551 "pedidos repetidos" (eram linhas removidas; os pedidos são 547); comentário atribuía o determinismo ao parâmetro `kind`, que o pandas ignora | Mensagens separadas; comentário corrigido; `review_id` gravado |
| `rede_neural.py` | `loss` com p em formato (m,) e y em (m, 1) gerava broadcasting (m, m): a perda de validação registrada da 39-16-1 saía 0,597 em vez de 0,447 | Conversão para coluna e dois testes de regressão |
| `rede_neural.py` | `carregar` deixava o arquivo `.npz` aberto | Leitura dentro de `with` |
| `rede_referencia.py` | Softmax sem subtrair o máximo: estouraria com valores acima de ~709 | Subtração do máximo da linha (resultado igual) |
| `treinar.py` | Perda da rede base saturada subestimada pelo recorte de p (6,56 em vez de cerca de 10,6 na versão anterior) | Perda calculada pelos logits |
| `treinar.py` | Comparação com tanh rodava depois da avaliação no teste; regra de seleção usava "≤" e a documentação, "menor que" | Comparações antes do teste; regra com "<" |
| `tests/` | O teste de salvar e carregar passava mesmo sem ler o arquivo; o de alinhamento só comparava contagens | Rede treinada com ativação e semente fora do padrão; comparação linha a linha |
| `demonstracao.py` | Afirmava "todos abaixo de 1e-06" sem conferir | Mensagem condicional |
| `requirements.txt` | Versões mínimas podiam gerar outra divisão dos dados | Versões exatas |
