
"""Treinamento, seleção e avaliação final da rede neural.

Fluxo (o conjunto de teste só é usado no passo 5):
  1. Carrega treino, validação e teste preparados.
  2. Treina cada arquitetura candidata com Leaky ReLU e parada antecipada,
     e escolhe o limiar de classificação de cada uma na validação.
  3. Escolhe a arquitetura pela perda de validação, preferindo a mais
     simples quando a diferença é menor que a tolerância do config.
  4. Comparações, avaliadas só na validação e sem alterar nenhuma escolha:
     a. a arquitetura escolhida com tanh, a ativação da referência;
     b. a rede base da playlist (NnModel: tanh + softmax, lote completo).
  5. Avalia uma única vez no teste e compara com a classe majoritária.
  6. Salva parâmetros, histórico, gráfico e métricas em resultados/.
"""

from pathlib import Path
import csv
import json
import sys
import time

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from rede_neural import RedeNeural  # noqa: E402
from rede_referencia import NnModel  # noqa: E402
from metricas import avaliar, escolher_limiar  # noqa: E402

DADOS = RAIZ / "dados" / "dados_modelo"
RESULTADOS = RAIZ / "resultados"
ALVO = "satisfeito"
NOMES_ATIVACAO = {"leaky_relu": "Leaky ReLU", "tanh": "tanh"}


def carregar(nome):
    caminho = DADOS / f"{nome}.csv"
    if not caminho.exists():
        raise SystemExit(
            f"Arquivo não encontrado: {caminho}\n"
            "Execute antes: python src/preparar_dados_modelo.py"
        )
    df = pd.read_csv(caminho)
    return df.drop(columns=ALVO).to_numpy(float), df[ALVO].to_numpy(int)


def treinar(camadas, ativacao, config, dados):
    (X_tr, y_tr), (X_va, y_va) = dados["treino"], dados["validacao"]
    rede = RedeNeural(camadas, ativacao, config["alfa_leaky_relu"],
                      config["semente"])
    inicio = time.time()
    historico = rede.fit(
        X_tr, y_tr, X_va, y_va,
        epocas=config["epocas_maximas"],
        taxa_aprendizado=config["taxa_aprendizado"],
        tamanho_lote=config["tamanho_lote"],
        paciencia=config["paciencia"],
    )
    proba_va = rede.prever_proba(X_va)
    limiar, _ = escolher_limiar(y_va, proba_va, config["criterio_limiar"])
    resumo = {
        "camadas": camadas,
        "ativacao": ativacao,
        "melhor_epoca": rede.melhor_epoca,
        "epocas_executadas": len(historico["epoca"]),
        "segundos": round(time.time() - inicio, 1),
        "perda_validacao": RedeNeural.loss(y_va, proba_va),
        "limiar_validacao": limiar,
        "validacao_limiar_05": avaliar(y_va, (proba_va >= 0.5).astype(int)),
        "validacao_limiar_escolhido": avaliar(
            y_va, (proba_va >= limiar).astype(int)
        ),
    }
    return rede, historico, resumo


def entropia_cruzada_logits(z, y):
    """Entropia cruzada média calculada direto dos logits (log-softmax).

    É exata mesmo quando a softmax arredonda alguma probabilidade para 0,
    caso em que o recorte de p em RedeNeural.loss subestimaria a perda.
    """
    maximo = z.max(axis=1, keepdims=True)
    log_soma = maximo[:, 0] + np.log(np.exp(z - maximo).sum(axis=1))
    return float(np.mean(log_soma - z[np.arange(len(y)), y]))


def treinar_rede_base(config, dados):
    """Rede da playlist sem customização, avaliada só na validação.

    Os gradientes da NnModel são somados nos exemplos, então a taxa
    efetiva cresce com o número de linhas do treino. A primeira execução
    do config usa a taxa do vídeo (0,001) para mostrar esse efeito.
    """
    cfg = config["rede_base_playlist"]
    (X_tr, y_tr), (X_va, y_va) = dados["treino"], dados["validacao"]
    resultados = []
    for execucao in cfg["execucoes"]:
        lr, epocas = execucao["taxa_aprendizado"], execucao["epocas"]
        print(f"\n=== Rede base da playlist: {X_tr.shape[1]}-"
              f"{cfg['neuronios_ocultos']}-2, tanh + softmax, lote completo,"
              f" taxa {lr:g}, {epocas} épocas ===")
        inicio = time.time()
        modelo = NnModel(X_tr, y_tr, hidden_neurons=cfg["neuronios_ocultos"],
                         output_neurons=2, seed=config["semente"])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            modelo.fit(epochs=epocas, lr=lr)
            proba_va = modelo.forward(X_va)[:, 1]
        resumo = {
            "camadas": [X_tr.shape[1], cfg["neuronios_ocultos"], 2],
            "ativacao": "tanh + softmax",
            "taxa_aprendizado": lr,
            "epocas": epocas,
            "segundos": round(time.time() - inicio, 1),
            "finito": bool(np.isfinite(proba_va).all()),
        }
        if resumo["finito"]:
            limiar, _ = escolher_limiar(y_va, proba_va, config["criterio_limiar"])
            resumo.update({
                "perda_validacao": entropia_cruzada_logits(modelo.z2, y_va),
                "limiar_validacao": limiar,
                "validacao_limiar_05": avaliar(
                    y_va, (proba_va >= 0.5).astype(int)
                ),
                "validacao_limiar_escolhido": avaliar(
                    y_va, (proba_va >= limiar).astype(int)
                ),
            })
        resultados.append(resumo)
    return resultados


def grafico_perda_svg(historico, melhor_epoca, titulo, caminho):
    """Gráfico de linhas das perdas de treino e validação, sem matplotlib."""
    larg, alt = 760, 420
    esq, dir_, topo, base = 70, 150, 50, 60
    epocas = historico["epoca"]
    # Validação tracejada: a identidade não depende só da cor.
    series = [
        ("Treino", historico["perda_treino"], "#2a78d6", "none"),
        ("Validação", historico["perda_validacao"], "#eb6834", "6 4"),
    ]
    todos = historico["perda_treino"] + historico["perda_validacao"]
    y_min, y_max = min(todos), max(todos)
    folga = (y_max - y_min) * 0.08 or 0.01
    y_min, y_max = y_min - folga, y_max + folga
    x_max = max(epocas)

    def px(e):
        return esq + (e - 1) / max(1, x_max - 1) * (larg - esq - dir_)

    def py(v):
        return topo + (y_max - v) / (y_max - y_min) * (alt - topo - base)

    partes = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {larg} {alt}" '
        f'width="{larg}" height="{alt}" font-family="Segoe UI, Arial, sans-serif">',
        f'<rect width="{larg}" height="{alt}" fill="#fcfcfb"/>',
        f'<text x="{esq}" y="28" font-size="16" font-weight="600" '
        f'fill="#0b0b0b">{titulo}</text>',
    ]
    # Grade e eixo y (5 marcas)
    for v in np.linspace(y_min + folga, y_max - folga, 5):
        y = py(v)
        partes.append(f'<line x1="{esq}" x2="{larg - dir_}" y1="{y:.1f}" '
                      f'y2="{y:.1f}" stroke="#e4e3df" stroke-width="1"/>')
        partes.append(f'<text x="{esq - 8}" y="{y + 4:.1f}" font-size="12" '
                      f'text-anchor="end" fill="#52514e">{v:.3f}</text>')
    # Eixo x
    passo = max(1, int(np.ceil(x_max / 8 / 5) * 5))
    for e in [1] + list(range(passo, x_max + 1, passo)):
        x = px(e)
        partes.append(f'<text x="{x:.1f}" y="{alt - base + 20}" font-size="12" '
                      f'text-anchor="middle" fill="#52514e">{e}</text>')
    partes.append(f'<line x1="{esq}" x2="{larg - dir_}" y1="{alt - base}" '
                  f'y2="{alt - base}" stroke="#52514e" stroke-width="1"/>')
    partes.append(f'<text x="{(esq + larg - dir_) / 2}" y="{alt - 15}" '
                  f'font-size="13" text-anchor="middle" fill="#52514e">Época</text>')
    partes.append(f'<text x="18" y="{(topo + alt - base) / 2}" font-size="13" '
                  f'text-anchor="middle" fill="#52514e" transform="rotate(-90 18 '
                  f'{(topo + alt - base) / 2})">Entropia cruzada binária</text>')
    # Melhor época (pesos restaurados)
    xm = px(melhor_epoca)
    partes.append(f'<line x1="{xm:.1f}" x2="{xm:.1f}" y1="{topo}" '
                  f'y2="{alt - base}" stroke="#52514e" stroke-width="1" '
                  f'stroke-dasharray="2 3"/>')
    # Rótulo à esquerda da linha quando ela está perto da borda direita.
    perto_da_borda = xm > esq + 0.7 * (larg - esq - dir_)
    partes.append(f'<text x="{xm - 4 if perto_da_borda else xm + 4:.1f}" '
                  f'y="{topo + 12}" font-size="12" fill="#52514e" '
                  f'text-anchor="{"end" if perto_da_borda else "start"}">'
                  f'melhor época: {melhor_epoca} (pesos restaurados)</text>')
    # Séries com rótulo direto no fim da linha
    for nome, valores, cor, traco in series:
        pontos = " ".join(f"{px(e):.1f},{py(v):.1f}"
                          for e, v in zip(epocas, valores))
        partes.append(f'<polyline points="{pontos}" fill="none" stroke="{cor}" '
                      f'stroke-width="2" stroke-dasharray="{traco}" '
                      f'stroke-linejoin="round"><title>{nome}</title></polyline>')
        partes.append(f'<text x="{px(epocas[-1]) + 8:.1f}" '
                      f'y="{py(valores[-1]) + 4:.1f}" font-size="12" '
                      f'fill="#0b0b0b">{nome} {valores[-1]:.4f}</text>')
    # Legenda
    for i, (nome, _, cor, traco) in enumerate(series):
        y = topo + 40 + i * 20
        x = larg - dir_ + 18
        partes.append(f'<line x1="{x}" x2="{x + 22}" y1="{y}" y2="{y}" '
                      f'stroke="{cor}" stroke-width="2" stroke-dasharray="{traco}"/>')
        partes.append(f'<text x="{x + 28}" y="{y + 4}" font-size="12" '
                      f'fill="#0b0b0b">{nome}</text>')
    partes.append("</svg>")
    caminho.write_text("\n".join(partes), encoding="utf-8")


def salvar_historico_csv(historico, caminho):
    with open(caminho, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(historico.keys())
        escritor.writerows(zip(*historico.values()))


def main():
    config = json.loads((RAIZ / "config_modelo.json").read_text(encoding="utf-8"))
    dados = {n: carregar(n) for n in ("treino", "validacao", "teste")}
    n_entradas = dados["treino"][0].shape[1]
    RESULTADOS.mkdir(exist_ok=True)
    print(f"Entradas: {n_entradas}  |  treino {len(dados['treino'][1]):,}  "
          f"validação {len(dados['validacao'][1]):,}  "
          f"teste {len(dados['teste'][1]):,}")

    # 2. Arquiteturas candidatas com a ativação escolhida.
    candidatos = {}
    for nome, ocultas in config["arquiteturas_candidatas"].items():
        print(f"\n=== Candidata {nome} ({config['ativacao_oculta']}) ===")
        camadas = [n_entradas, *ocultas, 1]
        candidatos[nome] = treinar(camadas, config["ativacao_oculta"],
                                   config, dados)

    # 3. Seleção pela validação: menor perda; empate técnico -> mais simples.
    melhor_perda = min(r["perda_validacao"] for _, _, r in candidatos.values())
    elegiveis = [
        n for n, (_, _, r) in candidatos.items()
        if r["perda_validacao"] < melhor_perda + config["tolerancia_simplicidade"]
    ]
    escolhida = min(
        elegiveis,
        key=lambda n: sum(v.size for v in candidatos[n][0].model_dict.values()),
    )
    rede, historico, resumo = candidatos[escolhida]
    limiar = resumo["limiar_validacao"]
    print(f"\nArquitetura escolhida: {escolhida}  |  limiar (validação): {limiar}")

    # 4. Comparações (somente validação; não alteram nenhuma escolha).
    ativ_ref = config.get("comparar_com_ativacao_da_referencia")
    comparacao = None
    if ativ_ref:
        print(f"\n=== Comparação: {escolhida} com {ativ_ref} (referência) ===")
        _, hist_ref, comparacao = treinar(resumo["camadas"], ativ_ref,
                                          config, dados)
        salvar_historico_csv(hist_ref, RESULTADOS / f"historico_{ativ_ref}.csv")
    rede_base = (treinar_rede_base(config, dados)
                 if "rede_base_playlist" in config else None)

    # 5. Avaliação final no teste: única vez, com tudo já decidido.
    X_te, y_te = dados["teste"]
    proba_te = rede.prever_proba(X_te)
    teste = avaliar(y_te, (proba_te >= limiar).astype(int))
    teste_05 = avaliar(y_te, (proba_te >= 0.5).astype(int))
    majoritaria = int(np.bincount(dados["treino"][1]).argmax())
    referencia = avaliar(y_te, np.full_like(y_te, majoritaria))

    # 6. Salvar artefatos.
    rede.salvar(RESULTADOS / "modelo_final.npz")
    salvar_historico_csv(historico, RESULTADOS / "historico_treinamento.csv")
    grafico_perda_svg(
        historico, rede.melhor_epoca,
        f"Perda por época — rede {escolhida}, "
        f"{NOMES_ATIVACAO.get(config['ativacao_oculta'], config['ativacao_oculta'])}",
        RESULTADOS / "grafico_perda.svg",
    )
    metricas = {
        "config": config,
        "arquitetura_escolhida": escolhida,
        "n_parametros": int(sum(v.size for v in rede.model_dict.values())),
        "limiar": limiar,
        "candidatas": {n: r for n, (_, _, r) in candidatos.items()},
        "comparacao_ativacao_referencia": comparacao,
        "comparacao_rede_base_playlist": rede_base,
        "teste_limiar_escolhido": teste,
        "teste_limiar_05": teste_05,
        "teste_referencia_classe_majoritaria": referencia,
    }
    (RESULTADOS / "metricas.json").write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== VALIDAÇÃO: todas as variantes (limiar escolhido na validação) ===")
    linhas = [(f"{n} {r['ativacao']}", r) for n, (_, _, r) in candidatos.items()]
    if comparacao:
        linhas.append((f"{escolhida} {ativ_ref}", comparacao))
    for r in rede_base or []:
        rotulo = f"base playlist taxa {r['taxa_aprendizado']:g}"
        linhas.append((rotulo, r) if r["finito"] else (rotulo, None))
    for rotulo, r in linhas:
        if r is None:
            print(f"{rotulo:34s} não finito (divergiu)")
            continue
        v = r["validacao_limiar_escolhido"]
        print(f"{rotulo:34s} perda {r['perda_validacao']:.5f}  F1 macro "
              f"{v['f1_macro']:.4f}  acur. balanceada {v['acuracia_balanceada']:.4f}")

    print("\n=== TESTE (avaliação final) ===")
    for rotulo, r in [("Rede (limiar validação)", teste),
                      ("Rede (limiar 0,5)", teste_05),
                      ("Classe majoritária", referencia)]:
        print(f"{rotulo:26s} acurácia {r['acuracia']:.4f}  "
              f"acur. balanceada {r['acuracia_balanceada']:.4f}  "
              f"F1 macro {r['f1_macro']:.4f}")
    print("Matriz de confusão (linhas = real 0/1, colunas = previsto 0/1):")
    print(np.array(teste["matriz_confusao"]))
    print(f"\nResultados salvos em {RESULTADOS}")


if __name__ == "__main__":
    main()
