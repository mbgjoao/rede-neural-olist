
"""Métricas de classificação binária calculadas com NumPy."""

import numpy as np


def matriz_confusao(y, y_pred):
    """Linhas = classe real (0, 1); colunas = classe prevista (0, 1)."""
    y = np.asarray(y).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    matriz = np.zeros((2, 2), dtype=int)
    np.add.at(matriz, (y, y_pred), 1)
    return matriz


def _dividir(a, b):
    return float(a / b) if b else 0.0


def avaliar(y, y_pred):
    matriz = matriz_confusao(y, y_pred)
    por_classe = {}
    for c in (0, 1):
        vp = matriz[c, c]
        fp = matriz[:, c].sum() - vp
        fn = matriz[c, :].sum() - vp
        precisao = _dividir(vp, vp + fp)
        recall = _dividir(vp, vp + fn)
        f1 = _dividir(2 * precisao * recall, precisao + recall)
        por_classe[str(c)] = {
            "precisao": precisao,
            "recall": recall,
            "f1": f1,
            "suporte": int(matriz[c, :].sum()),
        }
    return {
        "acuracia": _dividir(np.trace(matriz), matriz.sum()),
        "acuracia_balanceada": (
            por_classe["0"]["recall"] + por_classe["1"]["recall"]
        ) / 2,
        "f1_macro": (por_classe["0"]["f1"] + por_classe["1"]["f1"]) / 2,
        "por_classe": por_classe,
        "matriz_confusao": matriz.tolist(),
    }


def escolher_limiar(y, proba, criterio="f1_macro"):
    """Limiar em [0.05, 0.95] que maximiza o critério (usar só na validação)."""
    candidatos = np.round(np.arange(0.05, 0.951, 0.01), 2)
    valores = [avaliar(y, (proba >= t).astype(int))[criterio] for t in candidatos]
    melhor = int(np.argmax(valores))
    return float(candidatos[melhor]), float(valores[melhor])
