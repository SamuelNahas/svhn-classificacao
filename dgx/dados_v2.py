"""Limpeza de vazamento dos dados do v2 (#35). Numpy puro: imagens uint8 (N, 3, 32, 32).

limpar() devolve a máscara das imagens de treino que ficam e um relatório; o treino é `train` + `extra`.
Idêntica = mesmo SHA-256. Quase-duplicata = dHash de 64 bits a até LIMIAR bits (fixado no plano do v2, Adendos 1 e 2).
"""
import hashlib

import numpy as np

LIMIAR = 1  # Adendo 2 do plano do v2 (era 4: removia 4% do treino, quase tudo falso positivo)
_LIM = np.linspace(0, 64, LIMIAR + 2).astype(int)
CORTES = [(int(a), int(b - a)) for a, b in zip(_LIM, _LIM[1:])]  # LIMIAR+1 pedaços: algum é igual se a distância <= LIMIAR


def _sha(x):
    return [hashlib.sha256(img.tobytes()).digest() for img in x]


def dhash(x):
    """Cinza, 8 linhas × 9 colunas por média de blocos; 1 bit por par de colunas vizinhas (esquerda < direita)."""
    g = x.astype(np.float32).mean(1).reshape(len(x), 8, 4, 32).mean(2)
    g = np.stack([c.mean(-1) for c in np.array_split(g, 9, axis=-1)], -1)
    bits = (g[:, :, 1:] > g[:, :, :-1]).reshape(len(x), 64)
    return np.packbits(bits, axis=1).view(">u8")[:, 0].astype(np.uint64)


def _pedaco(h, ini, larg):
    return (h >> np.uint64(ini)) & np.uint64((1 << larg) - 1)


def _proximos(h_tr, y_tr, h_ref, y_ref):
    """Por imagem de treino: há referência a <= LIMIAR bits com o mesmo rótulo? E só com rótulo diferente?"""
    mesmo, dif = np.zeros(len(h_tr), bool), np.zeros(len(h_tr), bool)
    achados = []
    for ini, larg in CORTES:
        chave = _pedaco(h_ref, ini, larg)
        ordem = np.argsort(chave, kind="stable")
        ch = chave[ordem]
        k = _pedaco(h_tr, ini, larg)
        achados.append((ordem, np.searchsorted(ch, k, "left"), np.searchsorted(ch, k, "right")))
    for i in range(len(h_tr)):
        cand = np.unique(np.concatenate([o[lo[i]:hi[i]] for o, lo, hi in achados]))
        if not len(cand):
            continue
        vizinhas = cand[np.bitwise_count(h_ref[cand] ^ h_tr[i]) <= LIMIAR]
        mesmo[i] = bool((y_ref[vizinhas] == y_tr[i]).any())
        dif[i] = bool(len(vizinhas)) and not mesmo[i]
    return mesmo, dif


def _vazamentos(x_tr, y_tr, x_val, y_val, x_te, y_te):
    ref = set(_sha(x_val)) | set(_sha(x_te))
    identica = np.array([h in ref for h in _sha(x_tr)])
    x_ref, y_ref = np.concatenate([x_val, x_te]), np.concatenate([y_val, y_te])
    mesmo, dif = _proximos(dhash(x_tr), y_tr, dhash(x_ref), y_ref)
    return identica, mesmo & ~identica, dif & ~identica


def limpar(x_tr, y_tr, x_val, y_val, x_te, y_te):
    """(máscara das imagens de treino que ficam, relatório). Validação e teste não mudam."""
    identica, mesmo, dif = _vazamentos(x_tr, y_tr, x_val, y_val, x_te, y_te)
    comuns = len(set(_sha(x_val)) & set(_sha(x_te)))
    val_te, _ = _proximos(dhash(x_val), y_val, dhash(x_te), y_te)
    return ~(identica | mesmo), dict(identicas=int(identica.sum()), quase_mesmo_rotulo=int(mesmo.sum()),
                                     quase_outro_rotulo=int(dif.sum()), validacao_teste_identicas=comuns,
                                     validacao_teste_quase_mesmo_rotulo=int(val_te.sum()) - comuns)


def verificar(x_tr, y_tr, x_val, y_val, x_te, y_te):
    """Falha se sobrou Vazamento no treino (idêntica ou quase-duplicata de mesmo rótulo)."""
    identica, mesmo, _ = _vazamentos(x_tr, y_tr, x_val, y_val, x_te, y_te)
    if identica.any() or mesmo.any():
        raise ValueError(f"vazamento: {int(identica.sum())} idênticas, {int(mesmo.sum())} quase-duplicatas")
