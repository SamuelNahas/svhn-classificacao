"""Confere a limpeza de vazamento dos dados do v2 (#35) com imagens sintéticas.

Uso: python tests/check_dados_v2.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "dgx"))
import dados_v2 as D  # noqa: E402

RNG = np.random.default_rng(0)


def imagens(n):
    return RNG.integers(0, 256, (n, 3, 32, 32), dtype=np.uint8)


def teste_remove_identica_da_validacao_e_do_teste():
    tr, val, te = imagens(6), imagens(2), imagens(2)
    tr[1], tr[4] = val[0], te[1]
    y = np.arange(6) % 10
    keep, rel = D.limpar(tr, y, val, np.array([0, 1]), te, np.array([2, 3]))
    assert keep.tolist() == [True, False, True, True, False, True], keep
    assert rel["identicas"] == 2, rel


def perto(img, n=10):
    """Cópia com n pixels alterados em 1: dHash praticamente igual, bytes diferentes."""
    out = img.copy()
    out.flat[RNG.choice(out.size, n, replace=False)] += 1
    return out


def teste_remove_quase_duplicata_de_mesmo_rotulo():
    tr, val, te = imagens(4), imagens(2), imagens(2)
    tr[2] = perto(val[1])
    y_tr, y_val = np.array([5, 6, 7, 8]), np.array([0, 7])
    keep, rel = D.limpar(tr, y_tr, val, y_val, te, np.array([2, 3]))
    assert keep.tolist() == [True, True, False, True], keep
    assert rel["quase_mesmo_rotulo"] == 1, rel


def teste_quase_duplicata_de_outro_rotulo_fica_e_relatada():
    tr, val, te = imagens(4), imagens(2), imagens(2)
    tr[2] = perto(val[1])
    y_tr, y_val = np.array([5, 6, 3, 8]), np.array([0, 7])
    keep, rel = D.limpar(tr, y_tr, val, y_val, te, np.array([2, 3]))
    assert keep.all(), keep
    assert rel["quase_outro_rotulo"] == 1, rel


def teste_verificar_falha_com_vazamento():
    tr, val, te = imagens(4), imagens(2), imagens(2)
    tr[0] = te[1]
    try:
        D.verificar(tr, np.zeros(4, int), val, np.zeros(2, int), te, np.zeros(2, int))
    except ValueError:
        return
    raise AssertionError("verificar deveria falhar")


def teste_apos_limpar_verificar_passa_e_validacao_teste_so_e_relatada():
    tr, val, te = imagens(5), imagens(3), imagens(3)
    tr[0], tr[3], te[0] = val[2], perto(te[1]), val[0]
    y_tr, y_val, y_te = np.array([1, 2, 3, 4, 5]), np.array([0, 1, 2]), np.array([9, 4, 5])
    keep, rel = D.limpar(tr, y_tr, val, y_val, te, y_te)
    D.verificar(tr[keep], y_tr[keep], val, y_val, te, y_te)
    assert rel["validacao_teste_identicas"] == 1, rel
    assert keep.sum() == 3, keep


TESTES = [v for k, v in sorted(globals().items()) if k.startswith("teste_")]

if __name__ == "__main__":
    falhas = 0
    for t in TESTES:
        try:
            t()
            print("ok  ", t.__name__)
        except Exception as e:
            falhas += 1
            print("FALHA", t.__name__, type(e).__name__, e)
    sys.exit(1 if falhas else 0)
