"""Confere o pipeline do v2 em dgx/svhn_dgx.py (#36) com um svhn.npz sintético pequeno.

Uso: python tests/check_pipeline_v2.py
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

N_TRAIN, N_EXTRA, N_TEST = 40, 10, 6
N_VAL = N_TRAIN // 10
TMP = Path(tempfile.mkdtemp())
RNG = np.random.default_rng(0)
# validação = perm42[:N_VAL]; treino do v2 = perm42[N_VAL:] seguido do extra, filtrado por keep_v2
PERM = torch.randperm(N_TRAIN, generator=torch.Generator().manual_seed(42)).numpy()
KEEP = np.ones(N_TRAIN - N_VAL + N_EXTRA, bool)
KEEP[[2, 5, N_TRAIN - N_VAL + 3]] = False
D = dict(x_train_full=RNG.integers(0, 256, (N_TRAIN, 3, 32, 32), dtype=np.uint8),
         y_train_full=RNG.integers(0, 10, N_TRAIN), x_test=RNG.integers(0, 256, (N_TEST, 3, 32, 32), dtype=np.uint8),
         y_test=RNG.integers(0, 10, N_TEST), perm42=PERM, x_extra=RNG.integers(0, 256, (N_EXTRA, 3, 32, 32), dtype=np.uint8),
         y_extra=RNG.integers(0, 10, N_EXTRA), keep_v2=KEEP)
np.savez(TMP / "svhn.npz", **D)
os.environ["SVHN_DATA"] = str(TMP / "svhn.npz")
sys.path.insert(0, str(Path(__file__).parent.parent / "dgx"))
import svhn_dgx as S  # noqa: E402


def carrega(**cfg):
    x_tr, y_tr, x_v, y_v, *_ = S.load_data(S.full(cfg))
    return x_tr.cpu().numpy(), y_tr.cpu().numpy(), x_v.cpu().numpy(), y_v.cpu().numpy()


def teste_v2_train_traz_so_o_train_que_a_limpeza_mantem_e_a_validacao_do_v1():
    x_tr, y_tr, x_v, y_v = carrega(v2="train")
    n_tr = N_TRAIN - N_VAL
    esperado = np.where(KEEP[:n_tr])[0]
    assert np.array_equal(x_tr, D["x_train_full"][PERM[N_VAL:]][esperado])
    assert np.array_equal(y_tr, D["y_train_full"][PERM[N_VAL:]][esperado])
    assert np.array_equal(x_v, D["x_train_full"][PERM[:N_VAL]]) and np.array_equal(y_v, D["y_train_full"][PERM[:N_VAL]])


def teste_v2_extra_soma_o_extra_limpo_ao_treino_e_nao_mexe_na_validacao():
    x_tr, y_tr, x_v, _ = carrega(v2="extra")
    todas = np.concatenate([D["x_train_full"][PERM[N_VAL:]], D["x_extra"]])[KEEP]
    assert np.array_equal(x_tr, todas) and len(y_tr) == KEEP.sum()
    assert np.array_equal(x_v, D["x_train_full"][PERM[:N_VAL]])


def teste_train_com_v2_nao_calcula_nem_grava_o_teste():
    if not torch.cuda.is_available():
        print("     (pulado: `train` precisa de CUDA)")
        return
    S.TRAIN_EVAL_N = "8"
    os.environ["TRAIN_EVAL_N"] = "8"
    base = dict(arch_args=dict(widths=(4, 8, 8), convs=1), epochs=1, batch=8, cutout=False, augment=False)
    for v2, tem_teste in [("train", False), (None, True)]:
        run = TMP / f"run_{v2}"
        S.train(S.full({**base, "v2": v2}), run)
        assert (run / "test_probs.npy").exists() == tem_teste, (v2, sorted(p.name for p in run.iterdir()))


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
