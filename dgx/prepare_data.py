"""Gera data/svhn.npz e data/adicional.npy, lidos por svhn_dgx.py (SVHN_DATA) e final_ensemble.py (--extra).

Uso, da raiz do repositório: python dgx/prepare_data.py
svhn.npz: x_train_full, y_train_full, x_test, y_test (uint8 NCHW, rótulos 0-9) e perm42 (randperm com semente 42).
Para o v2 (#35) também grava x_extra, y_extra (`split="extra"`) e keep_v2: máscara das imagens do treino do v2
(90% do train em ordem de perm42[len//10:], seguidas do extra) que ficam depois da limpeza de vazamento
(dados_v2.py); o relatório vai para data/limpeza_v2.json. A validação é perm42[:len//10], como no v1.
adicional.npy: imagens de data/adicional/*.png na ordem numérica dos nomes.
"""
from pathlib import Path

import json

import numpy as np
import torch
from PIL import Image
from torchvision import datasets

import dados_v2

CONTAGENS = dict(train=73257, test=26032, extra=531131)  # publicadas pelo SVHN

root = Path("data")
tr, te, ex = (datasets.SVHN(root, split=s, download=True) for s in ("train", "test", "extra"))
for nome, ds in dict(train=tr, test=te, extra=ex).items():
    assert len(ds.data) == CONTAGENS[nome], f"{nome}: {len(ds.data)} imagens, esperado {CONTAGENS[nome]}"
perm42 = torch.randperm(len(tr.data), generator=torch.Generator().manual_seed(42)).numpy()
n_val = len(perm42) // 10
x_val, y_val = tr.data[perm42[:n_val]], tr.labels[perm42[:n_val]].astype(np.int64)
x_v2 = np.concatenate([tr.data[perm42[n_val:]], ex.data])
y_v2 = np.concatenate([tr.labels[perm42[n_val:]], ex.labels]).astype(np.int64)
keep_v2, limpeza = dados_v2.limpar(x_v2, y_v2, x_val, y_val, te.data, te.labels.astype(np.int64))
dados_v2.verificar(x_v2[keep_v2], y_v2[keep_v2], x_val, y_val, te.data, te.labels.astype(np.int64))
limpeza.update(treino_v2_antes=len(x_v2), treino_v2_depois=int(keep_v2.sum()), validacao=n_val,
               classes_treino_v2=np.bincount(y_v2[keep_v2], minlength=10).tolist())
(root / "limpeza_v2.json").write_text(json.dumps(limpeza, indent=2))
print("limpeza v2:", limpeza)
np.savez(root / "svhn.npz", x_train_full=tr.data, y_train_full=tr.labels.astype(np.int64),
         x_test=te.data, y_test=te.labels.astype(np.int64), perm42=perm42,
         x_extra=ex.data, y_extra=ex.labels.astype(np.int64), keep_v2=keep_v2)
arquivos = sorted((root / "adicional").glob("*.png"), key=lambda f: int(f.stem))
if arquivos:
    np.save(root / "adicional.npy",
            np.stack([np.asarray(Image.open(f).convert("RGB")) for f in arquivos]).transpose(0, 3, 1, 2))
print("ok:", len(tr.data), len(te.data), len(ex.data), len(arquivos))
