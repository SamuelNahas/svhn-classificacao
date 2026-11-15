"""Gera data/svhn.npz e data/adicional.npy, lidos por svhn_dgx.py (SVHN_DATA) e final_ensemble.py (--extra).

Uso, da raiz do repositório: python dgx/prepare_data.py
svhn.npz: x_train_full, y_train_full, x_test, y_test (uint8 NCHW, rótulos 0-9) e perm42 (randperm com semente 42).
adicional.npy: imagens de data/adicional/*.png na ordem numérica dos nomes.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import datasets

root = Path("data")
tr, te = (datasets.SVHN(root, split=s, download=True) for s in ("train", "test"))
perm42 = torch.randperm(len(tr.data), generator=torch.Generator().manual_seed(42)).numpy()
np.savez(root / "svhn.npz", x_train_full=tr.data, y_train_full=tr.labels.astype(np.int64),
         x_test=te.data, y_test=te.labels.astype(np.int64), perm42=perm42)
arquivos = sorted((root / "adicional").glob("*.png"), key=lambda f: int(f.stem))
if arquivos:
    np.save(root / "adicional.npy",
            np.stack([np.asarray(Image.open(f).convert("RGB")) for f in arquivos]).transpose(0, 3, 1, 2))
print("ok:", len(tr.data), len(te.data), len(arquivos))
