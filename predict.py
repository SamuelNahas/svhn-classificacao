"""Predições do conjunto adicional com a solução final: 3 WRN-28-2 + RandAugment, TTA shift12.

Uso (da raiz do repositório, com data/train_32x32.mat e data/adicional/*.png):
  python predict.py [--out SAMUEL_WILKER_BERNARDO_RESULTS.txt] [--test]
Carrega s{42,47,48}.pt (de results/dgx/final_r5b/membros/ ou da raiz), tira a média do softmax dos membros em 9 vistas
(imagem + deslocamentos de ±1 e ±2 px) e grava o argmax, uma classe por linha. Com --test, também
imprime a acurácia no split="test" (só informativa).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import datasets

sys.path.insert(0, str(Path(__file__).parent / "dgx"))
import svhn_dgx as S  # noqa: E402

def _peso(semente):
    """results/dgx/final_r5b/membros/s<semente>.pt, ou s<semente>.pt na raiz (pesos anexados soltos)."""
    p = Path("results/dgx/final_r5b/membros") / f"s{semente}.pt"
    return p if p.exists() else Path(f"s{semente}.pt")


MEMBROS = [_peso(s) for s in (42, 47, 48)]


def normalizacao(root):
    """Média e desvio por canal do treino da divisão 42 (90% do split="train"), como no treino."""
    ds = datasets.SVHN(root, split="train", download=True)
    x = torch.from_numpy(ds.data)  # N,3,32,32 uint8
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(S.SEED))
    std, mean = torch.std_mean(x[perm[len(x) // 10:]].float().div(255), dim=(0, 2, 3))
    return mean.view(1, 3, 1, 1).to(S.DEVICE), std.view(1, 3, 1, 1).to(S.DEVICE)


@torch.no_grad()
def probs(modelos, x, mean, std):
    out = []
    for xb in x.split(512):
        xb = S.preprocess(xb.to(S.DEVICE), mean, std)
        views = [xb] + [S.shift(xb, dx, dy) for dx, dy in S.TTAS["shift12"]]
        ps = [m(v.contiguous(memory_format=torch.channels_last)).float().softmax(1) for m in modelos for v in views]
        out.append(torch.stack(ps).mean(0).cpu())
    return torch.cat(out)


def main(out, test, root="data"):
    modelos = []
    for p in MEMBROS:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = S.build(ck["config"])
        m.load_state_dict(ck["state_dict"])
        modelos.append(m.eval())
    mean, std = normalizacao(root)
    arquivos = sorted(Path(root, "adicional").glob("*.png"), key=lambda f: int(f.stem))
    x = torch.from_numpy(np.stack([np.asarray(Image.open(f).convert("RGB")) for f in arquivos])).permute(0, 3, 1, 2)
    pred = probs(modelos, x, mean, std).argmax(1).numpy()
    np.savetxt(out, pred, fmt="%d")
    print(f"{len(pred)} predições em {out}")
    if test:
        ds = datasets.SVHN(root, split="test", download=True)
        acc = (probs(modelos, torch.from_numpy(ds.data), mean, std).argmax(1).numpy() == ds.labels).mean()
        print(f"acurácia no teste: {acc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="SAMUEL_WILKER_BERNARDO_RESULTS.txt")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    main(a.out, a.test)
