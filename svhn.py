"""Classificação de dígitos SVHN com PyTorch.

Uso:
    python svhn.py experiments            # roda todos os experimentos (treino/validação/teste)
    python svhn.py final --model cnn      # treina o modelo escolhido no treino completo e salva os pesos
    python svhn.py predict --images DIR --out NOME1_NOME2_RESULTS.txt
"""
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

SEED = 42
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
WEIGHTS = Path("weights/best.pt")
MEAN, STD = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
NORMALIZE = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
AUGMENT = transforms.Compose([
    transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.9, 1.1)),
    NORMALIZE,
])
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def linear():
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))


def mlp(hidden=(512, 256), dropout=0.0, batchnorm=False):
    layers, size = [nn.Flatten()], 3 * 32 * 32
    for h in hidden:
        layers += [nn.Linear(size, h)]
        if batchnorm:
            layers += [nn.BatchNorm1d(h)]
        layers += [nn.ReLU(), nn.Dropout(dropout)]
        size = h
    return nn.Sequential(*layers, nn.Linear(size, 10))


def cnn(dropout=0.3):
    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
            nn.MaxPool2d(2),
        )
    return nn.Sequential(
        block(3, 32), block(32, 64), block(64, 128),
        nn.Flatten(), nn.Dropout(dropout), nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
        nn.Dropout(dropout), nn.Linear(256, 10),
    )


# Cada experimento muda uma coisa em relação ao anterior, para isolar o efeito.
EXPERIMENTS = {
    "linear":           dict(model=linear, lr=1e-3),
    "mlp_baseline":     dict(model=mlp, lr=1e-3),
    "mlp_lr_1e-4":      dict(model=mlp, lr=1e-4),
    "mlp_lr_1e-2":      dict(model=mlp, lr=1e-2),
    "mlp_batch_256":    dict(model=mlp, lr=1e-3, batch=256),
    "mlp_dropout":      dict(model=lambda: mlp(dropout=0.3), lr=1e-3),
    "mlp_bn_dropout":   dict(model=lambda: mlp(dropout=0.3, batchnorm=True), lr=1e-3),
    "mlp_deep":         dict(model=lambda: mlp((1024, 512, 256), 0.3, True), lr=1e-3),
    "cnn":              dict(model=cnn, lr=1e-3),
    "cnn_augment":      dict(model=cnn, lr=1e-3, augment=True),
}


def seed_all():
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def svhn(split, transform):
    return datasets.SVHN(DATA_DIR, split=split, download=True, transform=transform)


def loaders(batch, augment, full_train=False):
    """Split fixo 90/10 de split='train' em treino/validação (mesma semente sempre)."""
    train_tf = AUGMENT if augment else NORMALIZE
    train_full, val_full = svhn("train", train_tf), svhn("train", NORMALIZE)
    test = svhn("test", NORMALIZE)
    if full_train:
        return DataLoader(train_full, batch, shuffle=True, num_workers=2), None, DataLoader(test, 512)
    n_val = len(train_full) // 10
    idx = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(SEED)).tolist()
    train = torch.utils.data.Subset(train_full, idx[n_val:])
    val = torch.utils.data.Subset(val_full, idx[:n_val])
    return (DataLoader(train, batch, shuffle=True, num_workers=2),
            DataLoader(val, 512), DataLoader(test, 512))


def run_epoch(model, loader, loss_fn, opt=None):
    model.train(opt is not None)
    total_loss = correct = n = 0
    with torch.set_grad_enabled(opt is not None):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = loss_fn(out, y)
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def train(cfg, epochs, full_train=False):
    seed_all()
    train_dl, val_dl, test_dl = loaders(cfg.get("batch", 64), cfg.get("augment", False), full_train)
    model = cfg["model"]().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.CrossEntropyLoss()
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best_val, best_state = -1, None
    for ep in range(epochs):
        tl, ta = run_epoch(model, train_dl, loss_fn, opt)
        sched.step()
        hist["train_loss"].append(tl)
        hist["train_acc"].append(ta)
        msg = f"  ep {ep + 1:2d}/{epochs} train loss {tl:.4f} acc {ta:.4f}"
        if val_dl:
            vl, va = run_epoch(model, val_dl, loss_fn)
            hist["val_loss"].append(vl)
            hist["val_acc"].append(va)
            msg += f" | val loss {vl:.4f} acc {va:.4f}"
            if va > best_val:
                best_val, best_state = va, {k: v.clone() for k, v in model.state_dict().items()}
        print(msg, flush=True)
    if best_state:  # avalia no teste a época com melhor validação
        model.load_state_dict(best_state)
    _, test_acc = run_epoch(model, test_dl, loss_fn)
    return model, hist, best_val, test_acc


def plot(name, hist):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for key, ax in (("loss", a1), ("acc", a2)):
        ax.plot(hist[f"train_{key}"], label="treino")
        ax.plot(hist[f"val_{key}"], label="validação")
        ax.set(title=f"{name} - {'Loss' if key == 'loss' else 'Acurácia'}", xlabel="época")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{name}.png", dpi=120)
    plt.close(fig)


def cmd_experiments(args):
    RESULTS_DIR.mkdir(exist_ok=True)
    names = args.only or list(EXPERIMENTS)
    rows = []
    for i, name in enumerate(names, 1):
        print(f"[{i}/{len(names)}] {name}")
        _, hist, val_acc, test_acc = train(EXPERIMENTS[name], args.epochs)
        plot(name, hist)
        (RESULTS_DIR / f"{name}.json").write_text(json.dumps(hist))
        rows.append(dict(experimento=i, modificacao=name, train_acc=round(hist["train_acc"][-1], 4),
                         val_acc=round(val_acc, 4), test_acc=round(test_acc, 4)))
        print(f"  -> melhor val {val_acc:.4f} | teste {test_acc:.4f}")
    with open(RESULTS_DIR / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    best = max(rows, key=lambda r: r["val_acc"])
    print(f"\nMelhor na validação: {best['modificacao']} ({best['val_acc']})")


def cmd_final(args):
    model, *_, test_acc = train(EXPERIMENTS[args.model], args.epochs, full_train=True)
    WEIGHTS.parent.mkdir(exist_ok=True)
    torch.save({"model": args.model, "state_dict": model.state_dict()}, WEIGHTS)
    print(f"Modelo final '{args.model}' salvo em {WEIGHTS} | acurácia teste {test_acc:.4f}")


def natural_key(path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def cmd_predict(args):
    ckpt = torch.load(WEIGHTS, map_location=DEVICE)
    model = EXPERIMENTS[ckpt["model"]]["model"]().to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    files = sorted((p for p in Path(args.images).iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}), key=natural_key)
    preds = []
    with torch.no_grad():
        for i in range(0, len(files), 512):
            x = torch.stack([NORMALIZE(Image.open(p).convert("RGB").resize((32, 32)))
                             for p in files[i:i + 512]])
            preds += model(x.to(DEVICE)).argmax(1).tolist()
    Path(args.out).write_text("\n".join(map(str, preds)) + "\n")
    print(f"{len(preds)} predições ({files[0].name} ... {files[-1].name}) -> {args.out}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("experiments")
    e.add_argument("--epochs", type=int, default=15)
    e.add_argument("--only", nargs="*", choices=list(EXPERIMENTS))
    f = sub.add_parser("final")
    f.add_argument("--model", default="cnn_augment", choices=list(EXPERIMENTS))
    f.add_argument("--epochs", type=int, default=15)
    pr = sub.add_parser("predict")
    pr.add_argument("--images", required=True)
    pr.add_argument("--out", default="NOME1_NOME2_RESULTS.txt")
    args = p.parse_args()
    {"experiments": cmd_experiments, "final": cmd_final, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
