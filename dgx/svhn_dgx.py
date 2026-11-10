"""Bateria de experimentos pt4 (#20) nas DGX Spark.

Treina um experimento por config JSON, com o mesmo protocolo do `svhn.ipynb`: mesma divisão treino/validação
(semente 42), mesma normalização, augment, cutout, AdamW, LR cosseno por época e escolha da melhor época pela
validação. Acrescenta Wide ResNet, EMA dos pesos, brilho/contraste, intensidade do augment, cutout com
probabilidade, precisão mista (bf16), checkpoint por época (retomada) e salva as probabilidades de validação.

O teste é calculado e salvo em arquivo (`test_probs.npy`), mas fica fora do `summary.json` e de qualquer escolha:
só é lido na avaliação final.

Uso:
  python svhn_dgx.py worker --queue Q --runs R --id ID   # treina as configs de Q/*.json ainda não concluídas
  python svhn_dgx.py run --config C.json --runs R        # treina uma config
  python svhn_dgx.py infer --run R/NOME --split val|test|extra --tta none|shift1|shift2|shift12 [--extra X.npy]
"""
import argparse
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

DATA = Path(os.environ.get("SVHN_DATA", "/w/data/svhn.npz"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

DEFAULTS = dict(  # padrões = cnn_best do notebook (#17), menos a arquitetura
    arch="cnn", arch_args={}, epochs=30, batch=64, lr=1e-3, optimizer="adamw", momentum=0.9, nesterov=False,
    weight_decay=0.05, label_smoothing=0.1, augment=True, geo=1.0, color=0.0, cutout=True, cutout_size=8,
    cutout_p=1.0, ema=0.0, amp="none", seed=SEED, split_seed=SEED, final=False, stop_epoch=None,
    policy=None, ra_n=2, ra_m=9,  # augment automático por imagem, na CPU (substitui o `augment` geométrico)
)


class PolicyData(torch.utils.data.Dataset):
    """Imagens uint8 com RandAugment, TrivialAugmentWide ou AutoAugment (política SVHN), uma por imagem."""

    def __init__(self, x, y, cfg):
        from torchvision import transforms as T
        self.x, self.y = x, y
        self.t = {"ra": lambda: T.RandAugment(num_ops=cfg["ra_n"], magnitude=cfg["ra_m"]),
                  "ta": T.TrivialAugmentWide,
                  "aa_svhn": lambda: T.AutoAugment(T.AutoAugmentPolicy.SVHN)}[cfg["policy"]]()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.t(self.x[i]), self.y[i]


def policy_loader(x, y, cfg):
    torch.set_num_threads(1)
    return torch.utils.data.DataLoader(
        PolicyData(x.cpu(), y.cpu(), cfg), batch_size=cfg["batch"], shuffle=True, num_workers=4,
        persistent_workers=True, drop_last=len(x) % cfg["batch"] == 1, pin_memory=True,
        generator=torch.Generator().manual_seed(cfg["seed"]))


# ---------------------------------------------------------------- dados

def load_data(cfg):
    """Tensores uint8 na GPU: (x_tr, y_tr, x_val, y_val, x_test, y_test) e a normalização da divisão 42."""
    d = np.load(DATA)
    x, y = torch.from_numpy(d["x_train_full"]), torch.from_numpy(d["y_train_full"])
    perm42 = torch.from_numpy(d["perm42"])
    assert torch.equal(perm42, torch.randperm(len(x), generator=torch.Generator().manual_seed(SEED))), "divisão 42"
    perm = perm42 if cfg["split_seed"] == SEED else torch.randperm(
        len(x), generator=torch.Generator().manual_seed(cfg["split_seed"]))
    val, tr = perm[: len(perm) // 10], perm[len(perm) // 10:]
    # Como no notebook: a normalização é sempre a da divisão principal (semente 42).
    px = x[perm42[len(perm42) // 10:]].float().div(255)
    std, mean = torch.std_mean(px, dim=(0, 2, 3))
    del px
    if cfg["final"]:  # versão final: todo o split="train"
        tr, val = perm, perm[:0]
    g = lambda t: t.to(DEVICE)
    return (g(x[tr]), g(y[tr]), g(x[val]), g(y[val]), g(torch.from_numpy(d["x_test"])),
            g(torch.from_numpy(d["y_test"])), g(mean.view(1, 3, 1, 1)), g(std.view(1, 3, 1, 1)))


def preprocess(x, mean, std, cfg=None, train=False):
    x = x.float() / 255
    if train and cfg["color"]:  # brilho e contraste por imagem, antes de normalizar
        n, a = len(x), cfg["color"]
        c = 1 + (torch.rand(n, 1, 1, 1, device=x.device) * 2 - 1) * a
        b = (torch.rand(n, 1, 1, 1, device=x.device) * 2 - 1) * a / 2
        m = x.mean((1, 2, 3), keepdim=True)
        x = ((x - m) * c + m + b).clamp(0, 1)
    return (x - mean) / std


def augment(x, geo=1.0):
    """Igual ao notebook (rotação ±8°, escala ±10%, translação ±8%), com a intensidade multiplicada por `geo`."""
    n, dev = len(x), x.device
    rand = lambda: torch.rand(n, device=dev) * 2 - 1
    angle, scale = rand() * math.radians(8) * geo, 1 + rand() * 0.1 * geo
    tx, ty = rand() * 0.16 * geo, rand() * 0.16 * geo
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


def cutout(x, size=8, p=1.0):
    n, _, h, w = x.shape
    cy, cx = (torch.randint(0, d, (n, 1, 1), device=x.device) for d in (h, w))
    ys, xs = torch.arange(h, device=x.device).view(1, h, 1), torch.arange(w, device=x.device).view(1, 1, w)
    hole = (ys >= cy - size // 2) & (ys < cy + size // 2) & (xs >= cx - size // 2) & (xs < cx + size // 2)
    if p < 1:
        hole &= torch.rand(n, 1, 1, device=x.device) < p
    return x * ~hole[:, None]


# ---------------------------------------------------------------- modelos

def cnn(widths=(32, 64, 128), dropout=0.3, convs=2):
    """A CNN do notebook: 3 blocos [conv-BN-ReLU x convs + maxpool] e uma camada densa de 256."""
    layers, cin = [], 3
    for w in widths:
        for _ in range(convs):
            layers += [nn.Conv2d(cin, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU()]
            cin = w
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers, nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin * 4 * 4, 256), nn.ReLU(),
                         nn.Dropout(dropout), nn.Linear(256, 10))


class WideBlock(nn.Module):
    """Bloco residual pré-ativado da Wide ResNet (BN-ReLU-conv x2, dropout entre as convoluções)."""

    def __init__(self, cin, cout, stride, dropout):
        super().__init__()
        self.bn1, self.conv1 = nn.BatchNorm2d(cin), nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2, self.conv2 = nn.BatchNorm2d(cout), nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.short = None if cin == cout and stride == 1 else nn.Conv2d(cin, cout, 1, stride, bias=False)

    def forward(self, x):
        o = F.relu(self.bn1(x))
        skip = x if self.short is None else self.short(o)
        o = self.conv1(o)
        o = self.conv2(self.drop(F.relu(self.bn2(o))))
        return o + skip


def wrn(depth=16, widen=4, dropout=0.0):
    """Wide ResNet WRN-depth-widen para 32x32 (Zagoruyko e Komodakis, 2016): 3 estágios 32→16→8."""
    n = (depth - 4) // 6
    assert depth == 6 * n + 4, "depth = 6n + 4"
    widths, layers, cin = [16 * widen, 32 * widen, 64 * widen], [nn.Conv2d(3, 16, 3, 1, 1, bias=False)], 16
    for i, w in enumerate(widths):
        for j in range(n):
            layers.append(WideBlock(cin, w, 2 if (i > 0 and j == 0) else 1, dropout))
            cin = w
    return nn.Sequential(*layers, nn.BatchNorm2d(cin), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Linear(cin, 10))


ARCHS = {"cnn": cnn, "wrn": wrn}


def build(cfg):
    torch.manual_seed(cfg["seed"])
    return ARCHS[cfg["arch"]](**cfg["arch_args"]).to(DEVICE).to(memory_format=torch.channels_last)


# ---------------------------------------------------------------- treino

def autocast(cfg):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["amp"] == "bf16")


@torch.no_grad()
def evaluate(model, x, y, mean, std, cfg):
    """(loss de entropia cruzada, acurácia, probabilidades) sem augment, em lotes de 2048."""
    model.eval()
    probs = []
    for xb in x.split(2048):
        with autocast(cfg):
            out = model(preprocess(xb, mean, std).contiguous(memory_format=torch.channels_last))
        probs.append(out.float().softmax(1))
    p = torch.cat(probs) if probs else torch.zeros(0, 10, device=x.device)
    if len(y) == 0:
        return float("nan"), float("nan"), p
    loss = F.nll_loss(p.clamp_min(1e-12).log(), y).item()
    return loss, (p.argmax(1) == y).float().mean().item(), p


def train(cfg, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    x_tr, y_tr, x_v, y_v, x_te, y_te, mean, std = load_data(cfg)
    model = build(cfg)
    if cfg["optimizer"] == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                              nesterov=cfg["nesterov"], weight_decay=cfg["weight_decay"])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg["ema"]), use_buffers=True) if cfg["ema"] else None
    n_epochs = cfg["stop_epoch"] or cfg["epochs"]
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "epoch_s", "eval_s")}
    best = dict(val_acc=-1.0, epoch=0, state=None, val_probs=None)
    start_ep, elapsed = 1, 0.0

    last = run_dir / "ckpt_last.pt"
    if last.exists():  # retomada
        ck = torch.load(last, map_location=DEVICE, weights_only=False)
        assert ck["config"] == cfg, "ckpt_last.pt de outra config"
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        if ema:
            ema.load_state_dict(ck["ema"])
        hist, best, start_ep, elapsed = ck["hist"], ck["best"], ck["epoch"] + 1, ck["elapsed"]
        torch.set_rng_state(ck["rng_cpu"].cpu()); torch.cuda.set_rng_state(ck["rng_cuda"].cpu())
        log(run_dir, f"retomando da época {start_ep}")

    n = len(x_tr) - 1 if len(x_tr) % cfg["batch"] == 1 else len(x_tr)  # sem lote de 1 imagem
    loader = policy_loader(x_tr, y_tr, cfg) if cfg.get("policy") else None

    def batches():
        if loader is not None:
            for xb, yb in loader:
                yield xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            return
        idx = torch.randperm(len(x_tr), device=DEVICE)
        for i in range(0, n, cfg["batch"]):
            j = idx[i:i + cfg["batch"]]
            yield x_tr[j], y_tr[j]

    for ep in range(start_ep, n_epochs + 1):
        t0 = time.time()
        model.train()
        for xraw, yb in batches():
            xb = preprocess(xraw, mean, std, cfg, train=True)
            if cfg["augment"]:
                xb = augment(xb, cfg["geo"])
            if cfg["cutout"]:
                xb = cutout(xb, cfg["cutout_size"], cfg["cutout_p"])
            with autocast(cfg):
                out = model(xb.contiguous(memory_format=torch.channels_last))
                loss = F.cross_entropy(out.float(), yb, label_smoothing=cfg["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if ema:
                ema.update_parameters(model)
        sched.step()
        torch.cuda.synchronize()
        t1 = time.time()
        evaluated = ema.module if ema else model  # com EMA, as métricas e a melhor época são as do modelo EMA
        n_tr_eval = int(os.environ.get("TRAIN_EVAL_N", len(x_tr)))  # amostra fixa do treino, para economizar tempo
        tl, ta, _ = evaluate(evaluated, x_tr[:n_tr_eval], y_tr[:n_tr_eval], mean, std, cfg)
        vl, va, vp = evaluate(evaluated, x_v, y_v, mean, std, cfg)
        t2 = time.time()
        for k, v in zip(hist, (tl, ta, vl, va, t1 - t0, t2 - t1)):
            hist[k].append(v)
        elapsed += t2 - t0
        if cfg["final"] or va > best["val_acc"]:  # versão final: sem validação, fica a última época
            best = dict(val_acc=va, epoch=ep, state={k: v.detach().clone() for k, v in evaluated.state_dict().items()},
                        val_probs=vp.cpu())
        if cfg["final"] and ep >= int(os.environ.get("FINAL_SAVE_FROM", 10**9)):  # para escolher a época depois
            torch.save({k: v.cpu() for k, v in evaluated.state_dict().items()}, run_dir / f"final_ep{ep}.pt")
        log(run_dir, f"ep {ep:3d}/{n_epochs} | treino loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {va:.4f} "
                     f"| {t1 - t0:.1f}s + aval {t2 - t1:.1f}s")
        ck = dict(config=cfg, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                  ema=ema.state_dict() if ema else None, hist=hist, best=best, epoch=ep, elapsed=elapsed,
                  rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state())
        torch.save(ck, last.with_suffix(".tmp"))
        last.with_suffix(".tmp").replace(last)

    # pesos da melhor época (EMA, se usado); o modelo avaliado tem a mesma arquitetura que build(cfg)
    final_model = build(cfg)
    final_model.load_state_dict({k.removeprefix("module."): v for k, v in best["state"].items()
                                 if k != "n_averaged"})
    _, _, test_probs = evaluate(final_model, x_te, y_te, mean, std, cfg)
    torch.save(dict(config=cfg, state_dict={k: v.cpu() for k, v in final_model.state_dict().items()},
                    epoch=best["epoch"]), run_dir / "best.pt")
    np.save(run_dir / "test_probs.npy", test_probs.cpu().numpy().astype(np.float32))  # só para a avaliação final
    if not cfg["final"]:
        np.save(run_dir / "val_probs.npy", best["val_probs"].cpu().numpy().astype(np.float32))
    b = best["epoch"] - 1
    warm = hist["epoch_s"][1:] or hist["epoch_s"]
    summary = dict(
        melhor_epoca=best["epoch"], epocas=n_epochs, val_acc=hist["val_acc"][b], val_loss=hist["val_loss"][b],
        train_acc=hist["train_acc"][b], train_loss=hist["train_loss"][b],
        parametros=sum(p.numel() for p in final_model.parameters()),
        epoca_s_media=sum(warm) / len(warm), aval_s_media=sum(hist["eval_s"]) / len(hist["eval_s"]),
        tempo_total_s=elapsed, ambiente=environment(),
    )
    (run_dir / "hist.json").write_text(json.dumps(hist))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    last.unlink()
    return summary


def environment():
    code = Path(__file__).read_bytes()
    return dict(host=socket.gethostname(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                torch=torch.__version__, cudnn_ativo=torch.backends.cudnn.enabled,
                train_eval_n=os.environ.get("TRAIN_EVAL_N", "todo"), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                python=platform.python_version(), codigo_sha256=hashlib.sha256(code).hexdigest()[:16],
                git=os.environ.get("GIT_COMMIT", "?"), dados_sha256=os.environ.get("DATA_SHA", "?"))


def log(run_dir, msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(f"[{run_dir.name}] {line}", flush=True)
    with open(run_dir / "log.txt", "a") as f:
        f.write(line + "\n")


def full(cfg):
    unknown = set(cfg) - set(DEFAULTS) - {"name"}
    assert not unknown, f"chaves desconhecidas: {unknown}"  # um erro de digitação não vira outro experimento
    return {**DEFAULTS, **{k: v for k, v in cfg.items() if k != "name"}}


def run_config(cfg_path, runs):
    raw = json.loads(Path(cfg_path).read_text())
    cfg, run_dir = full(raw), Path(runs) / raw["name"]
    saved = run_dir / "config.json"
    if saved.exists():
        assert json.loads(saved.read_text()) == cfg, f"{run_dir} já existe com outra config"
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        saved.write_text(json.dumps(cfg, indent=2))
    if (run_dir / "summary.json").exists():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = os.environ.get("NO_CUDNN") != "1"  # a imagem vLLM não tem o cuDNN de treino
    train(cfg, run_dir)


def worker(queue, runs, wid):
    """Treina, em ordem, as configs de `queue` que ainda não têm summary.json. Retoma as que este worker pegou
    antes. Sem trabalho, espera novas configs; para quando existir `queue/STOP`."""
    queue, runs = Path(queue), Path(runs)
    while not (queue / "STOP").exists():
        did = False
        for cfg_path in sorted(queue.glob("*.json")):
            run_dir = runs / cfg_path.stem
            if (run_dir / "summary.json").exists() or (run_dir / "error.txt").exists():
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            claim = run_dir / "claim"
            try:
                fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, wid.encode()); os.close(fd)
            except FileExistsError:
                if claim.read_text() != wid:
                    continue  # outro worker
            try:
                run_config(cfg_path, runs)
            except Exception:
                (run_dir / "error.txt").write_text(traceback.format_exc())
                print(traceback.format_exc(), flush=True)
            did = True
            break  # relê a fila: novas configs podem ter entrado na frente
        if not did:
            time.sleep(30)


# ---------------------------------------------------------------- inferência (TTA)

TTAS = {"none": [], "shift1": [(1, 0), (-1, 0), (0, 1), (0, -1)], "shift2": [(2, 0), (-2, 0), (0, 2), (0, -2)],
        "shift12": [(1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2)]}


def shift(x, dx, dy):
    pad = max(abs(dx), abs(dy))
    h, w = x.shape[-2:]
    p = F.pad(x, (pad, pad, pad, pad))
    return p[..., pad - dy: pad - dy + h, pad - dx: pad - dx + w]


@torch.no_grad()
def infer(run_dir, split, tta, extra=None):
    """Probabilidades médias (imagem + cópias deslocadas) do best.pt de `run_dir`; salva probs_<split>_<tta>.npy."""
    run_dir = Path(run_dir)
    ck = torch.load(run_dir / "best.pt", map_location=DEVICE, weights_only=False)
    cfg = ck["config"]
    model = build(cfg)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    x_tr, y_tr, x_v, y_v, x_te, y_te, mean, std = load_data({**cfg, "final": False})
    x = {"val": x_v, "test": x_te}.get(split)
    if split == "extra":
        x = torch.from_numpy(np.load(extra)).to(DEVICE)
    out = []
    for xb in x.split(2048):
        xb = preprocess(xb, mean, std)
        views = [xb] + [shift(xb, dx, dy) for dx, dy in TTAS[tta]]
        with autocast(cfg):
            ps = [model(v.contiguous(memory_format=torch.channels_last)).float().softmax(1) for v in views]
        out.append(torch.stack(ps).mean(0))
    probs = torch.cat(out).cpu().numpy().astype(np.float32)
    np.save(run_dir / f"probs_{split}_{tta}.npy", probs)
    return probs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["worker", "run", "infer"])
    ap.add_argument("--queue"); ap.add_argument("--runs"); ap.add_argument("--id", default=socket.gethostname())
    ap.add_argument("--config"); ap.add_argument("--run"); ap.add_argument("--split"); ap.add_argument("--tta")
    ap.add_argument("--extra")
    a = ap.parse_args()
    if a.cmd == "worker":
        worker(a.queue, a.runs, a.id)
    elif a.cmd == "run":
        run_config(a.config, a.runs)
    else:
        infer(a.run, a.split, a.tta, a.extra)
    sys.exit(0)
