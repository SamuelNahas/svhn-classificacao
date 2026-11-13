"""Monta o ensemble final com TTA shift12 e épocas escolhidas pela validação.

O padrão reproduz a pt4: 20 CNNs, sementes 42–61, prefixos e60_val/zf60.
A rodada 5b usa --seeds 42 47 48 --val r_wrn282_ra --final rf_wrn282_ra --epochs 200.
Cada membro carrega final_ep<E>.pt do retreino, com E vindo do summary.json de validação.
Saídas: membros/ (pesos e config), probs_test.npy, probs_extra.npy, membros.json e test_acc.txt.
Execute apenas após fixar a solução pela validação; o script avalia o teste ao concluir.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import svhn_dgx as S

SEEDS = list(range(42, 62))  # pt4: ordem fixa registrada antes dos resultados; k = 20
TTA = "shift12"


@torch.no_grad()
def probs(model, x, mean, std, cfg):
    out = []
    for xb in x.split(2048):
        xb = S.preprocess(xb, mean, std)
        views = [xb] + [S.shift(xb, dx, dy) for dx, dy in S.TTAS[TTA]]
        with S.autocast(cfg):
            ps = [model(v.contiguous(memory_format=torch.channels_last)).float().softmax(1) for v in views]
        out.append(torch.stack(ps).mean(0))
    return torch.cat(out)


def main(runs, extra, out, seeds=SEEDS, val="e60_val", final="zf60", epochs=60):
    runs, out = Path(runs), Path(out)
    (out / "membros").mkdir(parents=True, exist_ok=True)
    x_extra = torch.from_numpy(np.load(extra)).to(S.DEVICE)
    soma_test, soma_extra, membros = 0, 0, []
    for seed in seeds:
        epoca = json.loads((runs / f"{val}_s{seed}" / "summary.json").read_text())["melhor_epoca"]
        cfg = json.loads((runs / f"{final}_s{seed}" / "config.json").read_text())
        assert cfg["final"] and cfg["seed"] == seed and cfg["epochs"] == epochs
        state = torch.load(runs / f"{final}_s{seed}" / f"final_ep{epoca}.pt", map_location=S.DEVICE)
        model = S.build(cfg)
        model.load_state_dict(state)
        model.eval()
        _, _, _, _, x_te, y_te, mean, std = S.load_data({**cfg, "final": False})
        p_te, p_ex = probs(model, x_te, mean, std, cfg), probs(model, x_extra, mean, std, cfg)
        soma_test, soma_extra = soma_test + p_te, soma_extra + p_ex
        torch.save(dict(config=cfg, epoca=epoca, state_dict={k: v.cpu() for k, v in state.items()}),
                   out / "membros" / f"s{seed}.pt")
        membros.append(dict(semente=seed, epoca=epoca, origem=f"{final}_s{seed}/final_ep{epoca}.pt",
                            epoca_escolhida_por=f"{val}_s{seed}/summary.json"))
        print(f"s{seed}: época {epoca}", flush=True)
    p_test, p_extra = (soma_test / len(seeds)).cpu().numpy(), (soma_extra / len(seeds)).cpu().numpy()
    np.save(out / "probs_test.npy", p_test)
    np.save(out / "probs_extra.npy", p_extra)
    acc = float((p_test.argmax(1) == y_te.cpu().numpy()).mean())
    (out / "test_acc.txt").write_text(f"{acc:.6f}\n")
    (out / "membros.json").write_text(json.dumps(dict(tta=TTA, membros=membros, env=S.environment()), indent=2))
    print(f"teste (avaliação única da solução fechada): {acc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs"); ap.add_argument("--extra", default="data/adicional.npy")
    ap.add_argument("--out", default="final")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)  # rodada 5b: --seeds 42 ... 51
    ap.add_argument("--val", default="e60_val"); ap.add_argument("--final", default="zf60")
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    main(a.runs, a.extra, a.out, a.seeds, a.val, a.final, a.epochs)
