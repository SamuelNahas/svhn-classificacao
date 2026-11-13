"""Gera as tabelas (.tex) e as figuras novas do relatório a partir dos arquivos de resultados do repositório.

Uso, na raiz do repositório: .venv/bin/python relatorio/gerar.py
Lê results/results.csv, results/*.json, results/tta_ensemble.csv, results/final_predicao.json e results/dgx/.
Escreve relatorio/tabelas/*.tex e relatorio/figuras/*.pdf. Nenhum número das tabelas é digitado à mão.
"""
import csv
import json
import statistics as st
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

ROOT = Path(__file__).resolve().parent.parent
RES, DGX, OUT = ROOT / "results", ROOT / "results" / "dgx", ROOT / "relatorio"
(OUT / "tabelas").mkdir(exist_ok=True)
(OUT / "figuras").mkdir(exist_ok=True)

Y_TEST = scipy.io.loadmat(ROOT / "data" / "test_32x32.mat")["y"].ravel() % 10  # rótulo 10 = dígito 0


def pct(x, nd=2):
    return "—" if x is None else f"{100 * x:.{nd}f}".replace(".", ",")


def num(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}".replace(".", ",")


def delta(x, base):
    if x is None or base is None:
        return ""
    d = 100 * (x - base)
    return ("+" if d >= 0 else "$-$") + f"{abs(d):.2f}".replace(".", ",")


def verdict(x, base):
    if x is None or base is None:
        return ""
    return "melhorou" if x > base else "piorou" if x < base else "igual"


def row(cells):
    return " & ".join(cells) + r" \\"


# ---------------------------------------------------------------- dados
nb = {r["modificacao"]: r for r in csv.DictReader(open(RES / "results.csv"))}
f = lambda r, k: float(r[k]) if r and r.get(k) not in (None, "") else None
dg = {r["experimento"]: r for r in csv.DictReader(open(DGX / "resumo.csv"))}


def test_acc(names):
    """Acurácia de teste da média do softmax de `names` (sem TTA). Lida depois do fechamento da solução."""
    p = np.mean([np.load(DGX / n / "test_probs.npy") for n in names], 0)
    return float((p.argmax(1) == Y_TEST).mean())


def group(names):
    """Médias das métricas dos treinos em lote `names` (uma semente cada)."""
    return {k: st.mean(float(dg[n][k]) for n in names) for k in ("train_loss", "val_loss", "train_acc", "val_acc")} | {
        "test_acc": st.mean(test_acc([n]) for n in names)}


def single(n):
    return group([n])


TRI = ["t01_ref_bf16", "t03_wrn16_4", "t04_wrn16_4_sgd", "t05_wrn22_1", "t06_ema", "t07_color", "t08_geo05",
       "t09_geo15", "t10_cut12", "t11_cutp05", "t12_ep60", "t17_wrn16_8", "t18_wrn28_4"]
seeds = lambda pre, ss: [f"{pre}{s}" for s in ss]
GROUPS = {  # grupos de sementes: as linhas de confirmação são médias
    "w30": ["t03_wrn16_4", *seeds("w30_val_s", (43, 44))], "w60": ["t16_wrn16_4_ep60", *seeds("w60_val_s", (43, 44))],
    "e60": seeds("e60_val_s", (42, 43, 44)), "a_ra": seeds("a_ra_s", (42, 43, 44)),
    "a_aa": seeds("a_aa_s", (42, 43, 44)), "a_ta": seeds("a_ta_s", (42, 43, 44)),
    "cf30": seeds("cf30_d43_s", range(42, 52)), "cf60": seeds("cf60_d43_s", range(42, 52)),
}
g = {lab: group(runs) for lab, runs in GROUPS.items()}

# ensembles: validação da curva registrada; teste sem TTA (média dos test_probs dos membros de validação)
curve = {(r["estrategia"], r["tta"], int(r["k"])): float(r["val_acc"]) for r in csv.DictReader(open(DGX / "curvas_ensemble.csv"))}
tta5 = {r["metodo"]: r for r in csv.DictReader(open(RES / "tta_ensemble.csv"))}
final = json.loads((RES / "final_predicao.json").read_text())
if not (final["n_modelos"] == 20 and "pt4" in final["metodo"]):  # notebook ainda não rodou com weights/pt4/
    print("AVISO: final_predicao.json não é da pt4; usando o teste do script em lote (results/dgx/final/)")
    final = {"test_acc": float((DGX / "final" / "test_acc.txt").read_text())}
e60 = seeds("e60_val_s", range(42, 72))
T5 = lambda m: (float(tta5[m]["val_acc"]), float(tta5[m]["test_acc"]))
ENS = {  # rótulo: (val, teste)
    "E2": T5("ensemble de 5"), "E3": T5("ensemble de 5 + TTA"),
    "E4": (curve[("cnn60", "none", 5)], test_acc(e60[:5])), "E5": (curve[("cnn60", "none", 10)], test_acc(e60[:10])),
    "E6": (curve[("cnn60", "none", 20)], test_acc(e60[:20])), "E7": (curve[("cnn60", "none", 30)], test_acc(e60[:30])),
    "E8": (curve[("cnn60", "shift12", 10)], None), "E9": (curve[("wrn16_4", "shift12", 10)], None),
    "E10": (curve[("cnn60", "shift12", 20)], None), "E11": (curve[("heterogeneo", "shift12", 20)], None),
}

# ---------------------------------------------------------------- tabelas do corpo (9.2 e 9.3)
# (val, teste) de qualquer experimento: nome do notebook, treino em lote, grupo de sementes ou ensemble
acc = {n: (f(r, "val_acc"), f(r, "test_acc")) for n, r in nb.items()}
acc |= {n: (single(n)["val_acc"], single(n)["test_acc"]) for n in TRI}
acc |= {lab: (m["val_acc"], m["test_acc"]) for lab, m in g.items()}
acc |= ENS
acc["pt4"] = (ENS["E10"][0], final["test_acc"])
r5b = json.loads((DGX / "final_r5b" / "membros.json").read_text())
r5b_seeds = [member["semente"] for member in r5b["membros"]]
assert r5b_seeds == [42, 47, 48] and r5b["tta"] == "shift12"
r5b_probs = np.mean([np.load(DGX / f"r_wrn282_ra_s{s}" / "probs_val_shift12.npy") for s in r5b_seeds], 0)
r5b_labels = np.load(DGX / "val_labels.npy")
r5b_val = float((r5b_probs.argmax(1) == r5b_labels).mean())
r5b_test = float((DGX / "final_r5b" / "test_acc.txt").read_text())
assert r5b_val > acc["pt4"][0]
acc["final"] = (r5b_val, r5b_test)


def res(name, base):
    """'melhorou (+0,26)' / 'piorou (−0,11)' na validação, em relação a `base`."""
    d = delta(acc[name][0], acc[base][0])
    return f"{verdict(acc[name][0], acc[base][0])} ({d})"


EVOL = [  # 9.2, Tabela 1: cada linha parte da anterior (nome, experimento, modificação)
    ("linear", "Classificador linear", "camada linear 3072$\\to$10, softmax"),
    ("mlp_baseline", "MLP baseline", "MLP 512-256, ReLU, Adam lr $10^{-3}$"),
    ("mlp_sgd_momentum", "MLP + melhoria", "SGD com momentum 0,9 e lr $10^{-2}$ no lugar do Adam"),
    ("cnn", "CNN", "3 blocos convolucionais (32/64/128 filtros) no lugar da MLP"),
    ("cnn_augment", "CNN + augment", "data augmentation (rotação, escala e translação)"),
    ("cnn_best", "\\texttt{cnn\\_best}", "+ weight decay, 30 épocas, label smoothing, 3 convoluções por bloco e cutout"),
    ("E2", "Ensemble de 5", "média de 5 \\texttt{cnn\\_best} (sementes 42--46)"),
    ("E4", "Ensemble de 5, 60 épocas", "cada membro treinado por 60 épocas em vez de 30"),
    ("E6", "Ensemble de 20, 60 épocas", "20 membros (sementes 42--61) em vez de 5"),
    ("pt4", "CNN de 20 + TTA", "+ TTA shift12; retreino no treino inteiro"),
    ("final", "\\textbf{Solução final}", "3 WRN-28-2 + RandAugment + shift12$^{a}$"),
]
lines, prev = [], None
for i, (n, name, mod) in enumerate(EVOL, 1):
    va, te = acc[n]
    lines.append(row([str(i), name, mod, pct(va), pct(te), res(n, prev) if prev else "---"]))
    prev = n
(OUT / "tabelas" / "evolucao.tex").write_text("\n".join(lines) + "\n")

OUTROS = [  # 9.2, Tabela 2: (grupo) ou (nome, modificação, base)
    "MLP (base: \\texttt{mlp\\_baseline}, salvo indicação)",
    ("mlp_lr_1e-4", "lr $10^{-4}$", "mlp_baseline"), ("mlp_lr_1e-2", "lr $10^{-2}$", "mlp_baseline"),
    ("mlp_batch_32", "batch 32", "mlp_baseline"), ("mlp_batch_256", "batch 256", "mlp_baseline"),
    ("mlp_sgd", "SGD sem momentum, lr $10^{-2}$ (base: Adam com a mesma lr)", "mlp_lr_1e-2"),
    ("mlp_100", "uma camada oculta de 100", "mlp_baseline"), ("mlp_deep", "camadas 1024-512-256 (base: BatchNorm + Dropout)", "mlp_bn_dropout"),
    ("mlp_leaky_relu", "Leaky ReLU", "mlp_baseline"), ("mlp_tanh", "tanh", "mlp_baseline"),
    ("mlp_sigmoid", "sigmóide", "mlp_baseline"), ("mlp_sem_norm", "sem normalização (só /255)", "mlp_baseline"),
    ("mlp_so_media", "só subtrai a média", "mlp_baseline"), ("mlp_cinza", "tons de cinza", "mlp_baseline"),
    ("mlp_l2", "regularização L2, $\\lambda=10^{-4}$", "mlp_baseline"), ("mlp_dropout", "Dropout 0,3", "mlp_baseline"),
    ("mlp_dropout_0.5", "Dropout 0,5 (base: Dropout 0,3)", "mlp_dropout"),
    ("mlp_bn_dropout", "BatchNorm + Dropout 0,3 (base: Dropout 0,3)", "mlp_dropout"),
    ("linear_svm", "classificador linear com loss SVM (base: linear)", "linear"),
    "CNN (base: \\texttt{cnn\\_augment}); as que melhoraram formam a \\texttt{cnn\\_best}",
    ("cnn_wide", "filtros 64/128/256", "cnn_augment"), ("cnn_wd", "weight decay 0,05", "cnn_augment"),
    ("cnn_30ep", "30 épocas", "cnn_augment"), ("cnn_ls", "label smoothing 0,1", "cnn_augment"),
    ("cnn_deep", "3 convoluções por bloco", "cnn_augment"), ("cnn_cutout", "cutout 8×8", "cnn_augment"),
    "Sobre a \\texttt{cnn\\_best} (base: \\texttt{t01\\_ref\\_bf16}, a \\texttt{cnn\\_best} no script em lote)$^{b}$",
    ("t12_ep60", "60 épocas", "t01_ref_bf16"), ("t06_ema", "média móvel dos pesos (EMA)", "t01_ref_bf16"),
    ("t07_color", "brilho/contraste ±20\\%", "t01_ref_bf16"), ("t08_geo05", "augment geométrico ×0,5", "t01_ref_bf16"),
    ("t09_geo15", "augment geométrico ×1,5", "t01_ref_bf16"), ("t10_cut12", "cutout 12×12", "t01_ref_bf16"),
    ("t11_cutp05", "cutout em metade das imagens", "t01_ref_bf16"),
    ("t05_wrn22_1", "ResNet pequena (WRN-22-1)", "t01_ref_bf16"), ("t03_wrn16_4", "Wide ResNet 16-4", "t01_ref_bf16"),
    ("t04_wrn16_4_sgd", "WRN-16-4 com SGD (base: WRN-16-4)", "t03_wrn16_4"),
    ("t17_wrn16_8", "WRN-16-8 (base: WRN-16-4)", "t03_wrn16_4"), ("t18_wrn28_4", "WRN-28-4 (base: WRN-16-4)", "t03_wrn16_4"),
    "Confirmação com várias sementes (médias)$^{b}$",
    ("w60", "WRN-16-4 com 60 épocas (base: 30 épocas; 3 sementes)", "w30"),
    ("cf60", "CNN com 60 épocas (base: 30 épocas) em outra divisão (10 sementes)", "cf30"),
    ("a_ra", "RandAugment no lugar do augment (base: CNN de 60 épocas; 3 sementes)", "e60"),
    ("a_aa", "AutoAugment SVHN (idem)", "e60"), ("a_ta", "TrivialAugment (idem)", "e60"),
    "Ensembles e TTA$^{b,c}$",
    ("E3", "TTA ±2 px no ensemble de 5 \\texttt{cnn\\_best} (base: sem TTA)", "E2"),
    ("E5", "10 membros de 60 épocas (base: 5)", "E4"), ("E7", "30 membros de 60 épocas (base: 20)", "E6"),
    ("E10", "TTA shift12 no ensemble de 20 (base: sem TTA)", "E6"),
    ("E9", "10 WRN-16-4 no lugar de 10 CNNs, ambos com shift12", "E8"),
    ("E11", "20 membros mistos WRN e CNN no lugar de 20 CNNs, ambos com shift12", "E10"),
]
# loss de treino, loss de validação e acurácia de treino (ensembles não têm)
trn = {n: (f(r, "train_loss"), f(r, "val_loss"), f(r, "train_acc")) for n, r in nb.items()}
trn |= {n: tuple(single(n)[k] for k in ("train_loss", "val_loss", "train_acc")) for n in TRI}
trn |= {lab: tuple(m[k] for k in ("train_loss", "val_loss", "train_acc")) for lab, m in g.items()}
lines = []
for item in OUTROS:
    if isinstance(item, str):
        lines += [r"\midrule" if lines else "", r"\multicolumn{7}{l}{\textit{" + item + r"}}\\"]
        continue
    n, mod, base = item
    va, te = acc[n]
    tl, vl, ta = trn.get(n, (None, None, None))
    lines.append(row([mod, num(tl), num(vl), pct(ta), pct(va), pct(te), res(n, base)]))
(OUT / "tabelas" / "outros.tex").write_text("\n".join(l for l in lines if l) + "\n")

TREINO = [("linear", "Classificador linear"), ("mlp_baseline", "MLP baseline"), ("mlp_sgd_momentum", "MLP + melhoria"),
          ("cnn", "CNN"), ("cnn_augment", "CNN + augment"), ("cnn_best", "\\texttt{cnn\\_best}")]
lines = []
for n, name in TREINO:
    r = nb[n]
    lines.append(row([name, num(f(r, "train_loss")), num(f(r, "val_loss")), pct(f(r, "train_acc")), pct(f(r, "val_acc")),
                      pct(f(r, "test_acc")), f"{r['melhor_epoca']}/{r['epocas']}",
                      f"{100 * (f(r, 'train_acc') - f(r, 'val_acc')):.1f}".replace(".", ",")]))
e = dg["e60_val_s42"]
lines.append(row(["CNN de 60 épocas (s42)$^{b}$", num(float(e["train_loss"])), num(float(e["val_loss"])),
                  pct(float(e["train_acc"])), pct(float(e["val_acc"])), pct(single("e60_val_s42")["test_acc"]),
                  f"{e['melhor_epoca']}/{e['epocas']}",
                  f"{100 * (float(e['train_acc']) - float(e['val_acc'])):.1f}".replace(".", ",")]))
r = json.loads((DGX / "r_wrn282_ra_s42" / "summary.json").read_text())
lines.append(row(["WRN-28-2 + RA (s42)$^{b}$", num(r["train_loss"]), num(r["val_loss"]),
                  pct(r["train_acc"]), pct(r["val_acc"]), "---", f"{r['melhor_epoca']}/{r['epocas']}",
                  num(100 * (r["train_acc"] - r["val_acc"]), nd=1)]))
(OUT / "tabelas" / "treino.tex").write_text("\n".join(lines) + "\n")

# números citados no texto: um \newcommand por valor, para o .tex nunca digitar números à mão
mlp_bn_dropout = nb["mlp_bn_dropout"]
cmd = {
    "valFinal": pct(r5b_val), "testeFinal": pct(r5b_test),
    "valEnsCinco": pct(float(tta5["ensemble de 5 + TTA"]["val_acc"])),
    "testeEnsCinco": pct(json.loads(subprocess.run(  # solução do PR #21, retreinada; o notebook sobrescreve o arquivo
        ["git", "show", "75a5ec2:results/final_predicao.json"], cwd=ROOT, capture_output=True, text=True,
        check=True).stdout)["test_acc"]),
    "valCnnBest": pct(f(nb["cnn_best"], "val_acc")), "testeCnnBestFinal": pct(json.loads((RES / "final.json").read_text())["test_acc"]),
    "valLinear": pct(f(nb["linear"], "val_acc")), "treinoLinear": pct(f(nb["linear"], "train_acc")),
    "valMlp": pct(f(nb["mlp_baseline"], "val_acc")), "treinoMlp": pct(f(nb["mlp_baseline"], "train_acc")),
    "valMlpMelhor": pct(f(nb["mlp_sgd_momentum"], "val_acc")),
    "valMlpBnDropout": pct(f(mlp_bn_dropout, "val_acc")), "valCnn": pct(f(nb["cnn"], "val_acc")),
    "treinoCnn": pct(f(nb["cnn"], "train_acc")),
    "valWtrinta": pct(g["w30"]["val_acc"]), "valWsessenta": pct(g["w60"]["val_acc"]),
    "valCftrinta": pct(g["cf30"]["val_acc"]), "valCfsessenta": pct(g["cf60"]["val_acc"]),
    "valEsessentaMedia": pct(st.mean(float(dg[n]["val_acc"]) for n in e60)),
}
macros = {name: f"{value}\\%" for name, value in cmd.items()}
macros["difMlpBnDropout"] = num(100 * (f(mlp_bn_dropout, "train_acc") - f(mlp_bn_dropout, "val_acc")), nd=1)
(OUT / "tabelas" / "numeros.tex").write_text(
    "".join(f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in macros.items())
)

# ---------------------------------------------------------------- figuras novas


def curves(pairs, path, title):
    """Loss e acurácia (treino e validação) de dois modelos, lado a lado."""
    fig, (al, aa) = plt.subplots(1, 2, figsize=(11, 3.6))
    for (label, hist), color in zip(pairs, ("tab:blue", "tab:red")):
        ep = range(1, len(hist["val_acc"]) + 1)
        for ax, key in ((al, "loss"), (aa, "acc")):
            ax.plot(ep, hist[f"train_{key}"], color=color, ls="--", label=f"{label} — treino")
            ax.plot(ep, hist[f"val_{key}"], color=color, label=f"{label} — validação")
    al.set(title="Loss", xlabel="época")
    aa.set(title="Acurácia", xlabel="época")
    for ax in (al, aa):
        ax.grid(alpha=0.3)
    aa.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


h = lambda n: json.loads((RES / f"{n}.json").read_text())["hist"]
curves([("mlp_baseline (Adam)", h("mlp_baseline")), ("mlp_sgd_momentum", h("mlp_sgd_momentum"))],
       OUT / "figuras" / "mlp.pdf", "MLP: baseline × melhor MLP na validação")
curves([("cnn_best (30 épocas)", h("cnn_best")),
        ("membro de 60 épocas (semente 42)", json.loads((DGX / "e60_val_s42" / "hist.json").read_text()))],
       OUT / "figuras" / "cnn_best_e60.pdf", "CNN: cnn_best × membro da solução anterior pt4")
curves([("CNN de 60 épocas (s42)", json.loads((DGX / "e60_val_s42" / "hist.json").read_text())),
        ("WRN-28-2 + RandAugment (s42)", json.loads((DGX / "r_wrn282_ra_s42" / "hist.json").read_text()))],
       OUT / "figuras" / "cnn_wrn_ra.pdf", "Receita anterior × nova receita (semente 42)")
print("tabelas e figuras geradas em", OUT)
