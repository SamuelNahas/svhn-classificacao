"""Confere as saídas do notebook rodado em modo de teste (SVHN_SMOKE=1).

Uso: python tests/check_smoke.py [smoke]
"""
import csv
import json
import sys
from pathlib import Path

N_EXPERIMENTS = 15


def check(root):
    results, errors = root / "results", []
    rows = list(csv.DictReader(open(results / "results.csv")))
    if len(rows) != N_EXPERIMENTS:
        errors.append(f"results.csv tem {len(rows)} experimentos, esperado {N_EXPERIMENTS}")
    for r in rows:
        name = r["modificacao"]
        for k in ("train_acc", "val_acc", "test_acc"):
            if not 0 <= float(r[k]) <= 1:
                errors.append(f"{name}: {k} = {r[k]} fora de [0, 1]")
        for k in ("train_loss", "val_loss"):
            if not float(r[k]) > 0:
                errors.append(f"{name}: {k} = {r[k]} não é positiva")
        if not (results / f"{name}.png").exists():
            errors.append(f"{name}: curva {name}.png não foi gerada")
    if not (results / "comparacao_validacao.png").exists():
        errors.append("comparacao_validacao.png não foi gerada")

    final = json.loads((results / "final.json").read_text())
    if final["experimento"] not in {r["modificacao"] for r in rows}:
        errors.append(f"final.json aponta para um experimento inexistente: {final['experimento']}")
    for k in ("val_acc", "test_acc", "train_acc"):
        if not 0 <= final[k] <= 1:
            errors.append(f"final.json: {k} = {final[k]} fora de [0, 1]")
    if not (root / "weights" / "best.pt").exists():
        errors.append("weights/best.pt não foi salvo")
    return errors


if __name__ == "__main__":
    errors = check(Path(sys.argv[1] if len(sys.argv) > 1 else "smoke"))
    for e in errors:
        print("ERRO:", e)
    print("ok" if not errors else f"{len(errors)} erro(s)")
    sys.exit(1 if errors else 0)
