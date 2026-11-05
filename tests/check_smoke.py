"""Confere as saídas do notebook rodado em modo de teste (SVHN_SMOKE=1).

Uso: python tests/check_smoke.py [smoke]
"""
import csv
import json
import sys
from pathlib import Path

N_EXPERIMENTS = 28  # fixo de propósito: mudar os experimentos do notebook exige atualizar aqui


def check(root):
    try:
        return check_outputs(root)
    except Exception as e:  # arquivo ou chave ausente: vira erro na lista, não traceback
        return [f"{type(e).__name__}: {e}"]


def check_outputs(root):
    results, errors = root / "results", []
    with open(results / "results.csv") as f:
        rows = list(csv.DictReader(f))
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
    for f in ("comparacao_validacao.png", "taxas_aprendizado.png", "confusao.png", "erros.png"):
        if not (results / f).exists():
            errors.append(f"{f} não foi gerada")

    with open(results / "knn.csv") as f:
        knn = list(csv.DictReader(f))
    chosen = [r for r in knn if r["escolhido"] == "True"]
    if len(chosen) != 1 or not 0 <= float(chosen[0]["test_acc"]) <= 1:
        errors.append("knn.csv deve ter exatamente um K escolhido, com test_acc em [0, 1]")
    for r in knn:
        if not 0 <= float(r["val_acc"]) <= 1:
            errors.append(f"knn K={r['k']}: val_acc = {r['val_acc']} fora de [0, 1]")

    with open(results / "subsampling.csv") as f:
        sub = list(csv.DictReader(f))
    if len(sub) != 3:
        errors.append(f"subsampling.csv tem {len(sub)} modelos, esperado 3")
    for r in sub:
        if not 0 <= float(r["val_media"]) <= 1 or not 0 <= float(r["teste_media"]) <= 1:
            errors.append(f"subsampling {r['modelo']}: médias fora de [0, 1]")

    with open(results / "metricas_por_classe.csv") as f:
        per_class = list(csv.DictReader(f))
    if len(per_class) != 10:
        errors.append(f"metricas_por_classe.csv tem {len(per_class)} dígitos, esperado 10")
    for r in per_class:
        for k in ("precisao", "recall", "f1"):
            if not 0 <= float(r[k]) <= 1:
                errors.append(f"dígito {r['digito']}: {k} = {r[k]} fora de [0, 1]")

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
