"""Confere as saídas do notebook rodado em modo de teste (SVHN_SMOKE=1).

Uso: python tests/check_smoke.py [smoke]
"""
import csv
import json
import math
import sys
from pathlib import Path

N_EXPERIMENTS = 30  # fixo de propósito: mudar os experimentos do notebook exige atualizar aqui
N_SPLITS_SMOKE = 2  # divisões do random subsampling no modo de teste
N_SEEDS_SMOKE = 2  # modelos do ensemble no modo de teste
N_EXTRAS = 7  # #14: cnn_sem_bn, 3 inicializações da MLP e 3 tamanhos de treino


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
        if int(r["divisoes"]) != N_SPLITS_SMOKE:
            errors.append(f"subsampling {r['modelo']}: {r['divisoes']} divisões, esperado {N_SPLITS_SMOKE}")
        if not all(math.isfinite(float(r[k])) for k in ("val_desvio", "teste_desvio")):
            errors.append(f"subsampling {r['modelo']}: desvio não é finito")

    svm = [r for r in rows if r["modificacao"] == "linear_svm"]
    if not svm or not float(svm[0].get("val_loss_svm") or "nan") > 0:
        errors.append("linear_svm sem val_loss_svm positiva")

    with open(results / "metricas_por_classe.csv") as f:
        per_class = list(csv.DictReader(f))
    if len(per_class) != 10:
        errors.append(f"metricas_por_classe.csv tem {len(per_class)} dígitos, esperado 10")
    for r in per_class:
        for k in ("precisao", "recall", "f1"):
            if not 0 <= float(r[k]) <= 1:
                errors.append(f"dígito {r['digito']}: {k} = {r[k]} fora de [0, 1]")

    names = {r["modificacao"] for r in rows}
    final = json.loads((results / "final.json").read_text())
    if final["experimento"] not in names:
        errors.append(f"final.json aponta para um experimento inexistente: {final['experimento']}")
    for k in ("val_acc", "test_acc", "train_acc"):
        if not 0 <= final[k] <= 1:
            errors.append(f"final.json: {k} = {final[k]} fora de [0, 1]")
    if not (root / "weights" / "best.pt").exists():
        errors.append("weights/best.pt não foi salvo")

    with open(results / "extras.csv") as f:
        extras = list(csv.DictReader(f))
    if len(extras) != N_EXTRAS:
        errors.append(f"extras.csv tem {len(extras)} experimentos, esperado {N_EXTRAS}")
    for r in extras:
        if r["base"] not in names:
            errors.append(f"extra {r['experimento']}: base {r['base']} não está em results.csv")
        if r["experimento"].endswith("pct") and r["base"] != final["experimento"]:
            errors.append(f"extra {r['experimento']}: curva de aprendizado deveria ter base {final['experimento']}")
        if not math.isfinite(float(r["delta_val"])):
            errors.append(f"extra {r['experimento']}: delta_val não é finito")
        for k in ("train_acc", "val_acc", "test_acc"):
            if not 0 <= float(r[k]) <= 1:
                errors.append(f"extra {r['experimento']}: {k} = {r[k]} fora de [0, 1]")
    if not (results / "ensemble.png").exists():
        errors.append("ensemble.png não foi gerada")
    for f in ("extras_sem_bn.png", "extras_inicializacao.png", "extras_curva_aprendizado.png"):
        if not (results / f).exists():
            errors.append(f"{f} não foi gerada")

    with open(results / "tta_ensemble.csv") as f:
        methods = list(csv.DictReader(f))
    if len(methods) != 2 * N_SEEDS_SMOKE:
        errors.append(f"tta_ensemble.csv tem {len(methods)} métodos, esperado {2 * N_SEEDS_SMOKE} (1 a N modelos, com e sem TTA)")
    if sum(r["escolhido"] == "True" for r in methods) != 1:
        errors.append("tta_ensemble.csv deve ter exatamente um método escolhido")
    for r in methods:
        for k in ("val_acc", "test_acc"):
            if not 0 <= float(r[k]) <= 1:
                errors.append(f"método {r['metodo']}: {k} = {r[k]} fora de [0, 1]")
    pred = json.loads((results / "final_predicao.json").read_text())
    if not 0 <= pred["test_acc"] <= 1 or pred["n_modelos"] < 1:
        errors.append(f"final_predicao.json inválido: {pred}")
    n_final = len(list((root / "weights" / "ensemble").glob("final_seed*.pt")))
    if n_final != pred["n_modelos"] - 1:  # o primeiro modelo do ensemble é o weights/best.pt
        errors.append(f"weights/ensemble tem {n_final} modelos finais, esperado {pred['n_modelos'] - 1}")

    n_images = len(list((root / "adicional").glob("*.png")))
    lines = (root / "SAMUEL_WILKER_BERNARDO_RESULTS.txt").read_text().split("\n")
    if lines[-1] != "" or len(lines) - 1 != n_images:
        errors.append(f"RESULTS.txt tem {len(lines) - 1} linhas para {n_images} imagens (ou não termina em quebra de linha)")
    if any(line not in set("0123456789") for line in lines[:-1]):
        errors.append("RESULTS.txt tem linha que não é um único dígito de 0 a 9")
    return errors


if __name__ == "__main__":
    errors = check(Path(sys.argv[1] if len(sys.argv) > 1 else "smoke"))
    for e in errors:
        print("ERRO:", e)
    print("ok" if not errors else f"{len(errors)} erro(s)")
    sys.exit(1 if errors else 0)
