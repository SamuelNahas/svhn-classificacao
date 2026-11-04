# Classificação de Dígitos — SVHN

Trabalho de Redes Neurais: classificador linear, MLP e experimentos de melhoria no dataset
[SVHN](https://pytorch.org/vision/stable/generated/torchvision.datasets.SVHN.html), em PyTorch.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

O dataset é baixado automaticamente para `data/`.

## Protocolo

- `split="train"` é dividido 90/10 em treino/validação com semente fixa (`SEED = 42`).
- A validação escolhe a melhor época de cada experimento e o melhor modelo.
- `split="test"` é usado só para reportar a acurácia final de cada experimento.
- O modelo final é retreinado com todo o `split="train"`.
- O conjunto adicional do professor só é usado no passo `predict`, depois da escolha do modelo.

## Reproduzindo

```bash
# 1. Experimentos -> results/results.csv + curvas results/<nome>.png
python svhn.py experiments --epochs 15

# 2. Modelo final (o melhor na validação) -> weights/best.pt
python svhn.py final --model cnn_augment --epochs 15

# 3. Predições do conjunto adicional (imagens ordenadas pelo nome do arquivo)
python svhn.py predict --images caminho/das/imagens --out NOME1_NOME2_RESULTS.txt
```

## Experimentos

| # | Modificação | O que muda |
|---|---|---|
| 1 | `linear` | Classificador linear (3072 → 10) |
| 2 | `mlp_baseline` | MLP 3072-512-256-10, ReLU, Adam lr=1e-3, batch 64 |
| 3 | `mlp_lr_1e-4` | taxa de aprendizado menor |
| 4 | `mlp_lr_1e-2` | taxa de aprendizado maior |
| 5 | `mlp_batch_256` | batch maior |
| 6 | `mlp_dropout` | + Dropout 0.3 |
| 7 | `mlp_bn_dropout` | + BatchNorm |
| 8 | `mlp_deep` | mais uma camada (1024-512-256) |
| 9 | `cnn` | rede convolucional (3 blocos 32/64/128 filtros) |
| 10 | `cnn_augment` | + data augmentation (rotação/translação/escala) |

Os resultados (acurácia de treino, validação e teste) ficam em `results/results.csv`.

## Entregáveis

- [ ] Código: `svhn.py`
- [ ] Relatório em PDF
- [ ] Pesos: `weights/best.pt`
- [ ] `NOME1_NOME2_RESULTS.txt`
