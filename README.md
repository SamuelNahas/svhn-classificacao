# Classificação de Dígitos — SVHN

Trabalho de Redes Neurais (Samuel, Wilker e Bernardo): classificador linear, MLP, CNN e experimentos
de melhoria no dataset [SVHN](https://pytorch.org/vision/stable/generated/torchvision.datasets.SVHN.html), em PyTorch.

Os termos usados no projeto estão em [GLOSSARY.md](GLOSSARY.md).

## Como rodar

O código está todo em [`svhn.ipynb`](svhn.ipynb).

1. No Colab: `Arquivo → Fazer upload de notebook` e escolha o `svhn.ipynb`.
2. `Ambiente de execução → Alterar o tipo de ambiente → GPU`.
3. `Executar tudo`. O dataset é baixado automaticamente.

Os experimentos já concluídos ficam salvos em `results/` e não são retreinados quando o notebook roda de novo.
Para não perder nada se o Colab desconectar, descomente a célula que monta o Google Drive.

Para rodar localmente: `pip install -r requirements.txt jupyter`. Funciona em CUDA, na GPU do Mac (MPS) ou em CPU.

## Protocolo

- `split="train"` é dividido em 90% treino e 10% validação, com semente fixa (`SEED = 42`).
- A normalização usa a média e o desvio calculados só no conjunto de treino.
- Cada experimento é avaliado na época com melhor acurácia de **validação**.
- `split="test"` só informa a acurácia; não participa de nenhuma escolha.
- O conjunto adicional do professor só é usado depois que o modelo final está definido.

## Experimentos

Cada experimento muda uma coisa em relação a outro, indicado entre parênteses.

| # | Experimento | Mudança |
|---|---|---|
| 1 | `linear` | classificador linear 3072 → 10 |
| 2 | `mlp_baseline` | MLP 512-256, ReLU, Adam lr=1e-3, batch 64 |
| 3 | `mlp_lr_1e-4` | lr 1e-4 (2) |
| 4 | `mlp_lr_1e-2` | lr 1e-2 (2) |
| 5 | `mlp_batch_256` | batch 256 (2) |
| 6 | `mlp_dropout` | + Dropout 0.3 (2) |
| 7 | `mlp_bn_dropout` | + BatchNorm (6) |
| 8 | `mlp_deep` | camadas 1024-512-256 (7) |
| 9 | `cnn` | CNN 3 blocos 32/64/128 filtros |
| 10 | `cnn_augment` | + data augmentation (9) |
| 11 | `cnn_wide` | filtros 64/128/256 (10) |
| 12 | `cnn_wd` | weight decay 0.05 (10) |
| 13 | `cnn_30ep` | 30 épocas (10) |
| 14 | `cnn_ls` | label smoothing 0.1 (10) |
| 15 | `cnn_best` | (10) + as mudanças de 11–14 que melhoraram a validação |

Saídas em `results/`:
- `results.csv`: loss e acurácia de treino e validação, acurácia de teste e a melhor época de cada experimento.
- `<experimento>.png`: curvas de loss e acurácia.
- `comparacao_validacao.png`: acurácia de validação dos principais modelos.

## Entregáveis

- [x] Código: `svhn.ipynb`
- [ ] Relatório em PDF
- [ ] Pesos: `weights/best.pt`
- [ ] `SAMUEL_WILKER_BERNARDO_RESULTS.txt`
