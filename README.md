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

Cada experimento muda uma coisa em relação a outro, indicado entre parênteses. A coluna "Aula" indica de
qual material da disciplina veio a ideia.

| # | Experimento | Mudança | Aula |
|---|---|---|---|
| 1 | `linear` | classificador linear 3072 → 10, loss softmax | loss_function |
| 2 | `linear_svm` | loss SVM multiclasse (1) | loss_function |
| 3 | `mlp_baseline` | MLP 512-256, ReLU, Adam lr=1e-3, batch 64 | aula-2 |
| 4 | `mlp_lr_1e-4` | lr 1e-4 (3) | backpropagation |
| 5 | `mlp_lr_1e-2` | lr 1e-2 (3) | backpropagation |
| 6 | `mlp_batch_32` | batch 32 (3) | backpropagation |
| 7 | `mlp_batch_256` | batch 256 (3) | backpropagation |
| 8 | `mlp_sgd` | SGD, lr 0,01 (3) | backpropagation |
| 9 | `mlp_sgd_momentum` | SGD + momentum 0,9, lr 0,01 (3) | backpropagation |
| 10 | `mlp_100` | uma camada oculta de 100 (3) | aula-2 |
| 11 | `mlp_leaky_relu` | Leaky ReLU (3) | aula-2 |
| 12 | `mlp_tanh` | tanh (3) | aula-2 |
| 13 | `mlp_sigmoid` | sigmóide (3) | aula-2 |
| 14 | `mlp_sem_norm` | sem normalização, só /255 (3) | aula-2 |
| 15 | `mlp_so_media` | só subtrai a média por canal (3) | aula-2 |
| 16 | `mlp_cinza` | tons de cinza, 1024 entradas (3) | aula-2 |
| 17 | `mlp_l2` | regularização L2, λ = 1e-4 (3) | loss_function |
| 18 | `mlp_dropout` | Dropout 0,3 (3) | aula-2 |
| 19 | `mlp_dropout_0.5` | Dropout 0,5 (18) | aula-2 |
| 20 | `mlp_bn_dropout` | + BatchNorm (18) | |
| 21 | `mlp_deep` | camadas 1024-512-256 (20) | |
| 22 | `cnn` | CNN 3 blocos 32/64/128 filtros | |
| 23 | `cnn_augment` | + data augmentation (22) | |
| 24 | `cnn_wide` | filtros 64/128/256 (23) | |
| 25 | `cnn_wd` | weight decay 0.05 (23) | |
| 26 | `cnn_30ep` | 30 épocas (23) | |
| 27 | `cnn_ls` | label smoothing 0.1 (23) | |
| 28 | `cnn_best` | (23) + as mudanças de 24–27 que melhoraram a validação | |

A loss de validação é sempre a entropia cruzada, para comparar todos os experimentos; o `linear_svm` também
reporta a loss SVM (`val_loss_svm`). A loss SVM segue a fórmula da aula:
L_i = Σ_{j≠y_i} max(0, s_j − s_{y_i} + 1).

## Avaliações complementares

- **Checagem do gradiente** (backpropagation): `torch.autograd.gradcheck` compara o gradiente do autograd com o
  numérico numa MLP pequena.
- **Baseline kNN** (Lecture 11): kNN nos pixels normalizados, com K ∈ {1, 3, 5, 10, 20} escolhido pela validação
  e teste só no K escolhido. Serve de referência não neural.
- **Random subsampling** (Lecture 11): os 3 melhores na validação são treinados em mais 5 divisões treino/validação,
  com média ± desvio. A escolha do modelo final continua sendo pela divisão fixa (semente 42).
- **Análise de erros do modelo final** (Lecture 11): matriz de confusão no teste, precisão/recall/F1 por dígito e
  exemplos de erros.

Saídas em `results/`:
- `results.csv`: loss e acurácia de treino e validação, acurácia de teste e a melhor época de cada experimento.
- `<experimento>.png`: curvas de loss e acurácia.
- `comparacao_validacao.png`: acurácia de validação dos principais modelos.
- `taxas_aprendizado.png`: loss de treino e validação com lr 1e-4, 1e-3 e 1e-2.
- `knn.csv`: acurácia de validação por K e de teste no K escolhido.
- `subsampling.csv` e `subsampling/`: média ± desvio dos 3 melhores em 5 divisões.
- `confusao.png`, `metricas_por_classe.csv` e `erros.png`: análise de erros do modelo final.

## Modelo final

No fim do notebook, o experimento com a **maior acurácia de validação** é retreinado com todo o
`split="train"` (treino + validação), com a mesma config e semente, parando na época da sua melhor validação
(mesmo agendamento de LR do experimento).

- `weights/best.pt`: pesos, config, média e desvio da normalização (usados na predição).
- `results/final.json`: arquitetura, número de parâmetros, acurácia de validação do experimento e acurácia de teste.

Se `weights/best.pt` já existe com a mesma config, ele é carregado em vez de retreinado. Ao recarregar, o
notebook confere que os pesos salvos reproduzem a acurácia de teste.

## Testes e CI

O notebook tem um modo de teste, ligado por `SVHN_SMOKE=1`: usa dados aleatórios no formato do SVHN (sem
baixar o dataset), 2 épocas e grava tudo em `smoke/` (apagado a cada execução, para sempre treinar de novo), sem tocar em
`results/` nem em `weights/`. Sem a
variável, o notebook roda normalmente.

O GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) executa o notebook inteiro nesse modo
em cada PR e em cada push na `main`, e [`tests/check_smoke.py`](tests/check_smoke.py) confere as saídas:
28 experimentos, acurácias entre 0 e 1, losses positivas, curvas, `final.json`, pesos salvos, kNN,
random subsampling e análise de erros. O notebook
executado fica disponível como artefato do job.

Para rodar localmente (cerca de 1 min):

```bash
pip install -r requirements.txt nbconvert ipykernel
SVHN_SMOKE=1 jupyter nbconvert --to notebook --execute svhn.ipynb --output smoke-executed.ipynb
python tests/check_smoke.py smoke
```

## Entregáveis

- [x] Código: `svhn.ipynb`
- [ ] Relatório em PDF
- [ ] Pesos: `weights/best.pt` (gerado pelo notebook no Colab)
- [ ] `SAMUEL_WILKER_BERNARDO_RESULTS.txt`
