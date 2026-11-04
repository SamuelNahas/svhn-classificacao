# Classificação de Dígitos — SVHN

Trabalho de Redes Neurais (Samuel, Wilker e Bernardo): classificador linear, MLP, CNN e experimentos
de melhoria no dataset [SVHN](https://pytorch.org/vision/stable/generated/torchvision.datasets.SVHN.html), em PyTorch.

Os termos usados no projeto estão em [GLOSSARY.md](GLOSSARY.md).

## Estrutura do repositório

| Caminho | Conteúdo |
|---|---|
| [`svhn.ipynb`](svhn.ipynb) | Todo o código: dados, modelos, experimentos, resultados e modelo final |
| [`GLOSSARY.md`](GLOSSARY.md) | Vocabulário do projeto (conjuntos de dados, experimento, modelo final etc.) |
| `requirements.txt` | Dependências para rodar fora do Colab |
| `results/` | Gerado pelo notebook: `results.csv`, curvas por experimento, `comparacao_validacao.png` e `final.json` |
| `weights/best.pt` | Gerado pelo notebook: pesos do modelo final |
| `data/` | Download do SVHN (ignorado pelo git) |

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

## Modelo final

No fim do notebook, o experimento com a **maior acurácia de validação** é retreinado com todo o
`split="train"` (treino + validação), com a mesma config e semente, parando na época da sua melhor validação
(mesmo agendamento de LR do experimento).

- `weights/best.pt`: pesos, config, média e desvio da normalização (usados na predição).
- `results/final.json`: arquitetura, número de parâmetros, acurácia de validação do experimento e acurácia de teste.

Se `weights/best.pt` já existe com a mesma config, ele é carregado em vez de retreinado. Ao recarregar, o
notebook confere que os pesos salvos reproduzem a acurácia de teste.

## Decisões de implementação

- **Dados na GPU:** o dataset inteiro (~300 MB em `uint8`) vai para a GPU e os batches são montados lá, sem
  `DataLoader`. O data augmentation (rotação ±8°, escala ±10%, translação ±8%) também roda na GPU, por imagem.
- **Reprodutibilidade:** semente fixa (`SEED = 42`) em cada treino e cuDNN determinístico.
- **Cache por config:** cada experimento salva em `results/<nome>.json` a config completa (arquitetura,
  hiperparâmetros, épocas, semente). Ao rodar de novo, só é carregado se a config for idêntica; qualquer mudança retreina.
- **Configs como dados:** experimentos são descritos por `arch` + `arch_args` (sem lambdas), para poderem ser comparados com o salvo.
- **`cnn_best` automático:** combina as variações de CNN (11–14) que superaram a `cnn_augment` na validação; se nenhuma
  superar, reaproveita a `cnn_augment` sem retreinar.
- **Checkpoint seguro:** `weights/best.pt` é gravado com troca atômica; se estiver ausente ou corrompido, o modelo é retreinado.
- **Métricas comparáveis:** a loss de validação é sempre a entropia cruzada pura (mesmo com label smoothing), e as
  métricas de treino são medidas em modo de avaliação (sem dropout nem augmentation).

## Organização do trabalho

- **Board:** [Trabalho SVHN](https://github.com/users/SamuelNahas/projects/2), com as colunas Backlog → Sprint Backlog →
  Fazendo → Verificação → Validação → Pronto.
- **Issues:** a [#1](../../issues/1) reúne as decisões do trabalho; cada entregável é uma sub-issue:

  | Issue | Entregável | Status |
  |---|---|---|
  | [#2](../../issues/2) | Notebook + experimentos | Concluída (PR #6) |
  | [#3](../../issues/3) | Modelo final + pesos | Em revisão (PR #7) |
  | [#4](../../issues/4) | Predições do conjunto adicional | A fazer |
  | [#5](../../issues/5) | Relatório em LaTeX | A fazer |
  | [#8](../../issues/8) | Atualizar este README | Em andamento |

- **Branches:** uma por issue (`<número>-<descrição>`), com PR que fecha a issue. Quando uma issue depende de outra
  ainda aberta, o PR é empilhado sobre a branch da anterior.
- **Commits:** só os integrantes do grupo constam como autores.

## Histórico de mudanças

### PR #6 — Notebook + experimentos (#2)
- `svhn.ipynb` autocontido para o Colab substitui o `svhn.py`.
- 15 experimentos, uma mudança por vez; divisão 90/10 fixa, normalização só no treino, teste avaliado na melhor época de validação.
- Revisão aplicada: cache por config, cuDNN determinístico, loss e acertos acumulados na GPU, média e desvio numa só passada, `cnn_best` sem retreino desnecessário.
- `GLOSSARY.md` criado.

### PR #7 — Modelo final (#3)
- `train_final` retreina o melhor experimento em treino + validação, parando na melhor época com o mesmo agendamento de LR.
- `weights/best.pt` com pesos, config e normalização; carregado se já existir com a mesma config.
- Conferência de que os pesos recarregados reproduzem a acurácia de teste; `results/final.json` para o relatório.
- **Correção:** os acumuladores de loss e acertos apontavam para o mesmo tensor e geravam acurácias acima de 1. O commit
  entrou depois do merge do PR #6, então a correção chega à `main` por este PR.

### PR #9 — README (#8)
- Estrutura do repositório, decisões de implementação, organização do trabalho e este histórico.

## Entregáveis

- [x] Código: `svhn.ipynb`
- [ ] Pesos: `weights/best.pt` (gerado pelo notebook no Colab, issue #3)
- [ ] `SAMUEL_WILKER_BERNARDO_RESULTS.txt` (issue #4)
- [ ] Relatório em PDF (issue #5)

## Próximos passos

1. Rodar o `svhn.ipynb` no Colab com GPU e commitar `results/` e `weights/best.pt`.
2. Baixar o conjunto adicional e gerar as predições (#4).
3. Escrever o relatório com os números de `results/` (#5).
