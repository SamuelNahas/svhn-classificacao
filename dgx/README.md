# Experimentos nas DGX Spark (#20 e #22)

**Solução atual (13/11):** 3 WRN-28-2 + RandAugment + shift12, 97,57% de validação e 98,20% de teste.
Detalhes em [Solução atual: rodada 5b](#solução-atual-rodada-5b). As seções anteriores a ela registram a pt4 de 10/11.

Busca de mais acurácia com duas NVIDIA DGX Spark (GB10). O plano
de cada rodada foi registrado em [`PLANO.md`](PLANO.md) **antes** de rodar, e a pesquisa bibliográfica que
motivou a rodada 4 está em [`PESQUISA.md`](PESQUISA.md). Os resultados ficam em [`../results/dgx/`](../results/dgx/),
uma pasta por experimento, e a tabela com todos os treinos está em [`resumo.csv`](../results/dgx/resumo.csv).

**Resultado da pt4 (anterior):** a solução é um **ensemble de 20 CNNs `cnn_best` de 60 épocas (sementes 42–61) com TTA
shift12**. Ela tem **97,52% de validação**, contra 97,31% da solução do PR #21, e **97,89% no teste**, contra
97,63%. O teste foi lido uma única vez, já com a solução fechada. O erro de teste cai de 2,37% para 2,11%.

## Protocolo
- O mesmo do notebook. A divisão é a da semente 42: 65.932 imagens de treino e 7.325 de validação. Os índices
  ficam em `results/dgx/val_indices.npy` e os rótulos em `val_labels.npy`, e o script confere que a divisão é a
  mesma. A normalização é a da divisão 42. O treino usa augment, cutout, AdamW e LR cosseno por época, e a
  melhor época é escolhida pela validação.
- Tudo foi escolhido **só pela validação**. O teste é calculado em `test_probs.npy` para não precisar de
  inferência depois, mas não entrou em nenhum `summary.json` nem em nenhuma escolha. Ele foi lido uma vez, na
  solução fechada (`results/dgx/final/test_acc.txt`). O conjunto do professor só foi usado para o RESULTS.txt.
- Uma diferença em relação ao notebook: para economizar tempo, a loss e a acurácia de **treino** de cada época
  são medidas numa amostra fixa de 7.325 imagens do treino (`TRAIN_EVAL_N`). A validação é sempre completa.

## Infraestrutura
- Máquinas: duas DGX Spark, acessadas via NVIDIA Sync. GPU GB10 (sm_121), com 121 GB de memória unificada.
- Container: `nvcr.io/nvidia/pytorch:26.08-py3`, com torch 2.14, CUDA 13.4 e cuDNN 9.25. A imagem
  `nvcr.io/nvidia/vllm:26.08-py3` **não serve para treinar**: o cuDNN dela não tem o backward das
  convoluções. Sem cuDNN, a mesma época levava 27 s, contra 8,3 s.
- Código: [`svhn_dgx.py`](svhn_dgx.py), com `run`, `worker` (fila de configs) e `infer` (TTA), e
  [`final_ensemble.py`](final_ensemble.py), que monta a solução final. Nas máquinas fica em `~/svhn-dgx/`
  (`code/`, `data/`, as filas `q-*` e `runs/` com os checkpoints).
- Precisão: **bf16**. A `cnn_best` deu 96,66% em bf16 e 96,56% em fp32, e o bf16 é 1,85× mais rápido. Batch
  64 em tudo.
- Paralelismo: 3 ou 4 treinos por GPU. Com 4 treinos, cada um leva 10,6 s por época, contra 8,3 s de um treino
  sozinho, o que dá ~3× mais trabalho pelo mesmo tempo.
- No total, foram 119 treinos e ~27 h de GPU somadas, em duas janelas: 06/11 à tarde e 10/11 de 08:10 a 11:25.

### Incidentes
- 06/11, 15:27: os primeiros treinos do ensemble caíram ao retomar do checkpoint, porque o estado do RNG era
  carregado na GPU. Corrigido em `svhn_dgx.py`. Esses treinos tinham começado na imagem sem cuDNN e foram
  descartados (ficam em `~/svhn-dgx/runs-descartados/`).
- 10/11, 10:38: os retreinos `zf60_s62–65` (para k = 25/30) foram interrompidos e foram para `q-hold/` quando a
  curva mostrou que k = 25/30 não ganham. Ficam em `results/dgx/` só com `config.json` e `log.txt`.

## Resultados (validação, divisão 42, salvo indicação)

### Rodada 1 — modelos individuais (triagem, semente 42)
Referência: `t01_ref_bf16`, a `cnn_best` em bf16, com 96,66%.

| Experimento | Mudança | Validação | Δ | Loss val | Parâm. | s/época* |
|---|---|---|---|---|---|---|
| t01_ref_bf16 | referência (`cnn_best`) | 96,66% | — | 0,1902 | 1,01 M | 8,1 |
| t02_ref_fp32 | fp32 | 96,56% | −0,10 | 0,1920 | 1,01 M | 14,9 |
| t12_ep60 | 60 épocas | 96,66% | 0,00 | 0,1953 | 1,01 M | 8,3 |
| **t03_wrn16_4** | **Wide ResNet 16-4, dropout 0,3, mesma receita** | **97,08%** | **+0,42** | 0,1977 | 2,75 M | 24,2 |
| t04_wrn16_4_sgd | WRN-16-4 com SGD Nesterov lr 0,1, wd 5e-4 | 97,01% | +0,35 | 0,1884 | 2,75 M | 21,1 |
| t05_wrn22_1 | ResNet pequena (WRN-22-1) | 96,63% | −0,03 | 0,1968 | 0,27 M | 8,7 |
| t06_ema | EMA 0,999 (BN em média junto) | 96,68% | +0,02 | 0,1982 | 1,01 M | 15,9 |
| t07_color | brilho/contraste ±20% | 96,83% | +0,17 | 0,1927 | 1,01 M | 6,8 |
| t08_geo05 | augment geométrico ×0,5 | 96,60% | −0,06 | 0,1955 | 1,01 M | 6,5 |
| t09_geo15 | augment geométrico ×1,5 | 96,42% | −0,24 | 0,2111 | 1,01 M | 6,6 |
| t10_cut12 | cutout 12×12 | 96,79% | +0,13 | 0,1887 | 1,01 M | 7,5 |
| t11_cutp05 | cutout com p = 0,5 | 96,59% | −0,07 | 0,1958 | 1,01 M | 7,8 |
| t15_wrn16_4_color | WRN-16-4 + brilho/contraste | 96,83% | +0,17 | 0,1951 | 2,75 M | 31,3 |
| t16_wrn16_4_ep60 | WRN-16-4, 60 épocas | 96,85% | +0,19 | 0,2024 | 2,75 M | 28,8 |
| t17_wrn16_8 | WRN-16-8 | 96,91% | +0,25 | 0,1986 | 10,96 M | 60,3 |
| t18_wrn28_4 | WRN-28-4 | 96,86% | +0,20 | 0,1884 | 5,85 M | 51,4 |

\* com 3 ou 4 treinos dividindo a GPU.

- A WRN-16-4 é a única mudança grande. Redes maiores (WRN-16-8 e WRN-28-4) não passam dela, e somar
  brilho/contraste a ela piora.
- EMA, cutout com p = 0,5 e mexer na intensidade do augment geométrico não ajudam.

### Rodada 1b — WRN-16-4 com 30 contra 60 épocas (pareada)

| | semente 42 | 43 | 44 | Média |
|---|---|---|---|---|
| 30 épocas | 97,08% | 97,00% | 96,87% | **96,98%** |
| 60 épocas | 96,85% | 96,86% | 96,86% | 96,86% |

Pela regra registrada, a WRN ficou com **30 épocas** (ganha na média e em 2 dos 3 pares).

### Rodadas 2 e 3 — ensembles e TTA
Média simples do softmax, membros em ordem fixa de sementes. O TTA desloca a imagem: shift1 = ±1 px, shift2 =
±2 px e shift12 = os dois (8 cópias), sempre sem espelhamento.

| Estratégia | TTA | k = 1 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|---|
| CNN de 60 épocas | nenhum | 96,60% | 97,39% | 97,41% | 97,46% | 97,47% | 97,45% | 97,42% |
| CNN de 60 épocas | shift12 | 96,87% | 97,38% | 97,38% | 97,47% | **97,52%** | 97,49% | 97,47% |
| WRN-16-4, 30 épocas | shift12 | 97,12% | 97,38% | 97,28% | — | — | — | — |
| Heterogêneo (WRN e CNN intercaladas) | shift12 | — | 97,38% | 97,35% | — | 97,46% | — | — |

A curva completa (k de 1 a 30, os quatro TTAs e a loss) está em
[`curvas_ensemble.csv`](../results/dgx/curvas_ensemble.csv) e em [`ensembles.png`](../results/dgx/ensembles.png).

- **A WRN sozinha é melhor, mas o ensemble dela não é.** No k = 10, a WRN homogênea deu 97,28% (loss 0,188),
  contra 97,45% da CNN de 60 épocas (loss 0,176). Pela regra de parada das 09:05, a WRN parou em 10 membros.
  O mais provável é que as WRNs errem nas mesmas imagens, o que tira o ganho de juntar várias.
- **O heterogêneo não ganhou** da CNN homogênea no k = 20, em nenhum TTA (o melhor dele foi 97,46%). Só no
  k = 5 com shift2 ele passou à frente, por 0,03.
- **A CNN de 60 épocas estabiliza em k = 20.** De 20 para 25 e 30, a validação não sobe em nenhum TTA. Pela regra
  da rodada 3, o tamanho fica em 20 e o TTA é o melhor nesse k: shift12.
- Os membros individuais da CNN de 60 épocas têm média de 96,72% (desvio 0,11, 30 sementes).

### Rodada 3 — confirmação em outra divisão (`split_seed` 43)
A mesma pergunta numa divisão que não guiou nenhuma escolha: 60 épocas contra 30 (a receita do PR #21), com 10
sementes cada, sem TTA.

| | k = 1 | 5 | 10 | Média individual |
|---|---|---|---|---|
| 30 épocas | 96,60% | 97,15% | 97,24% | 96,68% |
| 60 épocas | 96,87% | 97,19% | **97,38%** | 96,78% |

As 60 épocas ganham também na outra divisão: +0,10 por modelo e +0,14 no ensemble de 10.

### Rodada 4 — augment automático (pareada com `e60_val_s42–44`)
Troca só o augment geométrico por uma política por imagem; o resto da CNN de 60 épocas fica igual.

| Política | s42 | s43 | s44 | Média | Δ | Loss val (média) |
|---|---|---|---|---|---|---|
| e60 (referência) | 96,60% | 96,96% | 96,81% | 96,79% | — | 0,195 |
| RandAugment N = 3, M = 9 (`a_ra`) | 96,89% | 97,04% | 96,79% | 96,91% | +0,12 | **0,186** |
| AutoAugment SVHN (`a_aa`) | 96,97% | 96,79% | 96,81% | 96,86% | +0,07 | 0,191 |
| TrivialAugment (`a_ta`) | 96,78% | 96,78% | 96,49% | 96,68% | −0,11 | 0,196 |

Nenhuma política passou da regra (pelo menos +0,15 na média e vitória na maioria dos pares), então fecha-se
com o ensemble atual. A RandAugment foi a que chegou mais perto e teve a menor loss nas três sementes. A
receita completa do artigo do RandAugment (WRN-28-2, SGD, cutout 16, 200 épocas) é um experimento futuro na
#22.

## Solução anterior pt4
- **Membros:** a `cnn_best` com 60 épocas e as sementes 42 a 61, cada uma retreinada em todo o
  `split="train"` (`zf60_s<semente>`). Cada membro usa os pesos da época em que o membro de validação da mesma
  semente (`e60_val_s<semente>`) teve a melhor validação, entre as épocas 40 e 60. A procedência de cada membro
  está em [`membros.json`](../results/dgx/final/membros.json).
- **Predição:** média do softmax dos 20 membros na imagem e nas 8 cópias de shift12.
- **Arquivos:** os pesos estão em `weights/pt4/` (20 × 4 MB, com a config e a época). As probabilidades do teste
  e do conjunto adicional calculadas pelo script, em bf16, estão em `results/dgx/final/`.
- **Predições:** o `SAMUEL_WILKER_BERNARDO_RESULTS.txt` é gerado pelo notebook, que é o código entregue: a seção
  "Solução final pt4" do [`svhn.ipynb`](../svhn.ipynb) carrega `weights/pt4/` e classifica o teste e o conjunto
  adicional. No Mac (MPS, fp32), o teste dá 97,89%, como no script. As predições do conjunto adicional diferem das
  do script (bf16) em 15 das 203.856 imagens.

## O que ficou para depois
- **Receita do RandAugment**: executada na #22 em 13/11, ver solução atual abaixo.
- **Mais épocas na CNN (100 e 200):** um teste pareado curto, também registrado na #22. A expectativa é baixa:
  30 → 60 rendeu só +0,10 por modelo, a WRN piorou com 60 épocas e o ensemble já estabilizou em k = 20.
- Os retreinos em `q-hold/` (`wf30_s46–51` e `zf60_s62–71` na Spark-1, e cópias de `zf60_s56–61`, já feitos, na
  Spark-2) não são necessários para a solução escolhida e podem ser descartados.

## Solução atual: rodada 5b

Em 13/11, a receita WRN-28-2 + RandAugment substituiu a solução pt4 descrita acima.

| Solução | Validação | Teste |
|---|---|---|
| CNN de 20 membros + shift12 (pt4, anterior) | 7143/7325 = 97,52% | 97,89% |
| WRN-28-2 + RA, 10 membros (só validação) | 7162/7325 = 97,77% | não avaliado |
| **WRN-28-2 + RA, 3 membros (entregue)** | **7147/7325 = 97,57%** | **98,20%** |

- Receita: WRN-28-2 (1.467.610 parâmetros), RA N=3/M=9, cutout 16, SGD Nesterov lr=0,005,
  momentum=0,9, weight decay=0,005, batch=128 e cosseno em 200 épocas. Sem label smoothing,
  Dropout ou augment geométrico adicional. Treino em bf16 nas Sparks; fechamento em fp32 no Mac/MPS.
- A solução entregue usa as sementes 42, 47 e 48, com média simples das probabilidades dos três membros.
- Retreinos em todo o split train; pesos escolhidos nas épocas 196, 188 e 181 pela validação da respectiva
  semente. O TTA é fixo em shift12. Mean Teacher foi descartado pela regra de triagem.
- A solução foi escolhida pela validação e avaliada no teste após o fechamento.
- Artefatos: [`final_r5b/`](../results/dgx/final_r5b/) contém os três pesos em `membros/`, as probabilidades,
  `test_acc.txt`, `membros.json` e `fechamento.json`. As pastas `r_wrn282_ra_s42–51` contêm os resultados
  de validação. O RESULTS entregue contém 203.856 classes, com 2.407 mudanças em relação à pt4.
- O notebook reproduz a pt4. Para gerar o RESULTS atual sem reavaliar o teste, execute
  o comando de argmax de `probs_extra.npy` no README principal.

O comando usado no fechamento foi (com os dados preparados em `data/` e os checkpoints por treino em `runs/`):

```bash
SVHN_DATA=data/svhn.npz python dgx/final_ensemble.py \
  --runs runs --extra data/adicional.npy --out final_r5b \
  --seeds 42 47 48 --val r_wrn282_ra --final rf_wrn282_ra --epochs 200
```

`svhn.npz` contém `x_train_full`, `y_train_full`, `x_test`, `y_test` e `perm42` (permutação PyTorch
com semente 42). Imagens uint8 NCHW, rótulos 0–9; `adicional.npy` usa a ordem numérica dos nomes das imagens.
Os pesos empacotados da entrega estão em `final_r5b/membros/`; não são os checkpoints brutos usados por esse comando.
