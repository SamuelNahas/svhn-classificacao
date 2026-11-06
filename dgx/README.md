# Bateria pt4 (#20) nas DGX Spark — estado em 06/11/2026, 16:05

Busca de mais acurácia com as duas NVIDIA DGX Spark (GB10) do projeto p2p (Lenovo Partners/ClickTI). O plano
de cada rodada é registrado antes de rodar em [`PLANO.md`](PLANO.md). Resultados em
[`../results/dgx/`](../results/dgx/), uma pasta por experimento. **Nada aqui mudou ainda a solução entregue**
(ensemble de 5 `cnn_best` + TTA, 97,31% de validação, PR #21).

## Protocolo
- O mesmo do notebook. A divisão é a da semente 42: 65.932 imagens de treino e 7.325 de validação. Os índices
  ficam em `results/dgx/val_indices.npy` e os rótulos em `val_labels.npy`, e o script confere que a divisão é a
  mesma. A normalização é a da divisão 42. O treino usa augment, cutout, AdamW e LR cosseno por época, e a
  melhor época é escolhida pela validação.
- Tudo é escolhido **só pela validação**. O teste é calculado em `test_probs.npy` para não precisar de
  inferência depois, mas **não foi lido** e não está em nenhum `summary.json`; só será usado na solução
  fechada. O conjunto do professor não foi usado.
- Uma diferença em relação ao notebook: para economizar tempo, a loss e a acurácia de **treino** de cada época
  são medidas numa amostra fixa de 7.325 imagens do treino (`TRAIN_EVAL_N`). A validação é sempre completa.

## Infraestrutura
- Máquinas: `Spark-1-new` (spark-797f) e `Spark-2-new` (spark-2b14), via NVIDIA Sync. GPU GB10 (sm_121), com
  121 GB de memória unificada.
- Container: `nvcr.io/nvidia/pytorch:26.08-py3`, com torch 2.14, CUDA 13.4 e cuDNN 9.25. A imagem
  `vllm:26.08` que já estava nas máquinas **não serve para treinar**: o cuDNN dela não tem o backward das
  convoluções. Sem cuDNN, a mesma época levava 27 s, contra 8,3 s.
- Código: [`svhn_dgx.py`](svhn_dgx.py), com `run`, `worker` (fila de configs) e `infer` (TTA). Nas
  máquinas fica em `~/svhn-dgx/` (`code/`, `data/svhn.npz`, `q-ens/` e `q-tri/` com as filas, e `runs/` com
  os checkpoints).
- Precisão: **bf16**. A `cnn_best` deu 96,66% em bf16 e 96,56% em fp32, contra 96,68% no Mac, e o bf16 é
  1,85× mais rápido. Batch 64 em tudo.
- Paralelismo: 3 ou 4 treinos por GPU. Com 4 treinos, cada um leva 10,6 s por época, contra 8,3 s de um treino
  sozinho, o que dá ~3× mais trabalho pelo mesmo tempo.

### Produção (projeto p2p)
- ~15:15: as 4 vLLM (`spark-llm`, `spark-embeddings`, `spark-reranker` e `spark-llm-replica`) foram paradas com
  `docker stop`, sem remover nada, depois do backup do estado em `~/svhn-dgx/prod-backup-20261106/` em cada
  máquina (inspect, compose, `.env`, saúde e `restaurar.sh`). O `postgres` nunca foi tocado.
- 15:34–15:38: restauradas com `restaurar.sh`. A saúde foi conferida: `/health` 200 nas portas 8000, 8001 e
  8002, o chat respondeu "ok" nas duas máquinas, e os embeddings e o reranker servem os seus modelos.
- 15:38–16:04: os treinos rodaram **ao lado** da produção, com ~25 GB livres por máquina.
- 16:04: os treinos foram parados e a produção ficou com a GPU só para ela. A saúde foi conferida de novo (ok).

### Incidentes
- Às 15:27, os primeiros treinos do ensemble caíram ao retomar do checkpoint: o estado do RNG era carregado na
  GPU. Corrigido em `svhn_dgx.py`. Esses treinos começaram na imagem sem cuDNN e foram descartados (ficam em
  `~/svhn-dgx/runs-descartados/`).

## Resultados até agora (validação, semente 42 salvo indicação)

### Modelos individuais (triagem)
Referência: `t01_ref_bf16`, a `cnn_best` em bf16, com 96,66%.

| Experimento | Mudança | Validação | Δ | Loss val | Parâm. | s/época* |
|---|---|---|---|---|---|---|
| t01_ref_bf16 | referência (`cnn_best`) | 96,66% | — | 0,1902 | 1,01 M | 8,1 |
| t02_ref_fp32 | fp32 | 96,56% | −0,10 | 0,1920 | 1,01 M | 14,9 |
| **t03_wrn16_4** | **Wide ResNet 16-4, dropout 0,3, mesma receita** | **97,08%** | **+0,42** | 0,1977 | 2,75 M | 24,2 |
| t04_wrn16_4_sgd | WRN-16-4 com SGD Nesterov lr 0,1, wd 5e-4 | 97,01% | +0,35 | 0,1884 | 2,75 M | 21,1 |
| t05_wrn22_1 | ResNet pequena (WRN-22-1) | 96,63% | −0,03 | 0,1968 | 0,27 M | 8,7 |
| t06_ema | EMA 0,999 (BN em média junto) | 96,68% | +0,02 | 0,1982 | 1,01 M | 15,9 |
| t07_color | brilho/contraste ±20% | 96,83% | +0,17 | 0,1927 | 1,01 M | 6,8 |
| t08_geo05 | augment geométrico ×0,5 | 96,60% | −0,06 | 0,1955 | 1,01 M | 6,5 |
| t09_geo15 | augment geométrico ×1,5 | 96,42% | −0,24 | 0,2111 | 1,01 M | 6,6 |
| t10_cut12 | cutout 12×12 | 96,79% | +0,13 | 0,1887 | 1,01 M | 7,5 |
| t11_cutp05 | cutout com p = 0,5 | 96,59% | −0,07 | 0,1958 | 1,01 M | 7,8 |

\* com 3 ou 4 treinos dividindo a GPU.

Leitura, ainda com uma semente só:
- A WRN-16-4 é a única mudança grande (+0,42).
- Brilho/contraste (+0,17) e cutout 12×12 (+0,13) estão no limite do ruído entre sementes (~0,15).
- EMA, cutout com p = 0,5 e mexer na intensidade do augment geométrico não ajudam.
- A ResNet pequena empata com a CNN usando 1/4 dos parâmetros.

### Ensemble de 60 épocas da `cnn_best` (sementes 42–49, sem TTA)
Os membros individuais, com 60 épocas, ficaram entre 96,59% e 96,96%, com média de 96,79%, contra 96,66% com 30
épocas na semente 42.

| Membros | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Validação | 96,60% | 97,01% | 97,20% | 97,30% | **97,39%** | 97,32% | 97,37% | 97,38% |

Na mesma ordem de membros, o ensemble atual (30 épocas, Mac) deu 97,06% com 5 membros sem TTA e 97,31% com TTA.
O de 60 épocas, sem TTA, já passa a solução atual e estabiliza a partir de ~5 membros. A loss de validação com 8
membros é 0,1769. **Falta medir o TTA e confirmar**: essa curva é de uma ordem de sementes só.

## O que falta (em ordem)
1. **Retomar os treinos parciais.** Cada um tem um `ckpt_last.pt` nas DGX e retoma com o mesmo worker:
   - Spark-1: `e60_val_s50` (época 13/60), `e60_val_s51` (27/60), `w30_val_s43` (7/30) e `w30_val_s44` (5/30);
     depois `w60_val_s43` e `w60_val_s44`, que estão na fila.
   - Spark-2: `t12_ep60` (55/60), `t15_wrn16_4_color` (10/30) e `t16_wrn16_4_ep60` (9/60); depois
     `t17_wrn16_8` e `t18_wrn28_4`, que estão na fila.

   Para retomar, rode em cada máquina o mesmo `docker run ... worker --queue <fila> --runs runs --id <s1-wN ou
   s2-wN>` (os IDs `s1-w1`…`s1-w4` e `s2-w1`…`s2-w3` são donos dos treinos parciais).
2. **Confirmação pareada** "WRN-16-4 com 30 contra 60 épocas", nas sementes 42 a 44 (rodada 1b do `PLANO.md`).
3. **TTA na validação** (`infer --tta none|shift1|shift2|shift12`) para os candidatos, com o ganho registrado
   separado do ganho do ensemble.
4. **Ensemble homogêneo** da configuração confirmada, com 5, 10, 15 e 20 membros (25 e 30 se continuar subindo),
   e **ensemble heterogêneo** (WRN e CNN de 60 épocas intercaladas em ordem fixa). Para cada um: acurácia e loss
   de validação, parâmetros, tempo de treino e de inferência, armazenamento e sobreposição de erros.
5. Brilho/contraste e cutout 12×12 na arquitetura vencedora, com mais sementes antes de entrar na receita.
6. Confirmação do procedimento escolhido em outras divisões (só do split train).
7. **Fechamento:**
   - teste avaliado uma única vez na solução fechada;
   - retreino dos membros em todo o `split="train"` (`final: true`, salvando as épocas para parar na melhor
     época de validação de cada membro);
   - RESULTS.txt gerado automaticamente e conferido;
   - pesos, notebook e README atualizados.

   Se não houver ganho sustentado sobre 97,31%, a solução do PR #21 é mantida.
