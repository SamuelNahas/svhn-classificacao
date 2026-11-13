# Bateria pt4 (#20) nas DGX Spark — plano registrado antes de cada rodada

Protocolo comum (igual ao notebook): divisão fixa treino/validação da semente 42 (65.932 / 7.325), normalização
da divisão 42, escolha da melhor época, de configurações, de membros do ensemble e de TTA **só pela validação**.
O teste é calculado e gravado em `test_probs.npy`, mas não entra em `summary.json` nem em nenhuma escolha: só é
lido na avaliação final da solução fechada. O conjunto do professor só é usado para gerar o RESULTS.txt final.
Nada do split `extra` nem dados externos. Batch 64 em tudo, salvo quando o batch for o próprio experimento.

Referência (R) = `cnn_best` atual (#17): CNN 3 blocos 32/64/128 com 3 convoluções por bloco, augment (rotação
±8°, escala ±10%, translação ±8%), cutout 8×8, AdamW lr 1e-3, weight decay 0,05, label smoothing 0,1, 30 épocas
com LR cosseno, batch 64. No Mac (MPS): val 96,68%. Solução atual: ensemble de 5 sementes + TTA (val 97,31%).

## Rodada 0 — ambiente (antes de tudo)
- `r0_ref_fp32` e `r0_ref_bf16`: R em fp32 e em bf16 (semente 42). Decide a precisão do resto: bf16 só se a
  validação ficar dentro de ~0,15 ponto da fp32 e sem instabilidade na loss.

## Rodada 1 — modelos individuais (triagem, semente 42)
Cada linha muda uma coisa em relação à base indicada.

| Nome | Mudança | Base |
|---|---|---|
| r1_ref | R reproduzida na DGX | — |
| r1_ep60 / r1_ep100 | 60 / 100 épocas (cosseno ajustado ao total) | R |
| r1_wrn22_1 | ResNet pequena pré-ativada (WRN-22-1, ~0,27 M) com a receita de R | R |
| r1_wrn16_4 | Wide ResNet moderada (WRN-16-4, ~2,7 M) com a receita de R | R |
| r1_wrn16_4_sgd | WRN-16-4 com a receita usual da WRN: SGD Nesterov lr 0,1, momentum 0,9, wd 5e-4 | r1_wrn16_4 |
| r1_ema | EMA dos pesos (decaimento 0,999; estatísticas do BN também em média, `use_buffers=True`) | R |
| r1_color | brilho/contraste ±20% | R |
| r1_geo05 / r1_geo15 | augment geométrico com metade / 1,5× da intensidade | R |
| r1_cut12 | cutout 12×12 | R |
| r1_cutp05 | cutout com probabilidade 0,5 | R |

Depois: EMA e os ajustes de augment que ganharem também na melhor arquitetura, e as combinações promissoras
avaliadas explicitamente (melhorias isoladas podem não se somar). Candidatos promissores (≥ +0,15 ponto, que é
cerca de um desvio entre sementes no notebook) são repetidos com as sementes 43 e 44 antes de qualquer ensemble.

## Ensembles (registrado antes de ver resultados de ensemble)
- **Homogêneo:** a melhor configuração individual (média de 3 sementes), sementes 42, 43, …, 61 nessa ordem.
  Ensemble de 5 = sementes 42–46; 10 = 42–51; 15 = 42–56; 20 = 42–61. Extensão a 25/30 (62–71) só se 15 → 20
  ainda ganhar na validação.
- **Heterogêneo:** as duas ou três melhores configurações distintas, intercaladas em ordem fixa (A42, B42, C42,
  A43, B43, C43, …) até 20 membros.
- Média simples do softmax. Para cada tamanho: acurácia e loss de validação, parâmetros totais, tempo de treino
  e de inferência, armazenamento, sobreposição de erros e contribuição de cada membro.
- TTA (deslocamentos de 1 e/ou 2 px, sem espelhamento) escolhido na validação, com o ganho registrado à parte.

## Confirmação e final
- Finalistas confirmados com sementes extras e, se viável, em outras divisões só do split train (o procedimento
  é refeito em cada divisão; nenhum modelo é avaliado na validação que viu no treino).
- Solução fechada → avaliação única no teste → retreino dos membros em todo o split train (parando na melhor
  época de validação de cada membro, como no notebook) → RESULTS.txt gerado automaticamente.
- Sem melhora sustentada na validação: mantém a solução atual (#17) e documenta os resultados negativos.

## Rodada 1b — confirmação pareada (registrada às 16:05, antes dos resultados)
Triagem (semente 42): WRN-16-4 com AdamW deu 97,08% contra 96,66% da referência; o ensemble de 4 × 60 épocas da
CNN deu 97,30% sem TTA contra 97,05% do de 4 × 30 épocas. Antes de expandir qualquer ensemble, a combinação
"WRN-16-4 + 60 épocas" é medida junto, em pares com as mesmas sementes:

| | semente 42 | 43 | 44 |
|---|---|---|---|
| WRN-16-4, 30 épocas | t03_wrn16_4 | w30_val_s43 | w30_val_s44 |
| WRN-16-4, 60 épocas | t16_wrn16_4_ep60 | w60_val_s43 | w60_val_s44 |

Critério: a média das 3 sementes e a diferença em cada par. Uma configuração só vira base do ensemble se ganhar
na média e na maioria dos pares. Só depois disso o ensemble homogêneo grande é treinado (sementes 42, 43, … em
ordem fixa) e comparado com a solução atual (97,31%, ensemble de 5 + TTA) na validação. O teste fica para a
solução fechada.

## Rodada 2 — 10/11, 08:10 a 15:00 (registrada antes dos resultados da rodada 1b)
Os treinos rodam ao lado dos outros serviços das máquinas, que estão ociosos. Às 14:55 tudo para e os
serviços são conferidos.

A. **Em andamento:** retomada dos parciais (e60 s50/s51, t12_ep60, t15_wrn16_4_color, t16_wrn16_4_ep60, w30
   s43/s44), confirmação 1b (w60 s43/s44) e capacidade, com uma semente: WRN-16-8 (t17) e WRN-28-4 (t18).
B. **Configuração dos membros (C\*)**, escolhida pela regra da rodada 1b: WRN-16-4 com 30 ou 60 épocas. Uma versão
   maior (t17/t18) só substitui se ganhar ≥ 0,15 e se confirmar com as sementes 43 e 44.
C. **Ensemble homogêneo de C\*:** sementes 42, 43, …, 61, nessa ordem. Avaliado com k = 5, 10, 15 e 20 membros
   (e 1 a 20 na curva). Extensão para 62–71 (25/30) só se k = 15 → 20 ainda ganhar.
D. **Ensemble heterogêneo**, sem treino extra: C\* e a CNN de 60 épocas (e60) intercaladas em ordem fixa
   (C\*42, e60_42, C\*43, e60_43, …). Avaliado com os mesmos k.
E. **TTA** escolhido na validação entre none, shift1, shift2 e shift12, para cada estratégia, com o ganho
   registrado à parte. A média é sempre a simples do softmax; não há pesos por membro.
F. **Retreino final em paralelo:** os membros de C\* (e de e60, se o heterogêneo ganhar) são retreinados em todo o
   split train na mesma ordem de sementes (`final: true`, guardando as épocas a partir da 15). Cada membro para
   na melhor época de validação da mesma semente. Só os k primeiros da solução escolhida são usados.
G. **Fechamento** (se houver tempo): escolha final pela validação, uma única avaliação no teste, RESULTS.txt.
   Se não der tempo, o fechamento fica para a próxima janela e a solução do PR #21 continua valendo.

### Regra de parada da WRN homogênea (registrada às 09:05, com k = 7 visto e k = 10 ainda não)
No k = 10, se o ensemble homogêneo de WRN-16-4 (com o melhor TTA) ficar abaixo do ensemble homogêneo da CNN de
60 épocas na validação e com loss de validação maior, a WRN homogênea para em 10 membros. A GPU vai para a CNN de
60 épocas (sementes 52–61, para k = 15/20, e os retreinos finais 42–61) e para o heterogêneo (WRN 42–51 + CNN,
intercaladas). Os membros de validação 54–57 da WRN, já treinados, ficam registrados mas fora da ordem fixa.
- **09:28 — regra aplicada:** no k = 10, a WRN homogênea com o melhor TTA (shift12) deu 97,28% (loss 0,188),
  contra 97,45% da CNN de 60 épocas (shift2; loss 0,176). A WRN homogênea para em 10. Os retreinos de WRN que
  não tinham começado (wf30 46–51) ficam em espera (`q-hold/`), só para o caso de o heterogêneo vencer; os da
  CNN de 60 épocas passam para a frente da fila.

## Rodada 3 — 09:57 (registrada antes de rodar)
- k = 20 da CNN de 60 épocas: 97,47% sem TTA e 97,50–97,52% com TTA; de 15 para 20 ainda ganha de +0,01 a +0,05.
  Pela regra, estende-se para **25 e 30** (sementes 62–71) com os retreinos finais correspondentes.
- **Confirmação do procedimento em outra divisão** (`split_seed` 43; a validação é outra parte do split train e
  os modelos dessa divisão nunca são avaliados na validação da divisão 42): CNN de 60 épocas contra CNN de 30
  épocas (a receita do PR #21), sementes 42–51 em cada, comparando os ensembles com k = 5 e 10, com e sem TTA.
  A pergunta é se "60 épocas + ensemble grande" também ganha numa divisão que não guiou nenhuma escolha.
- **Escolha final:** o tamanho é fixado pela curva (o menor k em que ela já estabilizou, entre 20, 25 e 30; um k
  maior só se ganhar ≥ 0,05) e o TTA é o melhor nesse k. Só depois o teste é lido uma vez.

## Rodada 4 — augment automático (10:48, registrada antes de rodar; teste ainda não lido)
Motivada por `dgx/PESQUISA.md` (RandAugment, TrivialAugment e AutoAugment-SVHN, com a melhor evidência no SVHN
sem o split extra). Base: a CNN de 60 épocas (e60), trocando só o augment geométrico (`augment: false`) por uma
política por imagem; o cutout 8×8, o label smoothing e o resto ficam iguais.

| Nome | Política | Sementes |
|---|---|---|
| a_ta | TrivialAugmentWide | 42, 43, 44 |
| a_ra | RandAugment N = 3, M = 9 | 42, 43, 44 |
| a_aa | AutoAugment, política SVHN | 42, 43, 44 |

Comparação pareada com e60_val_s42–44 (96,60 / 96,96 / 96,81; média 96,79). Uma política só vira a base de um
novo ensemble de 20 (sementes 42–61, com retreinos finais) se ganhar ≥ 0,15 na média e na maioria dos pares; o
novo ensemble só substitui o atual (k = 20, shift12, 97,52%) se ganhar na validação. Se nenhuma política
ganhar, fecha-se com o ensemble atual.

### Resultado da rodada 4 e fechamento (10/11, 13:00)
- Média das 3 sementes: RandAugment 96,91% (+0,12; ganha em 2 dos 3 pares), AutoAugment SVHN 96,86% (+0,07) e
  TrivialAugment 96,68% (−0,11). Nenhuma política chegou a +0,15, então **fecha-se com o ensemble atual**:
  CNN de 60 épocas, k = 20, shift12, 97,52% de validação.
- Teste, lido uma única vez na solução fechada: **97,89%** (`results/dgx/final/test_acc.txt`).
- Ficam para a #22: a receita completa do artigo do RandAugment (WRN-28-2, 200 épocas) e um teste pareado de
  mais épocas na CNN (100 e 200, sementes 42–44 contra `e60_val_s42–44`, com a mesma regra da rodada 4).

## Rodada 5 — receita do RandAugment e Mean Teacher (13/11, 09:40, registrada antes de rodar; #22)
Motivada pelas previsões feitas só com os resultados já existentes (sem treinar): o ensemble de 60 épocas tende a
~97,49% com shift12 (ajuste erro(k) = e∞ + v/k em 200 ordens dos 30 membros), e 60 → 200 épocas na CNN deve render
só +0,07 a +0,11 por modelo. Para passar desse teto, é preciso mudar o modelo ou a receita, não o k nem as épocas.
Janela: até as 14:00, com os serviços das máquinas (vLLM, Postgres) no ar. Os treinos rodam com
`--oom-score-adj=1000` e limite de memória por container, e no máximo 4 por máquina.

| Nome | O que muda | Sementes |
|---|---|---|
| r_wrn282_ra | Receita do apêndice A.2.2 do RandAugment: WRN-28-2, RA N = 3, M = 9, cutout 16 depois do RA, sem augment geométrico, SGD Nesterov lr 0,005, wd 5e-3, batch 128, cosseno, 200 épocas, sem label smoothing | 42, 43, 44 |
| mt1, mt10 | CNN de 60 épocas (e60) + Mean Teacher no próprio treino: professor = EMA 0,999 do aluno, consistência = MSE entre os softmax (soma nas classes) de duas versões aumentadas da mesma imagem, peso 1 ou 10 com rampa nas primeiras 12 épocas; avaliado o professor | 42, 43, 44 |

- Comparação pareada com e60_val_s42–44 (96,60 / 96,96 / 96,81; média 96,79), mesma regra da rodada 4: ≥ +0,15 na
  média e vitória na maioria dos pares. Para a receita do RandAugment, a triagem também exige que um modelo
  sozinho, com shift12, passe de 97,5% em alguma semente: abaixo disso, não supera o ensemble atual.
- Se algo passar, o ensemble de 20 fica para outra janela. Corte de tempo: se a época da WRN-28-2 passar de 70 s,
  a receita do RandAugment é abortada (não termina até as 14:00).
- O teste não é lido.

### Resultado da rodada 5 (13/11, 10:25; teste não lido)
| | s42 | s43 | s44 | Média | Δ vs e60 | Pares | Melhor shift12 |
|---|---|---|---|---|---|---|---|
| e60 (referência) | 96,60 | 96,96 | 96,81 | 96,79 | — | — | 96,98 |
| **r_wrn282_ra** | **97,37** | **97,28** | **97,41** | **97,35** | **+0,56** | **3/3** | **97,52** |
| mt1 | 96,83 | 96,81 | 97,08 | 96,91 | +0,12 | 2/3 | 97,15 |
| mt10 | 96,83 | 96,83 | 96,93 | 96,86 | +0,08 | 2/3 | 96,98 |

- **A receita do RandAugment passa a regra** (+0,56, 3/3 pares; s44 com shift12 = 97,52% > 97,5%). Loss de
  validação 0,117, contra 0,195 da e60. A melhor época fica entre 194 e 196: ainda subia no fim.
- O Mean Teacher não passa (+0,12 e +0,08), e fica fora.
- Sanidade, sem ser critério: o ensemble dos 3 r_wrn282_ra com shift12 dá 97,57%, contra 97,31% dos 3 e60.
- Pelo plano, o ensemble de 20 (com os retreinos finais) fica para outra janela.

## Rodada 5b — ensemble de 10 da receita do RandAugment (13/11, 10:35, registrada antes dos resultados)
- Membros: r_wrn282_ra, sementes 42–51 em ordem fixa (validação), e os retreinos rf_wrn282_ra_s42–51 em todo o
  split train, cada um na época da melhor validação da mesma semente (pesos salvos das épocas 170 a 200).
- Substitui a solução atual (k = 20 de CNN de 60 épocas, shift12, 97,52%) só se o ensemble de 10 com shift12
  ganhar na validação. O TTA é fixo em shift12, sem escolha posterior.
- **Corte às 13:50 (a GPU fica livre às 14:00):** os treinos não concluídos são parados. Se faltar algum
  retreino final, o ensemble usa as sementes com retreino concluído, na ordem fixa, e a comparação na validação
  é refeita com essas mesmas sementes.
- O teste só é lido depois da escolha, uma vez.

### Retomada da rodada 5b (13/11, 13:16)
- Os 10 membros de validação terminaram. Com shift12: **7162/7325 = 97,7747%**, contra
  **7143/7325 = 97,5154%** da CNN de 20 membros.
- A fila ficou presa após a primeira tarefa de cada worker: `worker` verifica o diretório pelo nome
  do arquivo (`a_...` / `b_...`), enquanto `run_config` grava pelo campo `name` (sem o prefixo).
  Assim, os workers voltavam à tarefa concluída e não avançavam. Só há retreinos finais completos
  das sementes **42, 47 e 48**. Os sete restantes não começaram.
- As filas foram encerradas com `STOP`; os containers de treino saíram. O TTA terminou com código 0
  nas duas máquinas. Os containers dos serviços de produção continuam no ar.
- Aplicada a contingência às mesmas sementes, em ordem fixa: **7147/7325 = 97,5700%**.
  O trio supera a referência por 4 acertos e é a solução escolhida para este fechamento.
  É uma antecipação do fechamento de contingência: não há tempo para os sete retreinos antes do corte.
- Épocas escolhidas exclusivamente pela validação: s42 = 196, s47 = 188, s48 = 181.
  Fechamento iniciado no Mac com MPS, usando `final_ensemble.py`; saídas em
  `results/dgx/final_r5b/`. Ao concluir, gera RESULTS candidato e registra a avaliação única do teste.

- Fechamento r5b no Mac concluído: teste da solução escolhida = 98.2022%; 203856 predições no RESULTS candidato, 2407 diferentes do entregue. Artefatos em `results/dgx/final_r5b/`; arquivo entregue e notebook preservados.
