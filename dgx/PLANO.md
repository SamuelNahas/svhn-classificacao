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
