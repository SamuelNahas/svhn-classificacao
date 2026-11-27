# Plano do v2 (#34) — registrado antes de qualquer treino do v2

Registrado em 26/11/2026, antes de qualquer treino do v2. Regras do v2: [`docs/V2.md`](../docs/V2.md) (#33). Este plano segue o
de [`PLANO.md`](PLANO.md). Mudar algo exige um adendo datado, feito antes de rodar o que ele afeta; o texto antigo
fica, marcado como substituído. Resultados do v2 só podem ser commitados depois deste commit.

## Protocolo comum

- **Dados:** `train` (73.257) + `extra` (531.131). `test` (26.032) fica fora de tudo até a avaliação final.
  As contagens são conferidas na #35 antes do primeiro treino; se não baterem, o plano para.
- **Validação:** a do v1, 7.325 imagens do `train`, semente 42. O Extra nunca entra nela. O treino é o resto do
  `train` (65.932) mais o `extra`. Normalização: média e desvio da divisão 42 do `train`, como no v1, para que a
  única diferença entre a reprodução e a receita com o Extra seja o dado.
- **Vazamento:** nenhum treino do v2 começa antes de a checagem por hash (#35) mostrar zero imagem repetida entre
  treino, validação e teste, ou antes de as repetições serem removidas e contadas.
- **Sementes:** 42, 47 e 48, as mesmas do v1, em todas as comparações pareadas.
- **Escolha:** melhor época, configuração e técnica **só pela validação**.
- **Teste:** os experimentos não calculam nem gravam o teste. Ele é lido uma única vez, no script de fechamento,
  depois de a solução estar congelada e registrada em adendo (mais estrito que o v1, que gravava `test_probs.npy`).
- **Conjunto adicional do professor:** fora do v2.

## O que conta como ganho

Uma mudança é comparada com a base acumulada, em pares com as mesmas três sementes. Ela **fica** se:

1. a média das diferenças de validação das três sementes for **maior que 0,3 ponto** (o erro-padrão da validação é
   ~0,18 ponto, segundo o [`PESQUISA.md`](PESQUISA.md)); e
2. a diferença for positiva em pelo menos 2 dos 3 pares; e
3. o McNemar sobre os acertos e erros das três sementes agrupadas tiver p < 0,05.

Caso contrário, a mudança é descartada e o resultado negativo é documentado. Custo maior que o saldo do orçamento
descarta a mudança mesmo que ela ganhe.

## Fases

Cada fase muda uma coisa em relação à base indicada. A ordem é fixa e registrada antes dos resultados.

| Fase | Mudança | Base | Hipótese | Decisão |
|---|---|---|---|---|
| 0 | Reprodução do v1 (3 WRN-28-2 + RandAugment, 200 épocas, só `train`) e custo por época com `train` + `extra` | — | O pipeline reproduz o v1: validação ~97,6%, dentro de 0,3 ponto de 97,57% | Se fora disso, parar e explicar o desvio antes da fase 1 (#36) |
| 1 | `extra` no treino, mesma receita | Fase 0 | O Extra sobe a validação em mais de 0,3 ponto | Regra de ganho |
| 2 | Mais épocas (o número vem do orçamento do Adendo 3) | Fase 1 | Treino mais longo sobe a validação | Regra de ganho |
| 3 | WRN-28-10 | melhor das fases 1–2 | A largura sobe a validação | Regra de ganho e custo ≤ saldo |
| 4 | SAM/ASAM | melhor até a fase 3 | Erro menor em pontos mínimos planos | Regra de ganho |
| 5 | Batch augmentation | melhor até a fase 4 | Mais cópias aumentadas por lote | Regra de ganho |
| 6 | Mixup / manifold mixup | melhor até a fase 5 | Regulariza sem prejudicar o RandAugment | Regra de ganho |
| 7 | Pré-treino (declarado no model card, origem e licença da fonte) | melhor até a fase 6 | Pesos iniciais melhores | Regra de ganho |

Fora do plano (marcadas "≈0" no `PESQUISA.md`): SWA, Lookahead, temperature scaling, snapshot ensembles.
Qualquer técnica ou fonte nova entra só por adendo datado, antes de rodar.

## Ensemble e fechamento

- **Ensemble:** 3 membros, sementes 42, 47 e 48, média simples do softmax, TTA shift12 (imagem e 8 cópias
  deslocadas ±1/±2 px), como no v1. É o modelo publicado.
- **Retreino final:** a configuração final é retreinada em todo o `train` (incluindo a validação) mais o `extra`, com
  a época fixada pela validação de cada membro, como no v1.
- **Fechamento:** solução congelada em adendo, depois **uma única leitura do teste**, registrada em `results/v2/`
  com a configuração completa.

## Orçamento de GPU e regra de parada

O custo por época com o `extra` só existe depois da fase 0, então o teto em GPU-horas **não é fixado neste
commit**. Ele entra no **Adendo 3**, commitado depois da fase 0 e antes da fase 1, com a tabela de custo por época
e por receita da #36. Regras que valem desde já:

- Uma fase só começa se o custo estimado (custo por época medido × épocas × sementes) couber no saldo.
- O saldo é o teto menos o que já foi gasto, registrado por fase.
- **Parada:** o v2 termina quando (a) o saldo acabar, (b) duas fases seguidas, entre as fases 3 a 7, forem
  descartadas, ou (c) a fase 7 acabar, o que vier primeiro. Depois, o fechamento.
- Se nenhuma mudança passar da regra de ganho, o v2 fecha com a receita da fase 1 (ou da fase 0, se a 1 também falhar)
  e documenta os resultados negativos.

## Adendos

### Adendo 1 — 26/11/2026, vazamento (#35), antes de qualquer treino

Regra de vazamento do protocolo, em detalhe. Imagem **idêntica** = mesmo SHA-256 dos 3.072 bytes. **Quase-duplicata** =
dHash de 64 bits (cinza, 8 × 9 blocos) a até **4 bits** de distância (limiar substituído pelo Adendo 2: 1 bit).
Cada imagem do treino (`train` fora da validação e `extra`) é comparada com a validação e o teste:

- idêntica: sai do treino;
- quase-duplicata de **mesmo rótulo**: sai do treino;
- quase-duplicata de rótulo diferente: fica e é só relatada (costuma ser dígito parecido, não cópia);
- repetição entre validação e teste: só relatada; validação e teste não mudam; repetição dentro do treino fica.

A mesma limpeza vale desde a fase 0, e as contagens removidas vão para `docs/DADOS_V2.md`. A implementação é
`dgx/dados_v2.py`, testada por `tests/check_dados_v2.py`; `dgx/prepare_data.py` a aplica e grava `keep_v2`.

Limite conhecido: quase-duplicatas além de 4 bits (recortes deslocados, outra escala) não são detectadas.

### Adendo 2 — 27/11/2026, limiar de quase-duplicata, antes de qualquer treino

**Substitui** o limiar do Adendo 1 (4 bits): quase-duplicata passa a ser dHash a até **1 bit**. Motivo: a preparação com
dados reais (`docs/DADOS_V2.md`) mostrou que, com 4 bits, 24.117 imagens (4% do treino) eram removidas e, numa amostra de
300, a diferença média de pixel para a "vizinha" (mediana 38 de 255) quase não diferia da de um par aleatório da mesma
classe (55): falsos positivos, não cópias. Nenhuma imagem era idêntica (SHA-256). O resto da regra do Adendo 1 não
muda. O limiar de 4 bits fica registrado como substituído.

### Adendo 3 — reservado

Teto de GPU-horas, depois da fase 0 e antes da fase 1 (#36).
