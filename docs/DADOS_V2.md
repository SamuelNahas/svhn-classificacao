# Dados do v2 (#35)

Treino = `train` (fora da validação) + `extra`; validação = a do v1 (7.325 imagens, semente 42); teste = `split="test"`.
Regras: [`docs/V2.md`](V2.md). Limpeza de vazamento: Adendo 1 de [`dgx/PLANO_V2.md`](../dgx/PLANO_V2.md).

| Conjunto | Imagens publicadas |
|---|---|
| `train` | 73.257 (65.932 no treino e 7.325 na validação) |
| `test` | 26.032 |
| `extra` | 531.131 |

`python dgx/prepare_data.py` baixa os três conjuntos, confere as contagens acima, aplica a limpeza e grava
`data/svhn.npz` (com `x_extra`, `y_extra` e `keep_v2`) e `data/limpeza_v2.json`. `python tests/check_dados_v2.py`
testa a limpeza com imagens sintéticas (roda no CI).

## Resultado da preparação (27/11/2026, Mac com 17 GB de RAM)

`python dgx/prepare_data.py`: as contagens bateram com as publicadas. A primeira rodada (26/11) levou 25 min, quase tudo
download do `extra` (1,3 GB) e swap; a segunda levou 4 min e 22 s, sem download.

Limiar de quase-duplicata: dHash a até **1 bit** (Adendo 2 do plano). A primeira rodada usou 4 bits e removeu
24.117 imagens (4,0% do treino), com diferença média de pixel para a "vizinha" (mediana 38 de 255) perto da de um par
aleatório da mesma classe (55): falsos positivos. O limiar de 4 bits foi substituído.

| | Imagens |
|---|---|
| Treino do v2 antes da limpeza (65.932 do `train` + 531.131 do `extra`) | 597.063 |
| Idênticas a validação ou teste (SHA-256) | **0** |
| Quase-duplicatas de mesmo rótulo (dHash ≤ 1 bit): removidas | 1.431 (0,24%); 1.209 do `extra` e 222 do `train` |
| Quase-duplicatas de rótulo diferente: só relatadas | 546 |
| Repetições idênticas entre validação e teste | 0 |
| **Treino do v2 depois da limpeza** | **595.632** |
| Validação | 7.325 |

Treino do v2 por classe (0 a 9): 50.014, 101.872, 84.183, 68.380, 57.178, 59.691, 46.739, 49.053, 39.889, 38.633.

### Leitura

- **Não há vazamento exato:** nenhuma imagem do treino é idêntica a uma da validação ou do teste.
- **O dHash acha poucas cópias de verdade.** Em 300 removidas (1 bit) sorteadas, a diferença média de pixel para a vizinha
  tem mediana 34,1 (par aleatório da mesma classe: 56,9) e só 1% fica abaixo de 10 de 255. Ou seja, quase nenhuma é
  quase-cópia; são dígitos de forma parecida. Como só 1.431 imagens (0,24%) saem, o custo é pequeno e fica.
- **Limite:** o dHash não distingue cópia de dígito parecido. Um critério de pixel (por exemplo, diferença média < 10)
  seria mais fiel, mas exigiria novo adendo; não é necessário com 0 idênticas.
