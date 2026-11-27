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

## Resultado da preparação (26/11/2026, Mac com 17 GB de RAM)

`python dgx/prepare_data.py`: 25 min no total, quase tudo download do `extra` (1,3 GB) e swap; CPU: ~1,6 min. As contagens
bateram com as publicadas.

| | Imagens |
|---|---|
| Treino do v2 antes da limpeza (65.932 do `train` + 531.131 do `extra`) | 597.063 |
| Idênticas a validação ou teste (SHA-256) | **0** |
| Quase-duplicatas de mesmo rótulo (dHash ≤ 4 bits): removidas | 24.117 (4,0%); 21.014 do `extra` e 3.103 do `train` |
| Quase-duplicatas de rótulo diferente: só relatadas | 12.015 |
| Repetições idênticas entre validação e teste | 0 |
| **Treino do v2 depois da limpeza** | **572.946** |
| Validação | 7.325 |

Treino do v2 por classe (0 a 9): 48.362, 91.069, 82.115, 67.088, 53.989, 58.696, 46.202, 48.183, 39.293, 37.949.

### Leitura

- **Não há vazamento exato:** nenhuma imagem do treino é idêntica a uma da validação ou do teste.
- **As 24.117 removidas parecem falsos positivos do dHash, não cópias.** Em 300 removidas sorteadas, a diferença média
  de pixel para a vizinha mais próxima tem mediana 38,3 (de 255) e p90 75,1; um par aleatório da mesma classe tem
  mediana 55,3, e só 9,7% das 300 têm diferença menor que 20. A distância de dHash de 173 das 300 é exatamente 4 bits
  (o limite). Não são imagens planas: o desvio de pixel das removidas (mediana 32,4) é igual ao das mantidas (31,6).
- **Consequência:** com o limiar de 4 bits, o v2 treina com 24 mil imagens a menos que o necessário (4%), em sua
  maioria do `extra`. Mudar o limiar exige adendo datado ao plano, antes de qualquer treino (decisão pendente).
