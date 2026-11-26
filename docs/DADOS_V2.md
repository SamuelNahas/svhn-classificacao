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

## Resultado da preparação

A preencher depois de rodar `prepare_data.py` (idênticas e quase-duplicatas removidas, quase-duplicatas de outro
rótulo relatadas, repetições validação–teste, distribuição das classes e tempo de preparo).
