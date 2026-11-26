# Glossário

**Conjunto de treino**: dados usados para ajustar os pesos. No v1, a parte do `split="train"` fora da validação; no v2, essa parte mais o Extra, menos o que for Vazamento.

**Conjunto de validação**: fatia fixa do `split="train"`, separada do treino, usada para acompanhar os experimentos e escolher o modelo final. Nunca é usada para ajustar pesos.

**Conjunto de teste**: o `split="test"` do SVHN. Só estima o desempenho no mundo real; não participa de nenhuma escolha.

**Conjunto adicional**: imagens sem rótulo fornecidas pelo professor. Só é tocado depois que o modelo final está definido, para gerar o arquivo de predições.

**Extra**: o `split="extra"` do SVHN (531 mil imagens rotuladas). Não é o Conjunto adicional do professor. O v1 não o usa; o v2 o usa no treino, nunca na validação.

**Vazamento**: imagem presente no treino e também na validação ou no teste, seja idêntica (mesmo hash) ou uma Quase-duplicata com o mesmo rótulo. É removida do treino; validação e teste não mudam.

**Quase-duplicata**: imagem cujo dHash de 64 bits difere de outra em até 4 bits. Com rótulo diferente, é só relatada, porque costuma ser um dígito parecido, não uma cópia.

**Experimento**: um treino em que se altera uma única coisa em relação a um anterior, para medir o efeito dessa mudança.

**Modelo final**: o modelo escolhido pela acurácia de validação. É o mesmo modelo descrito no relatório, salvo nos pesos entregues e usado para gerar o arquivo de predições.

**Arquivo de predições**: `SAMUEL_WILKER_BERNARDO_RESULTS.txt`; uma predição (0–9) por linha, na ordem dos nomes das imagens do conjunto adicional.

**Baseline**: modelo de referência simples (aqui, kNN nos pixels) usado para medir o ganho das redes neurais.

**Random subsampling**: repetir o holdout várias vezes, cada vez com uma divisão treino/validação sorteada, e
resumir o desempenho com média e desvio. Mede o quanto um resultado depende da sorte do sorteio.
