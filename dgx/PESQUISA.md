# Pesquisa bibliográfica: técnicas ainda não exploradas para o SVHN

Data da pesquisa: 10/11/2026. Escopo: técnicas **não** listadas no que já testamos (AdamW/SGD, label smoothing,
EMA, rotação/escala/translação, cutout, brilho/contraste, CNN mais larga/profunda, WRN 16-4/16-8/28-4/22-1,
30 × 60 épocas, bf16, TTA de deslocamento, ensembles homogêneos e heterogêneos por média do softmax).

## Fontes que consegui usar

| Fonte | Situação |
|---|---|
| Google Scholar | **Bloqueado.** A busca automática caiu numa página de verificação ("Verifying your request…") e a extensão do Chrome não estava conectada, então não deu para buscar pelo navegador. Nenhuma informação deste relatório veio do Scholar. |
| Semantic Scholar (API) | **Bloqueado** (HTTP 429, limite de requisições sem chave). |
| Papers with Code | **Indisponível**: o site redireciona para huggingface.co/papers (o serviço foi desativado), sem leaderboards. |
| arXiv | **Usado como fonte principal.** Baixei os PDFs de 37 artigos e extraí o texto. Todos os números citados abaixo foram lidos nas tabelas e no texto desses PDFs, salvo quando digo o contrário. |
| Documentação do torchvision | Usada para confirmar que `AutoAugmentPolicy` tem as políticas IMAGENET, CIFAR10 e **SVHN**. |
| Busca web genérica | Usada uma vez, para procurar resultados do SVHN sem o `extra`. Não trouxe nenhum número além dos que já estão no RandAugment. |

## Como ler os números (leitura crítica)

- **A maioria dos resultados "SOTA" no SVHN usa train + extra (604 mil imagens).** É o caso de WRN (1,54–1,64%),
  DenseNet (1,59–1,79%), stochastic depth (1,75%), Cutout (1,30%), AutoAugment (1,0–1,1%), SAM (0,99%) e
  temperature scaling (o ResNet-152 SD com 598.388 imagens de treino). Esses números **não são comparáveis** aos
  nossos e servem só como evidência de **ganho relativo**.
- **Resultados só com o train (73.257 imagens, o "SVHN core")** que encontrei: RandAugment (Tabela 2), TrivialAugment
  (Tabelas 2, 4, 6 e 7), Mean Teacher (Tabela 1, coluna "73257 labels"), UDA (baseline totalmente supervisionado) e,
  pela ordem de grandeza do erro, Manifold Mixup (o artigo não diz qual split usou).
- **O Snapshot Ensembles é ambíguo:** o texto fala só em 73.257 imagens de treino, mas o DenseNet-40 sozinho dá
  1,77% de erro, praticamente o mesmo do artigo do DenseNet, que usa o extra (1,79%). Por isso trato esse resultado
  como "provavelmente com extra, não confirmado".
- **Os baselines dos artigos sobre SVHN core são fracos:** só cutout, sem rotação, escala nem translação. Nossa
  receita já tem augment geométrico + cutout, então os ganhos publicados são um **teto otimista**.
- **Ruído da nossa validação:** com 7.325 imagens e cerca de 97,5% de acurácia, o erro-padrão de uma acurácia é
  √(0,975·0,025/7325) ≈ **0,18 ponto**. Diferenças abaixo de ~0,2–0,3 ponto entre duas configurações exigem
  comparação pareada (mesmas sementes e McNemar nos acertos e erros) ou várias sementes.
- **A coluna "ganho esperado" das tabelas é estimativa minha**, não número de artigo.

## Tabela-resumo priorizada (ganho esperado × custo)

Ordem: primeiro pelo ganho esperado **no ensemble final**, depois pelo custo. "Pontos" = pontos percentuais de
acurácia na validação.

| # | Técnica | Melhor evidência (split) | Ganho esperado aqui | Custo de implementação | Custo de treino | Compatível? |
|---|---|---|---|---|---|---|
| 1 | **TrivialAugment / RandAugment** (+ cutout depois) | SVHN core: WRN-28-2 96,7 → **98,3** (RA); WRN-28-10 97,12 → **98,11** (TA) | **+0,3 a +0,8** num modelo sozinho; +0,2 a +0,5 no ensemble | Muito baixo (1 linha no torchvision) | Precisa de mais épocas (100–200) | Sim |
| 2 | **AutoAugment com a política SVHN** | SVHN core (replicado pelo RA): WRN-28-2 98,0, WRN-28-10 98,1; com extra: 1,5 → 1,1% de erro | Parecido com o item 1 | Muito baixo (`AutoAugmentPolicy.SVHN`) | Como o item 1 | Sim: a política já pronta é só usada, sem nova busca |
| 3 | **SAM / ASAM** | SVHN com extra: WRN-28-10 1,58 → 1,42 (básico) e 1,14 → 0,99 (com AA); CIFAR-10 com ASAM: WRN-28-2 95,13 → 95,94 | +0,1 a +0,3 por modelo | Baixo (~30 linhas ou pacote) | **~2×** por época | Sim |
| 4 | **Consistência no próprio train (Π-model / Mean Teacher)** | SVHN core com **todos** os 73.257 rótulos: 2,75 → **2,50%** de erro (ConvNet de 13 camadas) | +0,1 a +0,25 (menos com o RA ativo) | Médio | ~2× (duas passadas) | Sim, só com o train |
| 5 | **Batch augmentation** (M cópias aumentadas da mesma imagem no lote) | CIFAR-10: WRN-28-10 + cutout 96,6 → 97,15 (M=6); ResNet-44 + cutout 93,7 → 95,43 (M=40) | +0,1 a +0,3 (maior com o RA) | Baixo | M× por época | Sim |
| 6 | **Manifold Mixup / mixup** | SVHN (split não informado, erro ~2,8% sugere só o train): PreActResNet18 2,89 → 2,76 (input mixup) e → **2,27** (manifold) | 0 a +0,3; arriscado com LS e ensemble | Baixo (input) / médio (manifold) | Pede treino longo (o artigo usa 1200 épocas) | Sim |
| 7 | **Diversidade de membros por hiperparâmetro** (hyper-deep ensembles: membros com augment ou regularização diferentes) | CIFAR-10/100, WRN-28-10 (Wenzel 2020; número não conferido) | +0,05 a +0,2 no ensemble | Baixo (já temos o pipeline) | Zero extra (troca membros) | Sim |
| 8 | **Destilação** (do ensemble para um modelo; born-again; self-distillation) | CIFAR-100 BAN: DenseNet-90-60 17,69 → 16,69; ensembles de BANs 15,39. Em fala: 1 modelo destilado ≈ ensemble de 10 | ≈0 no ensemble; +0,2 a +0,5 num modelo sozinho | Médio | +1 treino por aluno | Sim (o professor vem do próprio train). Atenção ao LS no professor |
| 9 | **ShakeDrop / Shake-Shake / stochastic depth** | CIFAR-100 PyramidNet 18,01 → 16,22 (ShakeDrop); SVHN com extra ResNet-152 1,80 → 1,75 (SD) | ≈0 a +0,1 nas nossas redes rasas | Médio | Shake-Shake usa 1800 épocas | Sim |
| 10 | **Arquiteturas: DenseNet / PyramidNet** | SVHN com extra: DenseNet (k=24) 1,59%; WRN-16-4 + dropout 1,64% | Parecida com a da WRN; serve para diversidade | Médio | Mais lento que a CNN | Sim |
| 11 | **CutMix** | CIFAR-10 PyramidNet-200: 3,85 → 2,88; CIFAR-100: 16,45 → 14,47 | Incerto (dígitos vizinhos colados) | Baixo | ~0 | Sim |
| 12 | **SWA** | CIFAR-10 WRN-28-10: 96,18 → 96,79 (1,5 orçamento) | ≈0 (já temos EMA) | Baixo | +25–50% de épocas | Sim |
| 13 | **Snapshot ensembles / FGE** | SVHN (provável extra): WRN-32 1,90 → 1,51; CIFAR: FGE ≈ independentes com 2–3 orçamentos | ≈0 com 20 membros independentes | Baixo | Mesmo custo de 1 treino | Sim |
| 14 | **Lookahead / one-cycle** | CIFAR-10 ResNet-18: 95,23 → 95,27 (Lookahead); one-cycle ganha em velocidade | ≈0 | Baixo | 0 | Sim |
| 15 | **Temperature scaling antes da média** | O artigo sobre SVHN mostra efeito na calibração, não no erro (1,98% → 1,98%) | ≈0 na acurácia | Muito baixo | 0 | Sim (T ajustado na validação) |
| 16 | **Semi-supervisionado "de verdade"** (FixMatch, UDA, pseudo-rótulo) | Feitos para poucos rótulos; o não rotulado aqui seria o próprio train | ≈ item 4 | Alto | 2–7× | Só se o não rotulado for o train; **teste ou conjunto do professor: proibido** |

---

## 1. Augmentation automática

### 1.1 RandAugment (RA)
- **Referência:** E. D. Cubuk, B. Zoph, J. Shlens, Q. V. Le. *RandAugment: Practical automated data augmentation
  with a reduced search space.* CVPR Workshops 2020. https://arxiv.org/abs/1909.13719
- **Resultado reportado (Tabela 2, média de 10 execuções, acurácia no teste):**
  - **SVHN core (73 mil imagens, sem extra):** WRN-28-2 com baseline 96,7, AA 98,0 e **RA 98,3**; WRN-28-10 com
    baseline 96,9, AA 98,1 e **RA 98,3**.
  - SVHN com extra: WRN-28-2 98,2 → 98,7; WRN-28-10 98,5 → 99,0.
  - O texto destaca que, na WRN-28-2, aplicar o RA só ao core **supera** usar o extra sem RA (98,3 contra 98,2).
- **Configuração (apêndice A.2.2, core):** N = 3 operações, magnitude M = 9 (testaram 5, 7, 9 e 11), **cutout
  aplicado depois do RA**, lr 5e-3, weight decay 5e-3, LR cosseno, **200 épocas**.
- **Crítica:** o TrivialAugment (abaixo) **não conseguiu reproduzir** o RA no SVHN core e obteve 98,06 com a
  WRN-28-10, 0,24 abaixo do publicado. O baseline do artigo é só cutout. Para nós, o ganho real deve ser menor,
  porque já temos augment geométrico. Mesmo assim, é a evidência mais forte e mais diretamente aplicável que
  encontrei, com o mesmo dataset, o mesmo split e uma rede de tamanho parecido: a WRN-28-2 tem ~1,5 M de
  parâmetros e a nossa CNN ~1 M.
- **Ganho esperado:** +0,3 a +0,8 ponto num modelo sozinho; no ensemble, menos (+0,2 a +0,5), porque o ensemble
  já remove parte da variância.
- **Custo:** implementação trivial (`torchvision.transforms.RandAugment(num_ops=3, magnitude=9)` antes de
  `ToTensor`/cutout). Treino: o augment no CPU pode virar gargalo. Vale medir o throughput do DataLoader na DGX
  (CPU ARM) ou aplicar o augment na GPU. Augmentation forte costuma pedir mais épocas: testar 60 e 100–120.
- **Compatível:** sim, porque não usa dados externos nem modelo pré-treinado. Escolher N e M pela validação.
- **Cuidado:** o RA do torchvision não tem flip, o que é bom para dígitos, mas tem rotação e shear com magnitude
  alta. Combinado com o nosso augment geométrico, pode haver dupla distorção. Testar o RA **no lugar** da nossa
  rotação/escala, e não somado a ela.

### 1.2 TrivialAugment (TA)
- **Referência:** S. G. Müller, F. Hutter. *TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation.*
  ICCV 2021. https://arxiv.org/abs/2103.10158
- **Resultado reportado (WRN-28-10, SVHN core, média de 10 execuções):** baseline (só cutout 16 px) 97,12 ±
  0,05, **TA 98,11 ± 0,03** (Tabela 2). Na reprodução do próprio artigo (Tabela 4c): AA 97,99, RA 98,06,
  TA(RA) 98,05, **TA(Wide) 98,11**.
  - **Força fixa no máximo é melhor no SVHN core** (Tabela 7, espaço RA): força {30} 98,16 contra {0..30} 98,05.
    O artigo explica que a maioria das operações é de cor, e trocar drasticamente a cor de fundo e do dígito
    continua gerando números válidos.
  - Espaço de augment (Tabela 6): o espaço "Full" fica pior (97,63). **Remover o Invert piora no SVHN** (97,97
    contra 98,04 do AA). O OHL, com só 3 forças, dá 98,10.
- **Configuração (apêndice A.2):** sem outras augmentations no SVHN, só cutout de 16 px no fim. SGD Nesterov
  0,9, lr 0,005, batch 128, cosseno. No core: **200 épocas, weight decay 5e-3**.
- **Ganho esperado:** igual ao RA, com a vantagem de ter **zero hiperparâmetros**. É o primeiro a testar.
- **Custo:** `torchvision.transforms.TrivialAugmentWide()`, que corresponde ao "TA (Wide)". A variante com força
  fixa em 30 exige uma subclasse pequena, porque o torchvision sorteia a magnitude.
- **Compatível:** sim.

### 1.3 AutoAugment (AA) com a política SVHN
- **Referência:** E. D. Cubuk, B. Zoph, D. Mané, V. Vasudevan, Q. V. Le. *AutoAugment: Learning Augmentation
  Strategies from Data.* CVPR 2019. https://arxiv.org/abs/1805.09501
- **Resultado reportado (Tabela 2, erro %):** SVHN **com extra** (core + extra, seguindo a WRN): WRN-28-10 com
  baseline 1,5, cutout 1,3 e **AA 1,1**; Shake-Shake 26 2×96d 1,4 / 1,2 / **1,0**. "Reduced SVHN" (1.000
  imagens): WRN-28-10 13,2 → 8,2. Sem extra, o número vem de replicações do RA: WRN-28-2 98,0 e WRN-28-10 98,1
  (core), e 97,99 na reprodução do TA.
- **Política:** foi buscada em 1.000 imagens do **train** do SVHN. As operações mais frequentes são **Invert,
  Equalize, ShearX/Y e Rotate**. O artigo justifica que o shear aparece porque os números reais já vêm inclinados.
- **Ganho esperado:** parecido com o RA e o TA. A busca de política não precisa ser refeita.
- **Custo:** `transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN)`, política confirmada na documentação
  do torchvision.
- **Compatível:** sim, com uma ressalva para discutir no relatório do trabalho: a política foi aprendida por
  terceiros no train do SVHN, e não em dados externos. É uma "escolha de hiperparâmetro feita fora da nossa
  validação", e não um modelo pré-treinado. Na minha leitura é permitido, mas vale citar.

### 1.4 Fast AutoAugment
- **Referência:** S. Lim, I. Kim, T. Kim, C. Kim, S. Kim. *Fast AutoAugment.* NeurIPS 2019.
  https://arxiv.org/abs/1905.00397
- **Resultado:** SVHN com extra, WRN-28-10: 98,8 (conforme a Tabela 1 do RA e a Tabela 2 do TA), contra 98,9 do
  AA. Não traz ganho sobre o AA ou o TA e exige busca. **Não recomendado.**

---

## 2. Regularização por mistura

### 2.1 mixup
- **Referência:** H. Zhang, M. Cisse, Y. N. Dauphin, D. Lopez-Paz. *mixup: Beyond Empirical Risk Minimization.*
  ICLR 2018. https://arxiv.org/abs/1710.09412
- **Resultado (CIFAR, erro %):** CIFAR-10 PreAct ResNet-18 5,6 → 4,2; WRN-28-10 3,8 → 2,7. CIFAR-100 WRN-28-10
  19,4 → 17,5. O artigo não traz SVHN.
- **SVHN (pelo artigo do Manifold Mixup, abaixo):** input mixup α=1 dá só 2,89 → 2,76 (PreActResNet18) e 2,80 →
  2,68 (WRN-28-10). Ganho pequeno.
- **Ganho esperado aqui:** 0 a +0,15 ponto. **Riscos:** (a) somado ao label smoothing, que já usamos, os alvos
  ficam suaves duas vezes; (b) Wen et al. 2021 (seção 8.4) mostram que ensembles de modelos treinados com mixup
  ficam **subconfiantes**. Na acurácia o efeito é pequeno, mas exige recalibrar.
- **Custo:** ~10 linhas, custo de treino ~0, mas costuma pedir mais épocas.

### 2.2 Manifold Mixup
- **Referência:** V. Verma, A. Lamb, C. Beckham, A. Najafi, I. Mitliagkas, D. Lopez-Paz, Y. Bengio. *Manifold
  Mixup: Better Representations by Interpolating Hidden States.* ICML 2019. https://arxiv.org/abs/1806.05236
- **Resultado no SVHN (Tabela 2, 5 execuções, erro %):**

  | Modelo | Sem mixup | Input mixup (α=1) | Manifold mixup (α=2) |
  |---|---|---|---|
  | PreActResNet18 | 2,89 | 2,76 | **2,27** |
  | PreActResNet34 | 2,97 | 2,67 | **2,18** |
  | WRN-28-10 | 2,80 | 2,68 | **2,06** |

- **Crítica:** o artigo **não diz** se usou o extra. Os erros de ~2,8% no baseline são típicos do core sem
  augment (com o extra, a WRN chega a ~1,6%), então **provavelmente é só o train, mas não confirmei**. O treino
  tem **1200 épocas** (lr ÷10 nas épocas 400 e 800), e o baseline não tem nenhuma augmentation forte.
- **Ganho esperado aqui:** +0 a +0,3, provavelmente bem menos do que os ~0,6–0,7 do artigo, porque já temos
  augment, cutout e LS, e treinamos 60 épocas.
- **Custo:** médio, porque é preciso misturar ativações num bloco sorteado (fácil na nossa CNN de 3 blocos).
  Treino com custo ~0 por época, mas pede mais épocas.
- **Compatível:** sim.

### 2.3 CutMix
- **Referência:** S. Yun, D. Han, S. J. Oh, S. Chun, J. Choe, Y. Yoo. *CutMix: Regularization Strategy to Train
  Strong Classifiers with Localizable Features.* ICCV 2019. https://arxiv.org/abs/1905.04899
- **Resultado:** CIFAR-10, PyramidNet-200 (Tabela 7, top-1 erro): baseline 3,85, cutout 3,10, mixup (α=1) 3,09,
  manifold mixup 3,15, **CutMix 2,88**. CIFAR-100: 16,45 → 14,47. Sem SVHN.
- **Crítica específica do SVHN:** o rótulo é o dígito **central**, e as imagens já trazem dígitos vizinhos. Colar
  um retângulo de outra imagem pode pôr um dígito "inteiro" na região central e criar ruído de rótulo. O cutout
  não tem esse problema. **Não encontrei evidência no SVHN**, e o ganho pode não se transferir.
- **Ganho esperado:** incerto, entre −0,2 e +0,2. Custo baixo (p = 0,5, α = 1, como no artigo).

---

## 3. Otimizadores e procedimentos

### 3.1 SAM (Sharpness-Aware Minimization)
- **Referência:** P. Foret, A. Kleiner, H. Mobahi, B. Neyshabur. *Sharpness-Aware Minimization for Efficiently
  Improving Generalization.* ICLR 2021. https://arxiv.org/abs/2010.01412
- **Resultado no SVHN (Tabela 5, erro %, train + extra, explicitamente "73257 + 531131"):**

  | Modelo | Augment | SGD | SAM |
  |---|---|---|---|
  | WRN-28-10 | básico | 1,58 | **1,42** |
  | WRN-28-10 | AutoAugment | 1,14 | **0,99** |
  | Shake-Shake 26 2×96d | básico | 1,58 | **1,44** |
  | Shake-Shake 26 2×96d | AutoAugment | 1,03 | 1,07 (piora) |

  Hiperparâmetros para o SVHN (Tabela 7): lr 0,01, wd 5e-4, **ρ = 0,01**. No CIFAR-10, ρ = 0,05.
- **Crítica:** só há números com o extra. No core, o ρ ideal deve ser maior, porque há menos dados e mais
  overfitting (no CIFAR, de 50 mil imagens, o ρ é 0,05). Num caso o SAM piorou o resultado.
- **Ganho esperado:** +0,1 a +0,3 por modelo.
- **Custo:** duas passadas forward/backward por passo, portanto **~2× o tempo** (≈24 min por modelo). O artigo
  compara com o baseline treinado pelo dobro de épocas. O SAM funciona como invólucro de qualquer otimizador:
  dá para usar AdamW + SAM, mas os números do artigo são com SGD.
- **Compatível:** sim.

### 3.2 ASAM (Adaptive SAM)
- **Referência:** J. Kwon, J. Kim, H. Park, I. K. Choi. *ASAM: Adaptive Sharpness-Aware Minimization for
  Scale-Invariant Learning of Deep Neural Networks.* ICML 2021. https://arxiv.org/abs/2102.11600
- **Resultado no CIFAR-10 (Tabela 2, acurácia máxima no teste, 5 execuções):** WRN-28-2 com SGD 95,13, SAM 95,74
  e **ASAM 95,94**; WRN-28-10 96,34 / 96,98 / **97,28**; ResNet-56 94,58 / 95,18 / 95,42.
- **Configuração:** ρ = 0,5 no CIFAR-10 (ASAM) e 0,05 (SAM), η = 0,01, batch 128, SGD lr 0,1, momentum 0,9, wd
  5e-4, cosseno, 200 épocas para SAM e ASAM contra 400 para o SGD, e **label smoothing**, como no nosso caso.
- **Crítica:** o artigo reporta a **acurácia máxima no teste**, uma forma de escolha pelo teste que infla os
  números. Não há SVHN.
- **Ganho esperado e custo:** como o SAM. É uma troca de uma linha se usarmos uma implementação que tenha os dois.

### 3.3 Lookahead
- **Referência:** M. R. Zhang, J. Lucas, G. Hinton, J. Ba. *Lookahead Optimizer: k steps forward, 1 step back.*
  NeurIPS 2019. https://arxiv.org/abs/1907.08610
- **Resultado (Tabela 1, ResNet-18, validação):** CIFAR-10 com SGD 95,23, Polyak 95,26, Adam 94,84 e **Lookahead
  95,27**; CIFAR-100 78,24 → 78,34. O ganho fica dentro do desvio. **Ganho esperado ≈ 0. Não recomendado.**

### 3.4 SWA (Stochastic Weight Averaging)
- **Referência:** P. Izmailov, D. Podoprikhin, T. Garipov, D. Vetrov, A. G. Wilson. *Averaging Weights Leads to
  Wider Optima and Better Generalization.* UAI 2018. https://arxiv.org/abs/1803.05407
- **Resultado (Tabela 1):** CIFAR-10 WRN-28-10 com SGD 96,18, SWA com 1 orçamento 96,45 e com 1,5 orçamento
  96,79. CIFAR-100 WRN-28-10 80,82 → 82,15.
- **Crítica:** o SWA faz a mesma coisa que o **EMA dos pesos, que já testamos**. O baseline do artigo é um SGD com
  decaimento de LR pouco cuidadoso, enquanto o nosso já tem cosseno + EMA. **Ganho esperado ≈ 0–0,1.** Barato
  (`torch.optim.swa_utils`), mas baixa prioridade.

### 3.5 One-cycle / super-convergence
- **Referência:** L. N. Smith, N. Topin. *Super-Convergence: Very Fast Training of Neural Networks Using Large
  Learning Rates.* 2017/2019. https://arxiv.org/abs/1708.07120
- **Resultado:** ResNet-56 no CIFAR-100: 100–400 épocas de LR por degraus dão 60,8–61,6%, contra 65,4% (25
  épocas) e 66,4% (50 épocas) com 1cycle (lr 0,1–0,5, momentum 0,95–0,85). O artigo diz que o ganho é maior
  com poucos dados.
- **Crítica:** a comparação é com LR **em degraus**, não com cosseno. O benefício principal é **velocidade**
  (treinar em menos épocas), o que poderia baratear o ensemble, mas não aumenta o teto. **Ganho esperado em
  acurácia ≈ 0.** Implementação de uma linha (`OneCycleLR`).

### 3.6 Batch augmentation (augmentation repetida no lote)
- **Referência:** E. Hoffer, T. Ben-Nun, I. Hubara, N. Giladi, T. Hoefler, D. Soudry. *Augment your batch:
  better training with larger batches.* 2019. https://arxiv.org/abs/1901.09335
- **Resultado (Tabela 3, CIFAR-10):** ResNet-44 + cutout 93,7 → **95,43** (M = 40); WRN-28-10 + cutout 96,6 →
  **97,15** (M = 6). CIFAR-100 WRN-28-10 + cutout 79,85 → 80,13 (M = 10). O TrivialAugment também reporta
  variantes "x8" entre as melhores no CIFAR-100.
- **Crítica:** o custo de computação por época sobe M vezes (o lote efetivo é M× maior com o mesmo número de
  passos). O ganho é maior quando o augment é forte e aleatório (RA/TA). Sem SVHN.
- **Ganho esperado:** +0,1 a +0,3 com RA/TA. Na DGX, M = 2–4 cabe facilmente.

---

## 4. Destilação

### 4.1 Knowledge distillation (incluindo do ensemble para um modelo só)
- **Referência:** G. Hinton, O. Vinyals, J. Dean. *Distilling the Knowledge in a Neural Network.* 2015.
  https://arxiv.org/abs/1503.02531
- **Resultado:** só em fala (Tabela 1): o baseline dá 58,9% de acurácia de frame, o ensemble de 10 dá 61,1% e o
  **modelo destilado dá 60,8%**, quase todo o ganho do ensemble num só modelo. Sem SVHN nem CIFAR.
- **Contraponto:** S. Stanton, P. Izmailov, P. Kirichenko, A. A. Alemi, A. G. Wilson. *Does Knowledge
  Distillation Really Work?* NeurIPS 2021. https://arxiv.org/abs/2106.05945 . Em CIFAR-100 com ResNet-56, o
  aluno **não** reproduz o ensemble (a concordância fica baixa) e, com temperatura τ = 1, destilar de um ensemble
  de 3 não é melhor que destilar de um modelo só. É preciso τ maior (o artigo usa τ = 4).
- **Receita moderna:** L. Beyer, X. Zhai, A. Royer, L. Markeeva, R. Anil, A. Kolesnikov. *Knowledge
  distillation: A good teacher is patient and consistent.* CVPR 2022. https://arxiv.org/abs/2106.05237 . O
  professor e o aluno devem ver **a mesma imagem aumentada** ("consistent"), com treino longo ("patient"). Os
  experimentos usam professores pré-treinados no ImageNet, e esse detalhe não se aplica aqui. Só a receita se
  transfere.
- **Problema específico nosso:** R. Müller, S. Kornblith, G. Hinton. *When Does Label Smoothing Help?* NeurIPS
  2019. https://arxiv.org/abs/1906.02629 . **Professores treinados com label smoothing destilam pior.** No MNIST,
  o aluno de um professor com LS ficou com 0,91% de erro, contra 0,67% do aluno sem LS. Nossos 20 membros usam
  LS 0,1.
- **Ganho esperado:** para a **acurácia final do ensemble, ≈ 0**, porque o aluno quase nunca supera o professor.
  Num **modelo só**, o aluno pode ganhar +0,2 a +0,5 sobre os ~96,7% atuais. Só faz sentido se o trabalho
  valorizar custo de inferência.
- **Custo:** médio (um treino a mais). Compatível, porque o professor vem do próprio train. Atenção: os logits do
  professor sobre o train ficam muito confiantes, o que é outro motivo para usar τ alto e augment consistente.

### 4.2 Born-again networks (self-distillation em gerações)
- **Referência:** T. Furlanello, Z. C. Lipton, M. Tschannen, L. Itti, A. Anandkumar. *Born Again Neural
  Networks.* ICML 2018. https://arxiv.org/abs/1805.04770
- **Resultado (CIFAR-100, Tabela 2, erro %):** DenseNet-90-60 com professor 17,69 e BAN 16,69. Sequências de BANs
  em ensemble: DenseNet-80-120 chega a 15,13 (Ens×2) e 14,9 (Ens×3).
- **Para nós:** uma variante seria treinar cada novo membro destilando do ensemble de 20 e depois montar o
  ensemble de alunos. Não encontrei evidência de que isso supere um ensemble independente com o mesmo custo no
  SVHN. **Ganho esperado: incerto, ≈ 0–0,1** no ensemble, com custo dobrado.

### 4.3 Self-distillation interna ("Be Your Own Teacher")
- **Referência:** L. Zhang, J. Song, A. Gao, J. Chen, C. Bao, K. Ma. *Be Your Own Teacher: Improve the
  Performance of Convolutional Neural Networks via Self Distillation.* ICCV 2019. https://arxiv.org/abs/1905.08094
- **Resultado:** ganho médio de 2,65% no CIFAR-100, maior nas redes profundas (ResNet-101: +4,05). Nossa CNN de 9
  convoluções é rasa, e o artigo indica ganho menor em redes rasas. **Ganho esperado ≈ 0–0,1.**

---

## 5. ShakeDrop, Shake-Shake, stochastic depth e PyramidNet

### 5.1 Shake-Shake
- **Referência:** X. Gastaldi. *Shake-Shake regularization.* 2017. https://arxiv.org/abs/1705.07485
- **Resultado:** CIFAR-10, ResNet 26 2×96d: 3,58 → **2,86%** (Shake-Shake-Image), mas com **1800 épocas**. Com o
  extra e AA no SVHN, chega a 1,0% de erro (Tabela 2 do AutoAugment, Shake-Shake treinado 160 épocas pelo
  tamanho do extra).
- **Crítica:** precisa de redes residuais de 2 ramos e de treino muito longo para compensar o ruído. **Fora do
  nosso orçamento.** Ganho esperado ≈ 0 em 60–100 épocas.

### 5.2 ShakeDrop
- **Referência:** Y. Yamada, M. Iwamura, T. Akiba, K. Kise. *ShakeDrop Regularization for Deep Residual
  Learning.* IEEE Access 2019. https://arxiv.org/abs/1802.02375
- **Resultado:** CIFAR-100 com PyramidNet (Tabela 4, 300 épocas): vanilla 18,01 → **16,22–16,24**. Sem SVHN.
- **Crítica:** só existe para redes residuais profundas. Nossa CNN não é residual, e a WRN-16-4 é rasa. Ganho
  esperado ≈ 0–0,1, custo médio.

### 5.3 Stochastic depth
- **Referência:** G. Huang, Y. Sun, Z. Liu, D. Sedra, K. Q. Weinberger. *Deep Networks with Stochastic Depth.*
  ECCV 2016. https://arxiv.org/abs/1603.09382
- **Resultado no SVHN (train + extra, sem augmentation):** ResNet-152 1,80 → **1,75%**. O artigo comenta que o
  baseline sofre overfitting na 2.ª fase do LR. CIFAR-10+: 6,41 → 5,25.
- **Crítica:** o ganho no SVHN é só 0,05 ponto, numa rede de 152 camadas. Nas nossas redes de 9–16 camadas,
  **≈ 0**.

### 5.4 PyramidNet
- **Referência:** D. Han, J. Kim, J. Kim. *Deep Pyramidal Residual Networks.* CVPR 2017.
  https://arxiv.org/abs/1610.02915
- **Resultado (CIFAR, erro %):** PyramidNet-110 com α = 48 (1,7 M de parâmetros): 4,58 no CIFAR-10 e 23,12 no
  CIFAR-100; α = 270 (28,3 M): 3,73 e 18,25. **Não encontrei número no SVHN no artigo.**
- **Para nós:** a PyramidNet-110 com α = 48 tem o mesmo tamanho da WRN-28-2. Seria só mais uma arquitetura, com
  ganho incerto, já que a WRN rendeu menos que a CNN no ensemble.

---

## 6. Arquiteturas 32×32 que se destacam no SVHN

| Arquitetura | SVHN (erro %) | Split | Fonte |
|---|---|---|---|
| WRN-16-4 sem/com dropout 0,4 | 1,85 / **1,64** | train + extra | Zagoruyko & Komodakis 2016, Tabela 6 |
| WRN-16-8 + dropout 0,4 sem/com cutout | 1,60 / **1,30** | train + extra | DeVries & Taylor 2017, Tabela 1 |
| DenseNet (k=12) L=40 / L=100 | 1,79 / 1,67 | train + extra | Huang et al. 2017, Tabela 2 |
| DenseNet (k=24) L=100 (27,2 M) | **1,59** | train + extra | idem |
| DenseNet-BC (k=12) L=100 (0,8 M) | 1,76 | train + extra | idem |
| ResNet-152 + stochastic depth | 1,75 | train + extra | Huang et al. 2016 |
| WRN-28-2 + RA | **1,7** (98,3% de acurácia) | **só o train** | Cubuk et al. 2020 |
| WRN-28-10 + TA | 1,89 (98,11%) | **só o train** | Müller & Hutter 2021 |

- **Referências:** S. Zagoruyko, N. Komodakis. *Wide Residual Networks.* BMVC 2016.
  https://arxiv.org/abs/1605.07146 . G. Huang, Z. Liu, L. van der Maaten, K. Q. Weinberger. *Densely Connected
  Convolutional Networks.* CVPR 2017. https://arxiv.org/abs/1608.06993 . T. DeVries, G. W. Taylor. *Improved
  Regularization of Convolutional Neural Networks with Cutout.* 2017. https://arxiv.org/abs/1708.04552
- **Leitura:** no SVHN com extra, DenseNet e WRN com dropout ficam todas entre 1,6 e 1,8%, e a arquitetura pesa
  menos que regularização e augment. No core, a única comparação justa que achei (RA) mostra que a WRN-28-2
  pequena (1,5 M) empata com a WRN-28-10 (36 M): 98,3 contra 98,3. Ou seja, **no core, capacidade não é o
  gargalo, augment é.** Isso bate com nossa observação de que modelos maiores não ajudaram.
- **Detalhe útil:** WRN e DenseNet usam **dropout entre as convoluções no SVHN** (0,4 na WRN, 0,2 na DenseNet).
  Segundo o artigo da WRN, o efeito do dropout é mais evidente no SVHN, porque ali eles não usam augmentation e o
  BatchNorm sofre overfitting. Em `svhn_dgx.py`, o padrão é `wrn(..., dropout=0.0)`. Se as rodadas não passaram
  outro valor, nossa WRN-16-4 foi treinada **sem** dropout, e a WRN-16-4 com dropout 0,4 é uma configuração
  barata a testar antes de desistir da WRN.
- **Ganho esperado de trocar de arquitetura:** baixo. A DenseNet-BC (k=12, L=100, 0,8 M) é candidata a **membro
  diverso** do ensemble, por ter viés indutivo diferente da CNN e da WRN. Custo: médio, porque a DenseNet é lenta
  por causa das concatenações.

---

## 7. Semi-supervisionado usando só o próprio train como não rotulado

- **Ponto central:** os métodos SSL (Π-model, Mean Teacher, UDA, FixMatch) ganham porque usam **imagens sem
  rótulo adicionais**. Aqui, as únicas imagens permitidas são as do train, que já estão rotuladas. Usar o
  **teste** ou o **conjunto do professor** como não rotulado (transdução, pseudo-rótulos) é **proibido** pelo
  enunciado. Sobra a **regularização por consistência** sobre o próprio train, o que equivale a "augment + perda
  de consistência entre duas visões".
- **Evidência direta (o melhor achado desta seção):** A. Tarvainen, H. Valpola. *Mean teachers are better role
  models.* NeurIPS 2017. https://arxiv.org/abs/1703.01780 . Tabela 1, SVHN, ConvNet de 13 camadas, coluna
  **"73257 labels / 73257 images"** (todos os rótulos, sem extra): supervisionado 2,75 ± 0,10, **Π-model 2,50 ±
  0,07**, **Mean Teacher 2,50 ± 0,05** (média de 4 execuções). São **−0,25 ponto de erro** só com consistência
  sobre o próprio train. Ressalvas: o modelo usa EMA nos dois casos (já temos), o augment é fraco (translação +
  ruído gaussiano) e o custo é ~2 passadas por passo. Com RA/TA ativos, o ganho adicional deve ser menor.
- **UDA:** Q. Xie, Z. Dai, E. Hovy, M.-T. Luong, Q. V. Le. *Unsupervised Data Augmentation for Consistency
  Training.* NeurIPS 2020. https://arxiv.org/abs/1904.12848 . Com só **1.000** rótulos do SVHN e o resto do train
  como não rotulado, o UDA (RA) chega a **2,23%** de erro. Segundo o texto da Tabela 3, isso "iguala" o modelo
  totalmente supervisionado com 73.257 rótulos **sem RA** (2,84%). Leitura: o ingrediente forte é o RandAugment,
  e com todos os rótulos o próximo passo natural é RA supervisionado (seção 1), não UDA.
- **FixMatch:** K. Sohn et al. *FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence.*
  NeurIPS 2020. https://arxiv.org/abs/2001.07685 . Não encontrei no artigo um resultado com todos os rótulos do
  SVHN. O artigo inclui os dados rotulados no conjunto não rotulado, mas isso não substitui dados novos.
- **Faria sentido aqui?** Só como **regularizador de consistência** (Π-model: perda de MSE ou KL entre as
  predições de duas augmentations da mesma imagem; ou Mean Teacher com o EMA que já temos como professor). Ganho
  esperado +0,1 a +0,25 num modelo sozinho, a confirmar. Pseudo-rótulo e FixMatch sem dados novos: ≈ 0.

---

## 8. Técnicas específicas de ensemble

### 8.1 Snapshot ensembles
- **Referência:** G. Huang, Y. Li, G. Pleiss, Z. Liu, J. E. Hopcroft, K. Q. Weinberger. *Snapshot Ensembles:
  Train 1, get M for free.* ICLR 2017. https://arxiv.org/abs/1704.00109
- **Resultado no SVHN (Tabela, erro %, orçamento de 40 épocas, 5 ciclos de 8):** ResNet-110 com modelo único
  1,96 e snapshot 1,63–1,66; WRN-32 1,90 → 1,51–1,64; DenseNet-40 1,77 → 1,64–1,73.
- **Crítica:** o split é ambíguo, provavelmente com extra (ver a leitura crítica no início). O mais importante é
  que o ganho é **com o mesmo custo de 1 treino**. Nós já pagamos 20 treinos independentes, que são mais
  diversos que snapshots de uma trajetória. Fort et al. 2019, com modelos treinados no CIFAR-10 e avaliados no
  SVHN como fora da distribuição, medem divergência JS de 0,384 entre execuções independentes contra 0,153 dentro
  de uma trajetória. **Ganho esperado no nosso ensemble ≈ 0.** Só faria
  sentido para *aumentar* o número de membros a custo zero, por exemplo com 2–3 snapshots por semente.

### 8.2 FGE (Fast Geometric Ensembling)
- **Referência:** T. Garipov, P. Izmailov, D. Podoprikhin, D. Vetrov, A. G. Wilson. *Loss Surfaces, Mode
  Connectivity, and Fast Ensembling of DNNs.* NeurIPS 2018. https://arxiv.org/abs/1802.10026
- **Resultado (Tabela 1, erro %, CIFAR-10, WRN-28-10):** com 1 orçamento, independentes 3,82, SSE 3,73 e FGE 3,65;
  com 3 orçamentos, **independentes 3,31**, SSE 3,55 e FGE 3,52. Com orçamentos maiores, **ensembles
  independentes ganham**, e nós já estamos nesse regime. **Ganho esperado ≈ 0.**

### 8.3 Diversidade de membros (hyper-deep ensembles)
- **Referência:** F. Wenzel, J. Snoek, D. Tran, R. Jenatton. *Hyperparameter Ensembles for Robustness and
  Uncertainty Quantification.* NeurIPS 2020. https://arxiv.org/abs/2006.13570 . Ensembles que variam
  **hiperparâmetros** (por exemplo weight decay e dropout) além da semente superam deep ensembles comuns (WRN-28-10
  e ResNet-20 em CIFAR). Li só o resumo e **não conferi os números**.
- **Base empírica da diversidade:** S. Fort, H. Hu, B. Lakshminarayanan. *Deep Ensembles: A Loss Landscape
  Perspective.* 2019. https://arxiv.org/abs/1912.02757
- **Para nós:** nossos 20 membros são idênticos exceto pela semente. Uma variação barata e coerente com o achado
  de que o augment é o gargalo: membros com **políticas de augment diferentes** (TA, RA e AA-SVHN alternados,
  cutout 8 e 12). O teste do ensemble heterogêneo WRN+CNN não melhorou, mas lá o membro trocado era
  individualmente mais fraco no ensemble. **Ganho esperado:** +0,05 a +0,2. Custo de treino zero se os membros
  forem trocados, e não acrescentados.

### 8.4 Calibração e temperature scaling antes da média
- **Referências:**
  - C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. *On Calibration of Modern Neural Networks.* ICML 2017.
    https://arxiv.org/abs/1706.04599 . O temperature scaling **não muda o argmax de um modelo**. No SVHN
    (ResNet-152 SD, train + extra) o erro fica em 1,98% antes e depois, e o ECE cai de 0,44% para 0,17%.
  - R. Rahaman, A. H. Thiery. *Uncertainty Quantification and Deep Ensembles.* NeurIPS 2021.
    https://arxiv.org/abs/2007.08792 . O temperature scaling deve ser aplicado **depois** da média do ensemble.
    Com mixup, ensembles ficam subconfiantes.
  - Y. Wen, G. Jerfel, R. Muller, M. W. Dusenberry, J. Snoek, B. Lakshminarayanan, D. Tran. *Combining Ensembles
    and Data Augmentation can Harm your Calibration.* ICLR 2021. https://arxiv.org/abs/2010.09875 . Ensemble +
    augment que suaviza rótulos (mixup) **mantém o ganho de acurácia**, mas fica subconfiante. O artigo propõe o
    CAMixup.
  - A. Ashukha, A. Lyzhov, D. Molchanov, D. Vetrov. *Pitfalls of In-Domain Uncertainty Estimation and Ensembling
    in Deep Learning.* ICLR 2020. https://arxiv.org/abs/2002.06470 . Compara métodos de ensemble com a
    temperatura ótima (*calibrated log-likelihood*).
- **Para nós:** uma temperatura por membro antes da média só altera o argmax do ensemble nos exemplos de fronteira.
  Não encontrei evidência de ganho de acurácia relevante. **Ganho esperado ≈ 0 (±0,05).** É barato (1 escalar por
  membro, ajustado na validação) e vale apenas se o trabalho reportar NLL ou ECE. Se adotarmos mixup, recalibrar
  depois da média é obrigatório para as métricas probabilísticas. Uma alternativa de custo igual é a **média dos
  logits** (ou dos log-probs) em vez do softmax, que também quase não muda a acurácia. Não encontrei referência
  com número para isso.

---

## Plano de teste sugerido (ordem, com as regras de validação do PLANO.md)

1. **Rodada A (augment, 1 semente, 60 e 120 épocas):** referência R, R com TA no lugar de rotação/escala/translação
   (+ cutout), R com RA (N=3, M=9), R com AA-SVHN. Os vencedores (≥ +0,15 ponto) seguem para as sementes 43 e 44.
2. **Rodada B (sobre o vencedor de A):** + SAM (ρ ∈ {0,02; 0,05}), + batch augmentation M=2, + consistência
   Π-model, + manifold mixup α=2 (sem LS). Cada um isolado, depois as combinações.
3. **Ensemble:** refazer o ensemble de 20 com a melhor receita. Depois, a variante "diversa" (membros alternando
   TA/RA/AA). Comparar com o atual (97,52% com TTA) de forma pareada.
4. Opcional, se o trabalho valorizar inferência barata: destilar o ensemble final num modelo (τ = 4, augment
   consistente, professor sem LS se possível).
