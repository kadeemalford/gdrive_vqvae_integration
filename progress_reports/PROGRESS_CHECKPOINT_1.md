# PROGRESS CHECKPOINT 1 — Reconstrução generalista FEBUS

Síntese dos **relatórios 1–59**, cobrindo a investigação de transferência do
VQ-VAE DAS offshore para doped, multicore, multimode e single-mode. Resultados
são de desenvolvimento; não há modelo final promovido ou validação externa.

## Principais avanços

- **Base experimental confiável (1–14, 32–35):** avaliação por família/configuração,
  rastreabilidade física/espacial e proteção reversível de extremos. A proteção
  adaptativa corrigiu 40 janelas problemáticas, sem explicar o erro das demais;
  correção numérica não equivale a ganho aprendido do modelo.
- **Representação e orçamento latente (16–24, 29–43):** alterar a alocação
  canal–tempo e aumentar a densidade de índices produziram ganhos importantes,
  assim como otimizar a referência transferida à mesma taxa no relatório 22.
  O benefício de uma geometria específica variou entre folds; treino do zero
  temporal superou scratch nativo, mas não a transferência naquele orçamento.
- **Seleção e limites (44–59):** validação completa, caudas por canal e guarda
  por configuração revelaram perdas escondidas pela média. O objetivo por canal
  passou na guarda em duas épocas do fold 0, mas não confirmou preservação no
  fold 1. Ganho global, maior uso de códigos e mais treino não foram garantias.

## Ganhos relevantes antes da curva do relatório 37

A referência que entrou no relatório 37 **já acumulava adaptações e ganhos de
fidelidade**. Os principais contrastes anteriores foram:

| Etapa | Intervenção | Correlação antes → depois | RMSE relativo antes → depois | Custo em compressão |
|---|---|---:|---:|---|
| [Relatório 2](2-FINETUNE_EVALUATION_REPORT.md) | Fine-tuning do offshore em FEBUS | 0,1664 → 0,3238 | 0,9896 → 0,9460 | Mesma quantidade de índices |
| [Relatório 17](17-TEMPORAL_DENSITY_FULL_RESULTS_AND_LONG_CONTINUATION.md) | Menor downsampling temporal, com mais índices e treino | 0,4503 → 0,5020 | 0,8676 → 0,8403 | Compressão nominal pooled: 620,2× → 165,9× |
| [Relatório 18](18-TEMPORAL_DENSITY_LONG_RESULTS_AND_DECISION_GATES.md) | Continuação do modelo temporal denso | 0,5020 → 0,5600 | 0,8403 → 0,8098 | Mesma taxa do modelo denso |
| [Relatório 22](22-ARCHITECTURE_INITIALIZATION_RESULTS.md) | Treinamento adicional da referência transferida, com novo regime de otimização | 0,5600 → 0,6823 | 0,8098 → 0,7199 | Mesma taxa: 165,9× |
| [Relatório 30](30-RATE_ALLOCATION_RESULTS.md) | Redistribuição espaço-temporal dos índices, com treino adicional | 0,6823 → 0,7098 | 0,7199 → 0,6859 | Taxa quase igual: 165,9× → 164,6× |

**Não juntar as linhas como uma única curva de avaliação:** o relatório 2 usa
a agregação total por janela do teste histórico do fold 0; os demais usam métricas
macro da validação do fold 1. Cada contraste deve ser interpretado dentro de seu
protocolo. Essa validação foi reutilizada no desenvolvimento, não é confirmação
externa independente.

O **relatório 22 demonstra um avanço expressivo sem aumentar o bitrate**:
65.536 atualizações adicionais, mesma geometria temporal densa e LR de `1e-4`.
Isso evidencia o papel da otimização/treinamento, mas não isola o efeito do
learning rate do orçamento adicional e das demais condições do regime.

Os ganhos também tiveram limites: o fine-tuning inicial piorou multimode;
o ganho macro da grade espacial do relatório 30 não se confirmou no outro fold
([relatório 32](32-FOLD_CONFIRMATION_AND_RESIDUAL_DYNAMIC_RANGE.md)). Nenhuma
dessas médias garante preservação de cada canal ou configuração.

**Os multiplicadores mudam de referência entre estudos:** o temporal-4× dos
relatórios 17–22 opera perto de 165,9× de compressão nominal; o 4× do relatório
37 opera perto de 42,1×. Não são a mesma geometria nem o mesmo orçamento.

## Curva posterior e custo em compressão

Na mesma validação completa do fold 1 ([relatório 37](37-NEURAL_RATE_DISTORTION_RESULTS.md)):

| Orçamento neural | Correlação macro | RMSE relativo macro | Compressão nominal pooled |
|---|---:|---:|---:|
| 1× | 0,7173 | 0,6805 | 164,6× |
| 2× | 0,7962 | 0,5904 | 83,8× |
| 4× | 0,8656 | 0,4900 | 42,1× |

1×/2×/4× são rótulos dessa curva FEBUS, **não o modelo offshore original**:
significam uma, aproximadamente duas e quatro vezes a quantidade de índices
da referência dessa curva. O 1× já treinado não é o ponto inicial congelado:
este tinha correlação **0,7098** e RMSE **0,6859**, com os mesmos 164,6× nominais.

Melhor agregado posterior: **correlação 0,8757 / RMSE 0,4668**, ~42,8×, no fold 0,
com regressão localizada. A continuação histórica do fold 1 atingiu
**0,8712 / 0,4777**, ~42,1×, preservando retrospectivamente os três critérios
contra seu source. São checkpoints/validações diferentes, não uma única rede
confirmada nos dois folds ([relatório 59](59-GENERALIST_RESEARCH_REFERENCE_REGISTRY.md)).

**Não atribuir todo o ganho ao fine-tuning.** O offshore original sem adaptação
transmite 2.016 índices numa janela 512×1.000; a variante densa transmite 32.000
(~15,9 vezes mais). Ambos continuam VQ, K=2.048, 11 bits por índice. Só nos índices
e nessa mesma janela, a compressão cai de **738,8× para 46,5×**. Os ~42–43× acima
incluem side e agregam shapes do corpus: não têm a mesma base do 738,8× original.
Container/modelo e codificação entrópica não estão integralmente contemplados.

## O que não se consolidou como solução

Especialização, refresh de códigos, normalização por canal, calibração de ganho
e mudanças de loss tiveram resultados pequenos ou compromissos entre grupos.
Bypass VQ contínuo melhorou cinco configurações, mas piorou todas as janelas
multimode, mesmo igualando energia latente. Isso não prova que quantização seja
ótima ou que arquitetura seja a única causa: o decoder recebe outra distribuição.
Persistem erros de amplitude, espectro e canais; correlação macro alta não basta.

## Artigo avaliado

A avaliação do [artigo CNN–Transformer/OWR](../CBA_2026_paper__7079.pdf)
considera **válida a adaptação do processamento convolucional/contexto temporal,
não a adoção de OWR como núcleo do codec**. Novos modos, núcleos ou dopantes são
principalmente desafios de domínio/geometria neste objetivo de reconstrução;
não exigem criar classes e decoders. No artigo, a CNN pura reconstruiu melhor
que CNN+Transformer na comparação sorteada; atenção permanece hipótese opcional.
Seus ciclos filtrados/alinhados e latente contínuo não demonstram nossa taxa/fidelidade.

## Mudança fundamental de direcionamento recomendada

**Deixar de presumir o fine-tuning offshore como solução e comparar eficiência
de representação neural no corpus FEBUS.** O modelo offshore permanece como
controle, não restrição. Os relatórios 21–24 já testaram scratch em outro regime;
não provaram que a transferência é o gargalo nem encerraram novas arquiteturas.

| Braço inicial | Arquitetura | Inicialização/treino |
|---|---|---|
| A | Nativa do modelo offshore | Pesos offshore, avaliados antes e após adaptação controlada em FEBUS. |
| B | Mesma nativa | Aleatória, treino FEBUS-only. |
| C | CNN residual multiescala anisotrópica | Aleatória, treino FEBUS-only; representação espacial e temporal explícita. |

Depois, se justificado, comparar **C versus C+atenção temporal/fatorizada**.
Isso aproveita o artigo sem adicionar OWR e sem atribuir ao Transformer um
benefício que sua comparação de reconstrução não demonstrou.

O VQ-VAE atual **já é uma CNN profunda residual**. Apenas chamar C de “Deep CNN”
não constitui mudança fundamental. A variante `temporal-dilated-v1` já testada
troca uma convolução de cada bloco residual por uma 1×9 dilatada, preservando a
topologia geral ([factory local](../finetuning/model_factory.py)). C deve explicitar
mudanças além disso: blocos fatorizados espacial/temporal, contexto multiescala
antes do gargalo e desenho independente de redução/reconstrução dos eixos.
São hipóteses a especificar e testar, não uma rede nova pronta derivada do PDF.

Para C, evitar tratar eixos canal/tempo como intercambiáveis. Capturar contexto
temporal maior e relações espaciais apenas onde contiguidade/topologia forem
válidas; não misturar núcleos/arquivos como vizinhos artificiais. Preservar shapes
variáveis com padding/máscaras/posições explícitos. Atenção global sobre todas
as 512×1.000 amostras seria custosa; operar em tokens reduzidos ou eixos fatorizados.

**Condições para um experimento que realmente decida o caminho:**

1. Mesmo corpus FEBUS e splits por aquisição/configuração, incluindo as quatro
   famílias; não trocar dados e arquitetura simultaneamente. Não filtrar o alvo.
2. Mesmo gargalo VQ inicial, K/dimensão de embedding controlados e pelo menos
   dois orçamentos de bits comuns: ponto compacto e ponto de maior fidelidade.
   Contar índices reais por shape, padding e side, não só comparar dimensões latentes.
3. Decoder recebe somente representação transmitida: sem skips contínuos do
   encoder que contornem o gargalo, salvo se codificados e cobrados.
4. Loss/normalização comuns na primeira comparação; não combinar atenção,
   novo quantizador, nova loss e novo corpus em uma única mudança sem controles.
5. Capacidade/compute medidos, treino de scratch com oportunidade de convergência,
   triagem limitada e extensão seletiva. Pré-treino offshore não é compute gratuito
   ou orçamento total equivalente ao scratch.
6. RMSE/correlação, amplitude/fase/bandas e caudas por configuração, além de
   latência/memória e taxa. Eficiência = mesma fidelidade com menos bits, ou
   ganho sustentado à mesma taxa; não simplesmente transmitir mais posições.
7. Ampliar dados FEBUS/compatíveis em ablação separada e preservar aquisições
   independentes para confirmação. Mais janelas da mesma gravação não criam
   diversidade equivalente a novas aquisições.

Limiares físicos finais e aquisições externas são necessários para promoção,
mas não impedem uma triagem arquitetural exploratória bem controlada. Declarar
essa diferença para não transformar requisitos de aceitação em bloqueio a toda
investigação antes de existir um candidato.

Não há prova de que C vencerá nem de que o corpus atual é insuficiente. Há
justificativa para **mudar a pergunta experimental** e interromper a sequência
de refinamentos marginais da mesma representação. A solução pretendida segue
neural; ampliar benchmarks clássicos não é pré-requisito para essa decisão.

Também não há evidência de que outra arquitetura devolverá os 738,8× originais
com a fidelidade desejada: detalhes/ruído pouco previsíveis podem exigir mais
bits. Essa curva deve ser medida, não prometida, e o bypass não estabeleceu um
limite de informação. O ganho procurado é eficiência de representação demonstrada.
