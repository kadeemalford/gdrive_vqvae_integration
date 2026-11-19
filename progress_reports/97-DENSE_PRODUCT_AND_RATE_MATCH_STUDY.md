# Produto VQ denso e curva de taxa congelada

**Etapa consolidada dos registros 105, 106, 107, 108.** Economia de bytes com mais erro; a grade medida não demonstrou dominância RD.

Protocolos, resultados e auditorias foram reunidos sem alterar resultados, budgets ou gates. As partes históricas seguem em ordem: estados de lançamento/execução descrevem aquele momento, não tarefas ainda pendentes. Consulte as últimas partes para o encerramento.

## Registros desta etapa

- [Registro 105](#report-105) — Produto VQ denso full-tile: piloto com taxa nominal comum e métricas prospectivas
- [Registro 106](#report-106) — Produto full-tile denso: economia de bytes sem melhoria de fidelidade
- [Registro 107](#report-107) — Curva congelada para comparar flat e produto por bytes reais
- [Registro 108](#report-108) — Curvas congeladas: nenhuma dominância demonstrada na grade fixada

<a id="report-105"></a>

## Registro histórico 105 — Produto VQ denso full-tile: piloto com taxa nominal comum e métricas prospectivas

Fonte original preservada no arquivo: `105-DENSE_FULL_TILE_PRODUCT_PROTOCOL.md`.

<a id="report-105-produto-vq-denso-full-tile-piloto-com-taxa-nominal-comum-e-métricas-prospectivas"></a>

**15/11/2026. Registrado antes da preparação/treino.**

<a id="report-105-contraste-único"></a>

#### Contraste único

Flat2048 versus produto32×64 no VQ **denso full-tile** maduro, mesma fonte
época 23 e configuração `spatial4-temporal4`. Ambos têm uma grade de IDs compostos
de 2.048 estados (11 bits nominais por posição), mesmo encoder/decoder inicial,
normalização, padding, loss, dados e coder FAK1. Produto não adiciona outro stream
ou índices: ID = left×64 + right. Taxa real pode diferir e será cobrada.

Motivação e limites no relatório 104. O estudo 89 compacto mantém seu gate
reprovado e resultados históricos; não será reclassificado por este piloto.
O produto tem menos centroides livres e inicialização recém-ajustada, enquanto
o flat herda um dicionário maduro. Não alegar capacidade/função inicial idêntica.

<a id="report-105-preparação-train-only"></a>

#### Preparação train-only

44 recortes centrais de ajuste já fixados. No máximo 256 vetores de encoder por
recorte, amostrados por seed **2117** sem reposição. PCA ortogonal completa
128D, sem whitening/truncamento; sinal de autovetores canônico e distribuição
alternada das direções entre duas metades. Lloyd farthest-first de 12 iterações,
32/64 centroides em 64D por metade. PCA/centro congelados, livros treináveis.
Publicar assets e SHA; nenhum sinal de checks participa dessa preparação.

<a id="report-105-treino-e-cobertura"></a>

#### Treino e cobertura

- Mesmas 5.600 janelas dos 44 arquivos fit; excluir 43 arquivos de checks de
  todos os batches novos. Fonte já os viu historicamente.
- Uma janela nativa por update, sequência pareada seed 2117.
- **8.192 updates por braço, 16.384 científicos no total**. Não há extensão
  automática neste piloto; a próxima decisão usa evidência completa publicada.
- AdamW novo em ambos, LR1e-4, wd1e-5, clip1, FP32 determinístico, sem scheduler.
- Loss histórica 0,5 MSE + 0,5 L1 mascaradas + VQ. Não mudar loss junto ao produto.
- Forward hard usa vetores exatos e ponte de gradiente zero `z-z.detach()`.
- Calibração FAK1 somente nos 44 recortes fit. Literal/escapes, mapas, registros
  e framing pagos igualmente. Pesos exportados completos separados da taxa.

Checks do piloto: um maior recorte por configuração, mais o menor tile por
arquivo check, deduplicados por metadados. Incluem constantes e caudas, sem
denominador patológico. Seleção inicial elegível; score fixo:
`macro_famílias NRMSE*² + 0,1 * bpc_stream / 0,7`.
Não comparar essa seleção a rankings com outra população/contrato.

Após 8.192, avaliar selecionados/endpoints nos **136 checks train-only** fixados
no diagnóstico 103, sem reseleção. Publicar fonte madura congelada na mesma
população e calibração, sem igualar coortes diferentes por arredondamento.

<a id="report-105-integração-decisão-e-recursos"></a>

#### Integração, decisão e recursos

Antes do piloto, dois updates de smoke por braço e teste 1+resume versus 2
contínuos: parâmetros, momentos AdamW e RNG precisam ser bit-exatos. Livros
compostos/gradientes finitos e receptor independente só com índices/registro.
Oito updates técnicos no total. Implementação versionada separada de 89.

Investimento adicional somente se endpoint do produto: melhorar ≥1% contra
inicialização, ≥0,2% no último bloco (6.144→8.192) e ficar até 3% acima do
controle selecionado em erro. O produto selecionado deve custar até 105% dos
bytes do controle selecionado, com perplexidade ≥16 e replay/cobertura válidos.
Isso não é promoção: a fronteira prática exige melhoria de erro ≥5% sem mais
bytes, grupos e guardas físicas, além de confirmação independente.

Não retunar esses limiares, PCA, divisão de bits ou inicialização pelos checks.
Não abrir validação/teste/EDAS para esta triagem; não retomar treino negativo
apenas por ter concluído o piloto.

GPU 0 flat, GPU 1 produto. Destino:
`results/experiments/dense_full_tile_product_fold0_20261115/`.
O supervisor faz preparação → smoke/resume → piloto pareado → avaliação ampliada.
Sem commit/push, resultados históricos preservados e cópia remota sincronizada.

<a id="report-105-integração-seleção-intermediária-não-divergência-numérica"></a>

#### Integração: seleção intermediária, não divergência numérica

A primeira integração parou **antes de qualquer update científico**. Os dois
braços passaram igualdade de parâmetros, momentos AdamW e RNG entre 2 updates
contínuos e 1+resume. Contudo, o teste exigia igualdade do checkpoint inteiro:
o caminho interrompido avaliou também update 1, enquanto o contínuo avaliou
apenas update 2. No produto, update 1 foi melhor e permaneceu selecionado
no resume, divergindo somente `best` e `selected_update`. O flat não apresentou
essa diferença por acaso de ranking.

Correção: avaliar **ambos os updates em todo smoke**, igualando os eventos de
seleção antes de exigir novamente igualdade do checkpoint inteiro. Não relaxar
a guarda e não alterar a cadência científica de 2.048 updates.

Primeira tentativa preservada em `dense_full_tile_product_fold0_20261115/`;
reteste/piloto em `dense_full_tile_product_v2_fold0_20261115/`. Reutilizar os
mesmos assets já ajustados e verificados da primeira tentativa, sem nova PCA
ou busca de inicialização. Total técnico previsto passa a **16 updates** entre
as duas integrações; teto científico permanece **16.384**.

<a id="report-105-integração-corrigida-e-piloto-iniciado"></a>

#### Integração corrigida e piloto iniciado

O reteste v2 passou **igualdade do checkpoint inteiro** nos dois braços,
incluindo parâmetros, momentos, RNG, score acumulado e update selecionado.
Evidência em `RESUME_INTEGRATION.json`; a guarda não foi reduzida.

O supervisor iniciou os pilotos: GPU 0 flat, PID inicial 1659834; GPU 1 produto,
PID inicial 1659835. São 8.192 updates científicos por braço, seguidos da
avaliação ampliada automática. Os comandos constam em `LAUNCH_pilot_*.json`.
**442 testes locais passaram.** Ainda sem resultado científico nesta atualização.

<a id="report-105-encerramento"></a>

##### Encerramento

Os dois braços concluíram 8.192 updates e a avaliação ampliada. O gate reprovou
continuação; resultado e limites no [relatório 106](97-DENSE_PRODUCT_AND_RATE_MATCH_STUDY.md#report-106).
Não houve extensão ou acesso à validação.

---

<a id="report-106"></a>

## Registro histórico 106 — Produto full-tile denso: economia de bytes sem melhoria de fidelidade

Fonte original preservada no arquivo: `106-DENSE_FULL_TILE_PRODUCT_RESULTS.md`.

<a id="report-106-produto-full-tile-denso-economia-de-bytes-sem-melhoria-de-fidelidade"></a>

**15/11/2026.** Resultado do protocolo 105; sem extensão ou promoção.

<a id="report-106-execução-e-seleção"></a>

#### Execução e seleção

Após corrigir a cadência de avaliação do smoke, os dois braços passaram
igualdade do checkpoint inteiro entre execução contínua e retomada. A tentativa
técnica anterior permanece preservada; não executou o piloto científico.

O piloto v2 concluiu **8.192 updates por braço / 16.384 científicos no total**,
com **655.595.000 células por braço**, mesma sequência e máscara. Os 16 updates
técnicos das duas integrações são discriminados desse total.

Flat selecionou **update 0**, preservando a fonte madura. Produto selecionou
**4.096**, sem reseleção pelos checks ampliados. O flat adaptado não deslocou
seu ponto inicial sob o score registrado.

<a id="report-106-gate-reprovado"></a>

#### Gate reprovado

Produto inicial: NRMSE* 0,477924; endpoint: 0,472532. Há aprendizagem contra
a inicialização, mas a melhora 6.144→8.192 foi **0,159%**, abaixo de 0,2%.
O endpoint ficou **4,82% acima** do flat selecionado, fora da proximidade de 3%.
Bytes e saúde do quantizador passaram; perplexidade média do último bloco ~579.
Não relaxar os dois critérios reprovados nem prolongar o treino.

<a id="report-106-avaliação-ampliada-congelada"></a>

#### Avaliação ampliada congelada

**136 recortes / 43 arquivos / dez configurações / quatro famílias /
12.799.000 células**, incluindo os menores tiles e constantes do protocolo.
Replay físico de todos os selecionados, endpoints e referências foi bit-exato
com receptor separado. Registro FAK1 ajustado somente nos 44 recortes fit.

| Modelo | Update | NRMSE* macro famílias ↓ | Stream bpc ↓ |
|---|---:|---:|---:|
| Flat selecionado = fonte madura | 0 | **0,490336** | 0,679244 |
| Produto selecionado | 4.096 | 0,508095 | **0,654265** |
| Flat endpoint | 8.192 | 0,491848 | 0,678543 |
| Produto endpoint | 8.192 | 0,512216 | 0,652998 |

Entre selecionados, produto custa **3,68% menos bytes**, mas tem **3,62% mais
erro**, piorando nas dez configurações. Isso não demonstra dominância a taxa
real pareada: a paridade de 11 bits nominais por posição não iguala a entropia.

Auditoria pareada: **40/43 arquivos** com NRMSE* maior, **41/43** com p90
maior, 29/30 pares configuração×banda com erro espectral maior e 20/30 com
pior distância de amplitude RMS a 1. Fase continua descritiva e com cobertura
publicada, não uma guarda operacional certificada.

Exports completos: flat 111.867.797 B; produto 110.910.735 B. PCA e centro
estão incluídos; encoders também. Não são exports mínimos de receptor.

Artefatos: `results/experiments/dense_full_tile_product_v2_fold0_20261115/`,
com `INVESTMENT_GATE.json`, seleção, histórico, snapshots de fontes, estados,
frames e `PHYSICAL_SELECTED_COMPARISON.json`. Os resultados foram recuperados
localmente, com duplicatas técnicas de pesos preservadas no remoto.

<a id="report-106-decisão-e-diagnóstico-seguinte"></a>

#### Decisão e diagnóstico seguinte

Encerrar o treinamento 105, **sem extensão, P3 ou acesso à validação**.
O resultado não confirma a hipótese de ganho de fidelidade por geometria do
dicionário; tampouco basta comparar pontos com bytes diferentes para refutar
toda possível vantagem RD.

Antes de outra arquitetura, executar uma curva **congelada de decisão de índices
com penalidade de taxa**, em ambos os selecionados, sem novas posições ou pesos.
Isso verifica se a economia do produto sobrevive quando o flat também pode
trocar um pouco de erro por índices baratos. Não é uma nova hipótese de prior
neural nem apaga os testes históricos de busca/RD. É uma checagem delimitada
de comparabilidade dos dois pontos atuais, com framing e decoder reais.

Os parâmetros e o escopo são fixados no protocolo 107 antes desse diagnóstico.

---

<a id="report-107"></a>

## Registro histórico 107 — Curva congelada para comparar flat e produto por bytes reais

Fonte original preservada no arquivo: `107-FROZEN_DENSE_RATE_MATCH_PROTOCOL.md`.

<a id="report-107-curva-congelada-para-comparar-flat-e-produto-por-bytes-reais"></a>

**15/11/2026. Antes da inferência desta etapa.**

<a id="report-107-escopo-e-hipótese-limitada"></a>

#### Escopo e hipótese limitada

No piloto 105, o produto selecionado economizou 3,68% de bytes e piorou 3,62%
em erro. Não prolongar o piloto reprovado. Medir se isso oferece alguma vantagem
RD sobre o flat quando ambos usam decisões de índice penalizadas por taxa.
Essa é uma checagem de taxa, não novo treinamento ou promessa arquitetural.

Fixar os checkpoints selecionados **antes** desta curva:
flat update0 e produto update4096 do piloto v2. Encoder, decoder, livros, PCA,
normalização e grades permanecem congelados. Não varrer epochs/endpoints.

<a id="report-107-decisão-de-índices-e-calibração"></a>

#### Decisão de índices e calibração

Para cada vetor z, escolher em float32:

`argmin_k( ||z-q_k||² / 128 + lambda * -log2(p_k) )`.

As probabilidades são as mesmas contagens suavizadas FAK1 estimadas nos 44
recortes fit, **sem reajuste por lambda ou pelos checks**. A penalidade é um
proxy de taxa; o stream real é codificado e medido, inclusive fallback fixed,
side, registros, mapas e framing.

Calibrar uma escala por checkpoint somente nesses 44 recortes:
`scale = MSE latente pooled(z,q_nearest) / self-information média nearest`.
Grade predefinida: `lambda = scale × [0, 0.03, 0.1, 0.3, 1, 3]`.
Zero deve reproduzir índices e waveform hard do respectivo checkpoint.
As médias pooled de latentes são calibração heurística, não surrogate físico
certificado ou um limite informacional.

Serão **12 pontos** completos, sem otimização neural, passos adaptativos ou
busca por janelas. Escolhas dependentes do alvo feitas no encoder são válidas
somente porque os índices resultantes são efetivamente transmitidos. O receptor
não precisa da penalidade: reconstrói dos mesmos índices, livros e side.

<a id="report-107-avaliação-limites-e-decisão"></a>

#### Avaliação, limites e decisão

Mesmos **136 checks train-only / 43 arquivos** do protocolo 105; calibração de
44 recortes fit separada por arquivo. Sem waveforms val/test/EDAS.
Pool físico prospectivo, mesmas escalas e modo literal. Receptor separado,
índices/CRC efetivos e escapes bit-exatos. Publicar todas as curvas, regressões,
pesos e bytes, não só o melhor ponto.

Comparações de dominância usarão apenas pontos realmente medidos: candidato
com **bytes não maiores** e erro menor, sem interpolar para certificar igualdade
de taxa. Ausência de ponto dominante numa grade finita não prova otimalidade
nem autoriza mais sweep. Não transformar uma melhora de proxy latente em
melhora física.

Qualquer positivo escalar ainda exige grupos e guardas físicas; não há promoção,
reseleção de checkpoints ou P3 automático. O gate de treinamento 105 fica
reprovado independentemente da curva.

GPU 0 flat; GPU 1 produto, apenas inferência. Destino:
`results/experiments/frozen_dense_rate_match_fold0_20261115/`.

<a id="report-107-resultado"></a>

##### Resultado

Os 12 pontos concluíram sem updates. Não houve dominância medida entre codecs
na grade fixada; não ampliar a grade retrospectivamente.
Resultado no [relatório 108](97-DENSE_PRODUCT_AND_RATE_MATCH_STUDY.md#report-108).

---

<a id="report-108"></a>

## Registro histórico 108 — Curvas congeladas: nenhuma dominância demonstrada na grade fixada

Fonte original preservada no arquivo: `108-FROZEN_DENSE_RATE_MATCH_RESULTS.md`.

<a id="report-108-curvas-congeladas-nenhuma-dominância-demonstrada-na-grade-fixada"></a>

**15/11/2026.** Resultado do protocolo 107. Zero updates.

Os 12 pontos terminaram: 136 recortes e 12.799.000 células em cada ponto,
com os mesmos pesos, grades, alvos, side e registros fit congelados.
A penalidade zero reproduziu os frames de índices originais; seus scores e
bytes coincidem com a avaliação ampliada de 106.

| Multiplicador | Flat NRMSE* | Flat bpc | Produto NRMSE* | Produto bpc |
|---|---:|---:|---:|---:|
| 0 | 0,490336 | 0,679244 | 0,508095 | 0,654265 |
| 0,03 | 0,490329 | 0,679052 | 0,508120 | 0,654020 |
| 0,1 | 0,490356 | 0,678584 | 0,508177 | 0,653387 |
| 0,3 | 0,490455 | 0,677337 | 0,508425 | 0,651687 |
| 1 | 0,491187 | 0,673249 | 0,510000 | 0,645749 |
| 3 | 0,496846 | 0,661833 | 0,519855 | 0,629293 |

A grade não contém ponto de um codec com erro menor **e bytes não maiores**
que algum ponto do outro. As faixas de taxa medidas não se sobrepõem; não
interpolar ou acrescentar penalidades pós-hoc para certificar taxa pareada.
Isso é **inconclusivo sobre dominância RD geral**, não prova de equivalência
nem de superioridade do produto.

A decisão barata de índice deslocou pouco a curva do flat: o multiplicador
0,03 tem diferença escalar ínfima, sem evidência de ganho físico relevante.
Não promover esse ponto por arredondamento de macro.

**Encerrar 105/107 sem extensão, nova seed, P3 ou validação adicional.**
O gate de treinamento permanece reprovado; as curvas não o reabrem.
O VQ maduro continua a referência prática de alta fidelidade disponível,
sem alegar que seja ótimo em toda taxa.

Artefatos: `results/experiments/frozen_dense_rate_match_fold0_20261115/`.
`MEASURED_FRONTIER.json` preserva os 12 pontos e comparações sem interpolação.

<a id="report-108-continuação-justificada"></a>

#### Continuação justificada

H1/H2 e fatoração não deslocaram a referência prática nos budgets registrados.
Não lançar outra variante da mesma grade por tentativa indiscriminada.
Passar à hipótese H3 já proposta em 90: **condicionamento físico**, começando
por inventário dos cabeçalhos de treino e cobertura de metadados. Só depois
decidir se existe contraste identificável e informação suficiente.

Essa etapa não pressupõe que dx/GL/pulso expliquem causalmente o erro, não usa
nomes de excitação como alvo gratuito e não acessa waveforms reservados.
O protocolo 109 separa metadados de sinais e explicita as lacunas de proveniência
que impedem chamar arquivos diferentes de aquisições independentes.
