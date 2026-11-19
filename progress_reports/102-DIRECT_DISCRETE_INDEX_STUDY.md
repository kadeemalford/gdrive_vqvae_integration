# Decisão discreta de índices e auditoria ampliada

**Etapa consolidada dos registros 118, 119, 120, 121.** Redução escalar com regressões físicas; sem promoção ou distilação.

Protocolos, resultados e auditorias foram reunidos sem alterar resultados, budgets ou gates. As partes históricas seguem em ordem: estados de lançamento/execução descrevem aquele momento, não tarefas ainda pendentes. Consulte as últimas partes para o encerramento.

## Registros desta etapa

- [Registro 118](#report-118) — Decisão discreta de índices: probe limitado, distinto da relaxação E1
- [Registro 119](#report-119) — Índices discretos: ganho escalar pequeno, regressões espectrais importantes
- [Registro 120](#report-120) — Auditoria ampliada do mesmo encoder discreto congelado
- [Registro 121](#report-121) — Busca discreta ampliada: redução de NRMSE* com redistribuição de erro

<a id="report-118"></a>

## Registro histórico 118 — Decisão discreta de índices: probe limitado, distinto da relaxação E1

Fonte original preservada no arquivo: `118-DIRECT_DISCRETE_INDEX_PROBE_PROTOCOL.md`.

<a id="report-118-decisão-discreta-de-índices-probe-limitado-distinto-da-relaxação-e1"></a>

**15/11/2026. Registrado antes dos resultados.**

<a id="report-118-escopo-e-diferença-concreta"></a>

#### Escopo e diferença concreta

E1 já otimizou vetores contínuos e os reprojetou em índices; os relatórios
históricos não são reclassificados. Não anunciar nova família arquitetural.
Testar uma diferença específica: propostas **diretamente no conjunto finito
de índices**, sem Adam, LR ou vetor contínuo otimizado.

Fonte densa madura época23 congelada. Mesmo encoder, decoder, K2048, grade,
FNE1 e FAK1. O receptor continua recebendo apenas índices, pesos e side pagos.

<a id="report-118-algoritmos-e-orçamento-fixados"></a>

#### Algoritmos e orçamento fixados

Shortlist fixa dos oito centroides mais próximos do z original em cada posição.
A cada uma de **duas rodadas**, considerar apenas alternativas nessa shortlist
com auto-informação marginal não maior que a do ID corrente, diferente dele.
Probabilidades fixadas nos mesmos 44 recortes fit, sem recalibração nos checks.

- **Gradiente discreto:** ordenar alternativas pela variação linear do erro
  quadrático normalizado por canal via gradiente do decoder em q discreto.
  Usar apenas variações previstas negativas, escolhendo uma alternativa por posição.
  Nenhum passo contínuo é aplicado ou transmitido.
- **Controle latente:** ordenar pelo aumento de distância quadrática a z,
  dentre as mesmas alternativas de custo marginal não crescente.

Em cada rodada propor alterações em até 1%, 5%, 10% e 20% das posições,
ordenadas pelo respectivo critério. No máximo oito propostas hard por janela
e método. O gradiente custa backward adicional: não alegar FLOPs iguais.

Codificar o frame completo de cada proposta. Só decodificar candidatos cujo
tamanho não exceda **o frame original daquela janela**. Aceitar somente se o
NRMSE* físico macro de canais daquela janela diminuir após restauração float32.
Fonte e melhor candidato anterior são sempre fallback elegível.
Publicar rejeições de bytes/erro/invalidez, sem omitir janelas.

A loss usada para ordenar é média dos NRMSE* quadráticos, não idêntica à média
dos RMSE* nem ao pooling final. Contagens/variância incluem células físicas;
escapes não contribuem com erro porque são restaurados exatamente.
Guardas espectrais/correlação/p90 serão avaliadas depois, não presumidas
aprovadas pela aceitação escalar.

<a id="report-118-população-e-gate"></a>

#### População e gate

Dez janelas **train-only**, uma por configuração: primeiro arquivo check P1
lexicográfico, maior tile e janela central desse tile. Os 44 recortes fit
calibram exclusivamente o coder. Seleção dos recortes antes dos sinais,
sem escolher casos em que o gradiente funcionou. A população favorece tiles
maiores e é desenvolvimento consumido, não confirmação.

GPU0 gradiente; GPU1 controle latente. Zero updates de pesos/codebook; no máximo
20 backward de decoder no braço gradiente, 80 propostas hard por braço.
Todos os frames finais e originais serão salvos e reproduzidos em receptor
carregado separadamente.

Só considerar expansão para 136 checks com ganho ≥1% em NRMSE* macro famílias,
bytes completos não maiores e nenhuma regressão nas dez macros por grupo.
Isso é gate de oportunidade, não promoção. Sem ganho suficiente, encerrar sem
varrer shortlist, frações, rodadas ou LR. A melhoria prática continua exigindo
guardas físicas e comparação adequada; sem P3 automático.

Sem sinais val/test/EDAS. Pesos completos compartilhados discriminados.
Destino: `results/experiments/direct_discrete_index_probe_20261115/`.

<a id="report-118-lançamento-e-engenharia"></a>

#### Lançamento e engenharia

**467 testes locais passaram**, incluindo receptor independente, ausência de
mudança dos pesos, fallback de constantes e cap hard de bytes no probe.
Implementações: `finetuning/direct_discrete_index_probe.py` e
`tools/run_direct_discrete_probe.py`.

O supervisor foi lançado nas duas GPUs, sem treinamento. Os dois processos
precisam concluir antes da comparação; não promover médias sem auditar arquivos,
bytes, fases/caudas e os códigos de saída.

<a id="report-118-resultado-posterior"></a>

##### Resultado posterior

O gradiente passou o gate escalar do piloto; resultado e regressões em
[119](102-DIRECT_DISCRETE_INDEX_STUDY.md#report-119). A ampliação congelada
120 terminou sem avanço físico: decisão em
[121](102-DIRECT_DISCRETE_INDEX_STUDY.md#report-121). Sem promoção ou treinamento.

---

<a id="report-119"></a>

## Registro histórico 119 — Índices discretos: ganho escalar pequeno, regressões espectrais importantes

Fonte original preservada no arquivo: `119-DIRECT_DISCRETE_INDEX_PROBE_RESULTS.md`.

<a id="report-119-índices-discretos-ganho-escalar-pequeno-regressões-espectrais-importantes"></a>

**15/11/2026.** Resultado do protocolo118; zero updates de pesos/codebook.

Dez janelas de dez arquivos/configurações de treino, quatro famílias,
**1.142.000 células**. Os dois processos concluíram com código zero.
Mesma fonte densa, contagens fit, transporte e baseline.

| Método | NRMSE* macro famílias ↓ | Stream bpc ↓ | Erro vs. fonte | Bytes vs. fonte |
|---|---:|---:|---:|---:|
| Fonte | 0,471769 | 0,674438 | — | — |
| Controle latente | 0,471674 | 0,674158 | −0,020% | −0,042% |
| Gradiente discreto | **0,465588** | **0,672476** | **−1,310%** | **−0,291%** |

Cada método avaliou 80 propostas. Gradiente registrou 26 melhorias de incumbente;
controle, nove. Isso não equivale a 26 janelas independentes. Todos os frames
finais passaram replay no receptor separado e cap de bytes original por janela.
Não houve otimização contínua de vetores, LR ou mudança dos pesos.

<a id="report-119-por-que-o-gate-não-é-aceitação-física"></a>

#### Por que o gate não é aceitação física

Nenhuma macro por grupo piorou no piloto, mas o gradiente teve **um arquivo
com p90 de canais maior**, **23/30 pares grupo×banda com erro espectral maior**,
18/30 com pior distância de amplitude RMS a1 e 20 aumentos descritivos de fase.
O controle teve 9/30 regressões de erro espectral.
Suportes de fase podem variar; não declarar uma guarda de fase certificada.

A aceitação por NRMSE* de canais pode trocar erro entre canais e frequências.
O ganho de média **não demonstra preservação do waveform em todas as bandas**.
Não promover, destilar ou iniciar treino a partir dessa média.

<a id="report-119-continuação-registrada"></a>

#### Continuação registrada

Somente o gradiente passou o gate exploratório ≥1% sem bytes adicionais.
Ampliar seu diagnóstico para 136 checks, sem alterar shortlist8, duas rodadas,
frações, proxy, aceitação ou pesos. O controle latente não recebe extensão
após seu gate reprovado; não comparar sua média de dez janelas com a nova coorte.

Os fontes exatos foram conferidos e arquivados em `SOURCE_SNAPSHOT/` antes da
extensão do leitor de população. O protocolo120 exige identidade do núcleo
algorítmico e publica as regressões, sem acesso a validação/teste/EDAS.

Artefatos: `results/experiments/direct_discrete_index_probe_20261115/`.

---

<a id="report-120"></a>

## Registro histórico 120 — Auditoria ampliada do mesmo encoder discreto congelado

Fonte original preservada no arquivo: `120-DIRECT_DISCRETE_EXPANDED_AUDIT_PROTOCOL.md`.

<a id="report-120-auditoria-ampliada-do-mesmo-encoder-discreto-congelado"></a>

**15/11/2026. Antes dos sinais desta ampliação.**

O probe118 passou o gate escalar no braço gradiente, mas apresentou regressões
espectrais. Ampliar somente para **caracterizar o compromisso e suas caudas**,
não para salvar uma promoção ou ajustar novos parâmetros.

- Mesmos 136 checks train-only / 43 arquivos / dez grupos / quatro famílias /
  12.799.000 células dos protocolos105/115.
- Mesma origem densa e calibração fixa dos 44 arquivos fit.
- Shortlist8, duas rodadas, frações1/5/10/20%, proxy e cap de frame idênticos.
- Verificar o hash estrutural das funções do algoritmo e da grade contra
  o snapshot118 antes de abrir novos sinais. Apenas a população/limite total
  de inferências muda; fonte, pesos e coder não mudam.
- Até272 backward de decoder e1.088 propostas hard; constantes usam literal
  comum, sem exclusão da cobertura. Zero updates neurais/codebook.
- Manter baseline, frames finais e replay separado para todas as janelas.
  Publicar pooling físico por canal/arquivo, caudas e bandas.

Não garantir melhora do pooling por arquivo apenas pela aceitação por janela:
variância e ponderações das duas agregações não são equivalentes.
Cap de bytes continua válido janela a janela; setup comum é cobrado em ambos.

Não ampliar o controle latente reprovado nem selecionar outra seed/grade.
Não aplicar uma guarda espectral nova pós-hoc para renomear o mesmo resultado.
Uma variante guardada seria outro protocolo/contraste, se houver justificativa.

Nenhum acesso a waveforms de validação/teste/EDAS, nenhuma promoção/P3
automática, nenhum treino de distilação implícito. A fonte madura permanece
fallback e referência. O resultado irá decidir se existe oportunidade física
útil ou apenas redução escalar com redistribuição indesejada de erro.

GPU0, somente inferência/derivadas no decoder congelado.
Destino: `results/experiments/direct_discrete_expanded_20261115/`.

<a id="report-120-resultado"></a>

##### Resultado

Cobertura136 concluída, pesos/algoritmo/registro congelados e dez baselines do
piloto reproduzidos. Ganho escalar0,994%, mas regressões físicas importantes.
Não promover nem estender; [resultado121](102-DIRECT_DISCRETE_INDEX_STUDY.md#report-121).

---

<a id="report-121"></a>

## Registro histórico 121 — Busca discreta ampliada: redução de NRMSE* com redistribuição de erro

Fonte original preservada no arquivo: `121-DIRECT_DISCRETE_EXPANDED_RESULTS.md`.

<a id="report-121-busca-discreta-ampliada-redução-de-nrmse-com-redistribuição-de-erro"></a>

**15/11/2026.** Resultado do protocolo120, algoritmo congelado do probe118.

<a id="report-121-cobertura-e-integridade"></a>

#### Cobertura e integridade

136 janelas, 43 arquivos, dez configurações, quatro famílias,
**12.799.000 células train-only**. As duas janelas literais foram mantidas.
Houve 1.072 propostas nas 134 janelas neurais, dentro do teto1.088; 297
melhorias de incumbente, não 297 exemplos independentes.

O hash estrutural das funções/grade bateu com o snapshot118. O registro de
probabilidades permaneceu idêntico e os dez frames de baseline do piloto
foram reproduzidos byte a byte. Pesos e codebook não receberam updates.
Receptor separado e cap do frame original foram verificados em cada janela.

<a id="report-121-resultado-escalar-sem-arredondamento-do-gate"></a>

#### Resultado escalar, sem arredondamento do gate

| Método | NRMSE* macro famílias ↓ | Stream bpc ↓ |
|---|---:|---:|
| Fonte congelada | 0,490336 | 0,679245 |
| Índices por gradiente discreto | 0,485462 | 0,676880 |

Redução de erro: **0,994025%**. Economia de bytes: **0,348208%**.
Não arredondar 0,994% para alegar ganho ≥1%. O gate118 passou legitimamente
na população original de dez janelas; isso não é reclassificado aqui.
A ampliação120 não autorizava treinamento ou outra extensão por uma média
favorável. A flag legada `expansion_eligible` não é um novo gate de promoção.
Mesmo um ganho acima de 1% não resolveria os vetos físicos abaixo.

As dez macros por configuração melhoraram; houve, porém, **um arquivo com
NRMSE* maior**, pois aceitação por janela não garante o mesmo resultado no
pooling por arquivo. Isso estava explicitado no protocolo120.

<a id="report-121-guardas-e-diagnóstico-da-troca"></a>

#### Guardas e diagnóstico da troca

Auditoria pareada:

- p90 de canais maior em **11/43 arquivos**;
- erro espectral maior em **25/30 pares configuração×banda**;
- pior distância de amplitude RMS a1 em **24/30**;
- 25 aumentos descritivos de fase, com suporte/cobertura publicados, sem
  alegar guarda de fase comum certificada.

Uma análise **pós-hoc descritiva dos registros já calculados**, sem sinais,
encontrou aumento de SSE físico bruto em oito das dez configurações. Exemplos:
doped ~+4,26%, multimode7 ~+9,10%, single-mode11 ~+15,27%.
Single-mode4/7 tiveram pequenas reduções de SSE.

NRMSE* e SSE bruto têm ponderações diferentes de amplitude/canal. Esses números
não trocam retrospectivamente a métrica principal, mas explicam por que uma
média normalizada melhor não certifica preservação de todo o waveform.
O ganho escalar está acompanhado de redistribuição indesejada de erro.
Não chamar o residual de ruído irreduzível nem inferir causalidade universal.

<a id="report-121-decisão"></a>

#### Decisão

**Não promover, não iniciar treino de distilação, P3 ou nova rodada de
shortlist/frações/gates/LR para ultrapassar o limiar.**
A fonte madura continua default. O probe mostrou uma oportunidade escalar
limitada no encoder, não um codec fisicamente melhor sob as guardas.

O controle latente reprovado ficou na população original de dez janelas;
não comparar sua média com a coorte136 como se fosse um controle ampliado.
E1 histórico continua arquivado: este teste não é uma família arquitetural nova
e não reclassifica seus resultados.

Qualquer futura busca com aceitação física conjunta seria outro algoritmo,
com protocolo e custos próprios; não pode ser aplicada pós-hoc para renomear
este resultado. Não há autorização implícita para tal sweep por terminar aqui.

Nenhum waveform val/test/EDAS foi aberto, nenhum update foi acrescentado aos
245.760 updates de treinamento consolidados até116. Derivadas em q para escolher
índices no compressor não são updates de pesos ou vetores contínuos transmitidos.

Artefatos: `results/experiments/direct_discrete_expanded_20261115/`, incluindo
`PHYSICAL_COMPARISON.json`, `ERROR_REDISTRIBUTION.json`, frames e diagnóstico
por proposta. O SSE bruto é descritivo, não uma tolerância operacional definida.

Verificação de engenharia: **468 testes passaram**. Fontes foram conferidos
contra os hashes do protocolo e arquivados, contagens completas e os 136 caps
de frame auditados sem reabrir waveforms. Evidência em `POSTFLIGHT_AUDIT.json`.
