# Roadmap de reconstrução generalista FEBUS

## 1. Finalidade e escopo

Desenvolver um modelo ou sistema de compressão que reconstrua fielmente os dados
FEBUS do Drive nas quatro famílias: **doped, multicore, multimode e single-mode**.
O sucesso deve combinar fidelidade, transferência entre configurações e custo
real de codificação; uma média favorável não pode esconder falhas sistemáticas
em uma família.

A solução pretendida é **neural**. Métodos clássicos podem contextualizar resultados
históricos, mas não são uma alternativa de produto nem um benchmark obrigatório
para decidir entre arquiteturas neurais.

Este documento define etapas, entregáveis, dependências e critérios de decisão.
Os relatórios registram experimentos, resultados e limitações. Datas, comandos,
logs, checkpoints específicos e estado de execução pertencem aos relatórios,
não ao roadmap.

### Princípios permanentes

- **Reconstrução não é denoising:** preservar também o ruído presente no alvo.
  Não filtrar, fazer clipping ou descartar componentes para facilitar a tarefa
  sem declarar explicitamente uma mudança de objetivo.
  A seleção declarada da região espacial útil é anterior a esse alvo: registrar
  canais excluídos, coordenadas, regra versionada e incerteza da máscara automática.
  Preservar os brutos e não contabilizar redução de cobertura como compressão neural.
- **Rastreabilidade física:** distinguir variável/unidade, sample rate, espaçamento,
  gauge length, canais e duração. Strain, strain rate e OOI `dphi` não são alvos
  intercambiáveis; forma em pixels semelhante não implica domínio semelhante.
- **Generalização por grupos:** separar aquisições/configurações, mantendo arquivos,
  núcleos simultâneos e janelas relacionadas na mesma partição.
- **Comparações controladas:** mudar uma hipótese por vez, documentar exposição aos
  dados e orçamento, preservar referências e registrar resultados negativos.
- **Transferência é uma opção, não uma restrição:** pesos OOI, treino do zero,
  especialistas e novas arquiteturas devem ser comparados conforme a evidência.
- **Integridade e reversibilidade:** preservar originais e contabilizar metadados,
  máscaras, parâmetros de normalização e mecanismos de escape no codec completo.

### Direcionamento da comparação neural

A seleção espacial deve ser preparada e auditada antes dos controles e treinos,
mantendo a mesma regra/cobertura entre braços. Em corpus com tiles separados,
declarar sua ponderação para não equiparar um resto de um canal a uma região de
512 canais. Avaliação do codec deve reconstruir os vetores discretos exatos a
partir dos índices, sem depender de resíduos numéricos do straight-through.

A comparação estrutural de referência é **A) arquitetura nativa com inicialização offshore,
B) mesma arquitetura com inicialização aleatória e C) arquitetura orientada a
FEBUS com inicialização aleatória**, no mesmo corpus FEBUS e orçamento de bits.
O offshore original, sem fine-tuning, é o baseline de origem; referências já
adaptadas são outros pontos de comparação, não substitutos desse baseline.

A comparação A/B/C e sua confirmação não demonstraram superioridade de C; a
liderança transferida não testa nem descarta uma rede substancialmente maior.
**A prioridade estrutural passa a ser expansão controlada da capacidade, até
cerca de 100M parâmetros, mantendo o gargalo compacto**, seguida de alocação
adaptativa ROI com cobertura global. A seção 6 reúne a base bibliográfica e
separa resultados publicados de escolhas experimentais para FEBUS.

Usar a CNN residual generalista como primeiro controle de escala; contexto
multiescala anisotrópico e atenção temporal/fatorizada são ablações separadas,
não mudanças obrigatórias junto com o tamanho. O artigo CNN–Transformer com OWR
inspira componentes, não classificação/expansão de classes. **PINN fica fora
deste direcionamento.**

Essa comparação da etapa 8 depende de protocolo, integridade e controles básicos
das etapas 1–3, **não de esgotar fine-tuning, especialistas ou todas as losses**.
Ampliação de dados FEBUS/compatíveis deve ser separável da mudança arquitetural.
Novas condições de fibra podem ser mudanças fortes de domínio/geometria, sem
exigir uma nova classe de saída para reconstruir o sinal.
Limiares físicos finais e auditoria externa são requisitos de promoção, não
impedimentos à triagem exploratória controlada antes de existir um candidato.

## 2. Trabalho realizado que fundamenta o plano

As referências abaixo identificam estudos e implementações já documentados.
Não significam que uma linha de pesquisa esteja esgotada ou que seus modelos
tenham atendido aos critérios de produção.

| Referência | Trabalho realizado | Implicação para o roadmap |
|---|---|---|
| [Relatório 1 — diagnóstico de fidelidade](1-VQVAE_DAS_FIDELITY_DIAGNOSIS.md) | Comparação OOI/FEBUS e testes de inferência, geometria, normalização e uso do codebook. | Tratar deslocamento de domínio e escala física como hipóteses, sem atribuir a falha apenas ao formato de entrada. |
| [Relatório 2 — fine-tuning inicial](2-FINETUNE_EVALUATION_REPORT.md) | Adaptação FEBUS, splits por configuração e avaliação estratificada. | Ganho agregado pode coexistir com regressão de famílias; seleção de checkpoint precisa acompanhar fidelidade. |
| [Relatório 3 — amostragem e reconstrução](3-SAMPLING_FULL_AND_RECONSTRUCTION_DIAGNOSIS.md) | Comparação uniforme/hierárquica e diagnóstico de amplitude, espectro e quantização. | Balanceamento não demonstrou ganho generalista consistente; retirar só a quantização não resolveu a reconstrução. |
| [Relatório 4 — objetivo espectral e seleção macro](4-SPECTRAL_OBJECTIVE_AND_MACRO_SELECTION_PILOT.md) | Piloto STFT, métricas macro/por configuração e inicialização como candidata. | Melhor magnitude espectral não garante fase ou forma de onda melhores; preservar seletores de fidelidade separados da perda VQ. |
| [Relatório 5 — capacidade individual](5-TRAIN_ONLY_CAPACITY_PROBE.md) | Memorização de janelas independentes com VQ e autoencoder contínuo. | Capacidade de ajustar exemplos vistos não demonstra transferência; contínuo não é codec equivalente em bitrate. |
| [Relatório 6 — capacidade compartilhada](6-SHARED_CAPACITY_AND_TRANSFER_PROBE.md) | Ajuste de um pool multifamília, controles e validação por configuração. | Investigar overfitting, preservação da representação e interferência; não promover memorizadores como especialistas. |
| [Relatório 7 — cobertura e integridade](7-CORPUS_COVERAGE_AND_NUMERIC_INTEGRITY_AUDIT.md) | Auditoria temporal/espacial, redundância, conteúdo e valores extremos. | Medir informação útil, não só bytes; tratar qualidade numérica e rastreabilidade antes de atribuir tudo à falta de dados. |
| [Relatório 8 — proteção reversível](8-REVERSIBLE_NUMERIC_PROTECTION_AND_MANUAL_PILOT.md) | Escape exato, roundtrip do corpus e integração de inferência. | Separar fidelidade da região modelada, preservação dos escapes e overhead; proteção numérica não resolve sozinha a qualidade do modelo. |
| [Relatório 9 — adaptação numérica e preservação](9-NUMERIC_PILOT_RESULTS_AND_LONG_PRESERVATION_EXPERIMENT.md) | Piloto pareado, auditoria de exposição e definição de comparação full/decoder-only. | Ganhos pequenos em validação de duas famílias não estabelecem ganho generalista; exigir cobertura explícita e testar preservação com corpus amplo. |
| [Relatório 10 — resultados da preservação](10-LONG_PRESERVATION_RESULTS.md) | Comparação longa, validação completa e verificação do congelamento dos pesos. | Ajuste completo superou decoder-only, mas o ganho foi pequeno e uma configuração regrediu; investigar representação além de estender updates. |
| [Relatório 11 — representação por canal](11-CHANNEL_REPRESENTATION_AUDIT_AND_PILOT.md) | Auditoria de normalização reversível, erro por canal e avaliação comum após restauração. | Escala por canal altera a distribuição e o custo do codec; comparar modelos adaptados no mesmo alvo físico, não scores com denominadores diferentes. |
| [Relatório 12 — mapas espaciais](12-SPATIAL_CHANNEL_MAPS_AND_GEOMETRY_INTEGRITY.md) | Recuperação de índices físicos, verificação raw/NPZ e rastreabilidade nas novas preparações. | Lacunas no HDF5 bruto não implicam lacunas no crop final; validar a geometria realmente apresentada ao modelo antes de atribuir falhas a ela. |
| [Relatório 13 — normalização treinada e validação ampliada](13-CHANNEL_PILOT_RESULTS_AND_EXPANDED_VALIDATION.md) | Piloto pareado no mesmo alvo restaurado e protocolo de análise robusta por canal/taxa agregada. | Ganho em canais fracos ou em uma família não basta para promover uma representação; comparar fidelidade física e custos sem confundir denominadores. |
| [Relatório 14 — validação completa e controle de taxa](14-FULL_NORMALIZATION_VALIDATION_AND_RATE_CONTROL.md) | Avaliação completa dos candidatos, custos pooled e definição de controle clássico codificável. | Fidelidade global, erro por canal e bitrate podem ter trade-offs; usar curvas rate-distortion e controles explícitos antes de prolongar adaptações. |
| [Relatório 15 — controle DCT rate-distortion](15-SPARSE_DCT_RATE_DISTORTION_RESULTS.md) | Codec clássico com byte stream real, avaliação por família e curva de orçamentos. | Um controle simples pode superar a rede em certos grupos; mais bits melhoram erro global sem garantir fidelidade por canal ou estabelecer um lower bound. |
| [Relatório 16 — densidade temporal dos latentes](16-TEMPORAL_LATENT_DENSITY_PILOT.md) | Implementação de ablação de stride temporal, contagem real de índices e testes de memória/compatibilidade. | Alterações de geometria e taxa exigem decoder compatível, contabilização de índices e comparação no mesmo alvo; pesos de mesma forma não garantem a mesma representação. |
| [Relatório 17 — confirmação da densidade temporal](17-TEMPORAL_DENSITY_FULL_RESULTS_AND_LONG_CONTINUATION.md) | Avaliação completa dos candidatos 2x/4x e definição de continuação controlada por curva de aprendizagem. | Ganhos nas configurações reservadas justificam testar saturação, mas devem ser confrontados com taxa, piores canais e robustez entre seeds/aquisições. |
| [Relatório 18 — continuação longa e critérios de decisão](18-TEMPORAL_DENSITY_LONG_RESULTS_AND_DECISION_GATES.md) | Resultados longos 2x/4x, regressões por configuração e diagnóstico concorrente de amplitude/fase. | Melhor RMSE com queda de fase exige auditoria multimetricamente estratificada e controle de taxa pareado antes de mais updates ou promoção. |
| [Relatório 19 — auditoria temporal e DCT pareado](19-TEMPORAL_AUDIT_AND_MATCHED_DCT_RESULTS.md) | Avaliação robusta de canais, taxas pooled e controle clássico com teto exato por candidato/janela. | Vantagem neural macro pode coexistir com cauda pior e vantagem DCT em single-mode; separar mudança de ponderação espectral de falha física antes de alterar objetivos ou especializar. |
| [Relatório 20 — diagnóstico de fase com pesos do alvo](20-TARGET_WEIGHTED_PHASE_DIAGNOSIS.md) | Comparação pareada por bandas com cobertura de fase e pesos independentes da amplitude reconstruída. | Queda de alinhamento com pesos variáveis não implica regressão geral; direcionar ablações a bandas e caudas com falhas demonstradas, preservando métricas concorrentes. |
| [Relatório 21 — arquitetura e inicialização controladas](21-ARCHITECTURE_AND_SCRATCH_TRAINING_PROTOCOL.md) | Implementação/testes de factory versionada e definição de triagem transferido versus scratch nativo versus scratch temporal, com parâmetros/taxa pareados. | Investigar mudança estrutural sem trocar corpus simultaneamente; diagnosticar otimização/codebook e dar oportunidade de convergência antes de descartar treino do zero. |
| [Relatório 22 — resultados de arquitetura/inicialização](22-ARCHITECTURE_INITIALIZATION_RESULTS.md) | Triagem de três braços e avaliação completa dos selecionados, com curvas, utilização de códigos e trade-offs por canal/configuração. | Vantagem transferida macro coexiste com ganho da variante temporal sobre scratch nativo; scratch em melhora e codebook restrito exigem teste de convergência/quantização antes de descarte estrutural. |
| [Relatório 23 — orçamento adicional scratch](23-SCRATCH_BUDGET_CONTINUATION_PROTOCOL.md) | Protocolo/runner de continuação com optimizer/RNG e históricos preservados, referência congelada e quantizador inalterado. | Isolar ganho por exposição antes de intervir em códigos inativos; uso de codebook e compute adicional devem ser confrontados com fidelidade por grupo/canal. |
| [Relatório 24 — continuação e diagnóstico de quantização](24-SCRATCH_CONTINUATION_AND_QUANTIZATION_DIAGNOSIS.md) | Resultados de orçamento adicional e contraste quantizado/contínuo no mesmo encoder/decoder. | Ganho concentrado em configurações e maior gap VQ motivam ablação isolada; contínuo não é codec equivalente nem prova causal de limitação. |
| [Relatório 25 — reativação train-only pareada](25-PAIRED_CODEBOOK_REFRESH_PROTOCOL.md) | Implementação/testes e protocolo de reativação limitada, com controle, momentos seletivos e estado retomável. | Intervenções na quantização exigem validação após modificação, dados de treino somente e ganho físico, não apenas aumento de códigos utilizados. |
| [Relatório 26 — resultado da reativação](26-CODEBOOK_REFRESH_RESULTS.md) | Avaliação completa do controle/refresh, uso de códigos e regressões de famílias/canais. | Expandir códigos não resolveu fidelidade generalista; ganhos incompatíveis justificam testar especialização sem promover média favorável. |
| [Relatório 27 — especialistas e dose compartilhada](27-DOMAIN_SPECIALISTS_AND_BALANCED_CONTROL_PROTOCOL.md) | Protocolo/implementação de experts por família, fallback, roteamento não oracle e controle com dose familiar pareada. | Distinguir benefício de especialização de exposição adicional; medir modelos, rota, piores grupos e confirmação fora do desenvolvimento. |
| [Relatório 28 — resultado da especialização](28-DOMAIN_SPECIALIZATION_RESULTS.md) | Avaliação roteada completa e confronto com generalistas/dose balanceada. | Ganho pequeno com regressões e registry maior não justifica gating aprendido automaticamente; revisitar representação/taxa antes de escalar experts. |
| [Relatório 29 — alocação espaço-temporal](29-SPATIOTEMPORAL_RATE_ALLOCATION_PROTOCOL.md) | Implementação/testes de grids espacial/temporal, padding compensatório e comparação de taxa igual/quase igual. | Separar quantidade de índices de sua distribuição entre eixos; preservar alvo, contabilizar padding e declarar mudanças de receptive field/contexto. |
| [Relatório 30 — resultado da alocação](30-RATE_ALLOCATION_RESULTS.md) | Avaliação completa de grades, taxas reais de índices e métricas robustas por canal/configuração. | Ganhos espacial/equilibrado frente ao anchor justificam confirmação; nenhuma grade domina todos os grupos frente ao controle temporal. |
| [Relatório 31 — confirmação entre folds](31-RATE_ALLOCATION_FOLD_CONFIRMATION_PROTOCOL.md) | Protocolo/runner para novas configurações e seed, com fonte específica do fold e guards de lineage. | Não reutilizar pesos treinados em grupos reservados do novo split; comparar contrastes dentro de fold e separar replicação exploratória de auditoria externa. |
| [Relatório 32 — confirmação e faixa dinâmica residual](32-FOLD_CONFIRMATION_AND_RESIDUAL_DYNAMIC_RANGE.md) | Ranking dependente do fold, localização de caudas gigantes e diagnóstico de roundtrip sem modelo. | Escape absoluto pode deixar dominantes menores dentro da região modelada; reauditar representação e máscara comum/custos antes de atribuir tudo à arquitetura ou treinar mais. |
| [Relatório 33 — proteção adaptativa e identidade](33-DYNAMIC_RANGE_AUDIT_AND_FROZEN_COMPARISON.md) | Auditoria completa train/val de roundtrip, escapes e taxa; definição de comparação congelada com referência comum. | Melhora de preprocessing não implica ganho aprendido; medir cobertura/tails/custo e testar pesos congelados antes de repetir treino em dados cujo input não mudou. |
| [Relatório 34 — comparação congelada de faixa dinâmica](34-FROZEN_DYNAMIC_RANGE_RESULTS.md) | Avaliação completa por região, efeito localizado em 40 janelas e invariância de 5.880 janelas. | Mudança de denominadores exige referência comum; corrigir input não resolve fidelidade geral nem justifica treinar novamente quando o treino é inalterado. |
| [Relatório 35 — cobertura da proteção](35-DYNAMIC_RANGE_COVERAGE_CONCLUSION.md) | Auditorias train/val de dois folds cobrindo as 23.880 coordenadas canônicas e verificação de invariância dos treinos. | Proteção localizada sem input de treino alterado não exige nova adaptação; dados fora dos crops e novos grupos continuam requerendo auditoria. |
| [Relatório 36 — curva neural de taxas maiores](36-NEURAL_RATE_DISTORTION_SWEEP_PROTOCOL.md) | Implementação/testes de densidade temporal crescente com grade espacial fixa e taxas explicitamente diferentes. | Investigar limite de taxa com curvas/controles, cobrando índices/side/compute; melhor erro a bitrate maior não prova vantagem à mesma taxa nem aceitação operacional. |
| [Relatório 37 — resultado da curva neural](37-NEURAL_RATE_DISTORTION_RESULTS.md) | Ganhos de fidelidade nas seis configurações com taxa maior e controle DCT pareado atualizado. | Orçamento de índices é um gargalo relevante; multimode/canais ainda limitam aceitação e a vantagem requer confirmação fora desse desenvolvimento. |
| [Relatório 38 — continuação de maior taxa](38-HIGH_RATE_CONTINUATION_PROTOCOL.md) | Protocolo de extensão limitada da curva ainda em melhora, com optimizer/RNG e referência anterior preservados. | Medir saturação à mesma taxa sem repetir braços estagnados; compute adicional e candidatos anteriores fazem parte da decisão. |
| [Relatório 39 — resultado da continuação 4x](39-HIGH_RATE_CONTINUATION_RESULTS.md) | Melhora adicional à mesma taxa nas seis configurações, com limites por família/canal. | Ganho marginal menor favorece confirmação entre configurações; validar source contra todos os grupos reservados do novo split. |
| [Relatório 40 — confirmação de taxa com referência robusta](40-HIGH_RATE_FOLD0_ROBUST_PROTOCOL.md) | Integração experimental de loader/máscara/score robustos e protocolo de confirmação 1x/2x/4x em outro fold. | Comparar taxas em referência comum, preservar weights ao reencodar source e impedir rankings com denominadores incompatíveis. |
| [Relatório 41 — resultado da curva robusta no fold 0](41-ROBUST_RATE_FOLD0_RESULTS.md) | Ganhos monotônicos nas seis configurações em RMSE/correlação/p90 por canal, com custo pooled de índices+side. | Distinguir pior configuração de pior família e preservar diagnósticos de amplitude, banda alta e canais fracos. |
| [Relatório 42 — continuação robusta 2x/4x](42-ROBUST_RATE_CONTINUATION_PROTOCOL.md) | Protocolo limitado de orçamento adicional com optimizer/RNG e taxas preservados. | Comparar ganhos dentro de cada taxa antes de aumentar complexidade, explicitando compute desigual das referências congeladas. |
| [Relatório 43 — resultados e auditoria pareada](43-ROBUST_CONTINUATION_RESULTS_AND_PAIRED_AUDIT.md) | Ganhos macro modestos com regressão recorrente 4x em single-mode Configuration 2; auditoria por janela/arquivo e limite do seletor. | Separar representatividade da amostra e trade-offs do seletor; bootstrap de arquivos reutilizados não substitui aquisições independentes. |
| [Relatório 44 — estabilidade congelada](44-FROZEN_RATE_STABILITY_PROTOCOL.md) | Protocolo de comparação de endpoints, checkpoint retido por loss e midpoint fixo sem updates. | Testar suavização de trajetória sem aumentar taxa/modelo e sem presumir benefício da média de embeddings VQ; evitar busca adaptativa irrestrita na validação. |
| [Relatório 45 — resultado de estabilidade](45-FROZEN_RATE_STABILITY_RESULTS.md) | Midpoint 2x melhora RMSE/correlação/p90 nos seis grupos frente ao source, mas 4x mantém regressão localizada. | Preservar trade-offs entre critérios, não generalizar averaging entre taxas e exigir critérios de seleção/aceitação antes de mais compute. |
| [Relatório 46 — auditoria de amostragem e seleção](46-CHECKPOINT_SELECTION_AUDIT.md) | Comparação retrospectiva de amostragem por arquivo e seletor macro versus guarda por configuração em candidatos congelados. | Amostra maior não corrige um objetivo que admite perdas; triagem com guarda ainda exige confirmação completa pareada e não substitui avaliação independente. |
| [Relatório 47 — ganho global ajustado no treino](47-TRAIN_ONLY_GLOBAL_GAIN_PROTOCOL.md) | Protocolo isolado de calibração de um escalar por modelo congelado com ajuste train-only, índices preservados e custo global explícito. | Separar viés de escala de perda de forma de onda; subestimação de RMS não implica benefício de amplificação, nem autoriza ganhos por configuração. |
| [Relatório 48 — resultado do ganho train-only](48-TRAIN_ONLY_GAIN_RESULTS.md) | Ganhos ajustados quase unitários, regressões 2x e efeito macro diminuto/heterogêneo 4x. | Não confundir casamento de RMS com fidelidade; preservar controles negativos e evitar ampliar calibração na validação sem hipótese independente. |
| [Relatório 49 — controle de quantização congelada](49-FROZEN_QUANTIZATION_PROBE_PROTOCOL.md) | Protocolo VQ versus encoder contínuo no mesmo decoder, com custo integral dos latentes float32 e erro de quantização medido. | Separar diagnóstico de codec rate-matched; desvio de distribuição impede interpretar bypass como piso irreducível ou prova de adequação do codebook. |
| [Relatório 50 — resultado do bypass congelado](50-FROZEN_QUANTIZATION_RESULTS.md) | Melhora de RMSE em cinco configurações, mas regressão em todas as janelas multimode, inclusive sem escapes; grande expansão de payload. | Não atribuir regressão geral à proteção numérica nem equiparar bypass a codec; separar escala e estrutura dos latentes antes de inferir gargalos. |
| [Relatório 51 — controle de energia latente](51-NORM_MATCHED_LATENT_CONTROL_PROTOCOL.md) | Protocolo determinístico que preserva direção contínua e energia quantizada, sem ajuste de coeficiente ou novo treino. | Energia igual não significa distribuição igual; cobrar float32 integral e evitar extrapolar controles de decoder para melhorias rate-matched. |
| [Relatório 52 — resultado da energia igualada](52-NORM_MATCHED_LATENT_RESULTS.md) | Redução parcial do erro do bypass, com regressão persistente nas 600 janelas multimode, inclusive sem escapes. | Encerrar busca de escalas sobre o mesmo corpus; não equiparar norma à distribuição ou concluir adequação do quantizador pelo bypass. |
| [Relatório 53 — piloto full/decoder com guarda](53-GUARDED_DECODER_PILOT_PROTOCOL.md) | Protocolo pareado de adaptação a baixo LR, validação completa por época, snapshots e referência fixa por configuração/canal. | Isolar escopo de adaptação mantendo taxa/objetivo, preservar controle macro e distinguir admissão exploratória de aceitação física ou generalização independente. |
| [Relatório 54 — resultado full/decoder com guarda](54-GUARDED_DECODER_PILOT_RESULTS.md) | Ganhos pequenos de RMSE nos seis grupos, mas nenhum dos 16 snapshots satisfaz todos os critérios; caudas/correlação explicam vetos. | Não atribuir toda regressão à amostra curta nem tratar epsilon como tolerância física; preservar baseline sem estender treino automaticamente. |
| [Relatório 55 — objetivo por canal limitado](55-BOUNDED_CHANNEL_OBJECTIVE_PROTOCOL.md) | Protocolo de termo train-only com peso/cap fixos, representação intacta e reutilização de controle full completo. | Isolar objetivo sem repetir controle, evitar explosão em canais fracos e não relaxar guarda ou buscar pesos adaptativamente na validação. |
| [Relatório 56 — resultado do objetivo por canal](56-BOUNDED_CHANNEL_OBJECTIVE_RESULTS.md) | Admissão nas fases 4/8, com menor p90 em todos os grupos frente ao controle, mas RMSE ligeiramente pior; margens pequenas e perdas por janela. | Distinguir preservação contra baseline de dominância sobre controle e de aceitação física; confirmar antes de alterar defaults. |
| [Relatório 57 — confirmação entre configurações](57-CHANNEL_OBJECTIVE_FOLD_CONFIRMATION_PROTOCOL.md) | Protocolo pareado no fold 1 com seed adicional, pesos exclusivos do fold e fórmula/guarda fixadas. | Comparar contrastes dentro de fold, preservar denominadores e proveniência, e não confundir desenvolvimento reutilizado com aquisição externa independente. |
| [Relatório 58 — confirmação do objetivo por canal](58-CHANNEL_OBJECTIVE_FOLD_CONFIRMATION_RESULTS.md) | Redução de p90 contra controle nos seis grupos, mas nenhum snapshot preserva todos os critérios no fold 1. | Não promover resultado positivo de um fold a política generalista; registrar confirmação negativa sem retunar pesos ou tolerâncias. |
| [Relatório 59 — registro de referências 4x](59-GENERALIST_RESEARCH_REFERENCE_REGISTRY.md) | Checkpoints disponíveis/hashes verificados, guarda retrospectiva e fronteira por componentes em coortes compatíveis. | Distinguir referência de pesquisa de modelo final, computações históricas de comparações causais e redes por fold de um único codec validado externamente. |
| [Checkpoint de progresso 1](PROGRESS_CHECKPOINT_1.md) | Síntese dos relatórios 1–59, dos ganhos de reconstrução e do custo em densidade latente/bitrate; detalhes da mudança de direcionamento neural. | Usar o histórico como controle para uma comparação neural estrutural, sem confundir aumento de informação transmitida com eficiência de representação; testar CNN/atenção sem impor OWR. |
| [Relatório 62 — corpus corrigido e comparação A/B/C](62-SPATIAL_CORPUS_AND_CONTROLLED_NEURAL_COMPARISON.md) | Auditoria integral do corpus espacial, arquitetura fatorizada multiescala, replay discreto exato, controle de memória e protocolo em duas taxas. | Separar cobertura, inicialização, capacidade, taxa e compute; usar orçamento limitado com extensão por convergência e avaliação por configuração/canal, sem promoção por média isolada. |
| [Relatório 63 — resultados e confirmação](63-SPATIAL_ABC_RESULTS_AND_FOLD_CONFIRMATION.md) | Transferência liderou as duas taxas na coorte inicial; guardas não autorizaram promoção e scratch utilizou menos códigos. | Confirmar em outra rotação/configuração antes de acrescentar atenção ou mudar loss/quantizador; manter referência histórica compatível com os novos papéis de treino/validação. |
| [Relatório 64 — confirmação concluída](64-CROSS_FOLD_CONFIRMATION_RESULTS.md) | Transferência voltou a liderar, mas ganhos macro coexistem com regressões por arquivo/configuração. | Priorizar validação física e tolerâncias de aplicação; não abrir nova busca arquitetural nem promover por média isolada. |

## 3. Etapas e entregáveis

A sequência organiza as dependências de pesquisa. As etapas de representação,
adaptação, cobertura de dados e arquitetura podem ser revisitadas conforme o
gargalo identificado; não é necessário esgotar todas as variantes para explorar
uma hipótese bem sustentada.

### Etapa 1 — Requisitos de fidelidade e protocolo de avaliação

**Objetivo:** definir o que significa reconstrução suficientemente boa e como
comparar candidatos sem favorecer famílias, configurações ou escalas dominantes.

**Entregáveis a completar/validar:**
- Limiares de aceitação por família e pior configuração, definidos pelo uso pretendido.
- Protocolo com splits, seletores, métricas, orçamento, seeds e auditoria externa.
- Critérios conjuntos de fidelidade, bitrate, latência, memória e cobertura.

**Critério de passagem:** aceitação verificável em todas as famílias, e não apenas
melhoria relativa sobre um baseline fraco. Instrumentação dos relatórios 2–4 e 9
serve de base, não substitui requisitos de aplicação.

### Etapa 2 — Corpus íntegro e pipeline rastreável

**Objetivo:** garantir que o modelo receba uma representação correta do sinal e
que a decodificação possa restaurar geometria, escala e valores preservados.

**Entregáveis a completar/validar:**
- Inventário de aquisições, grupos independentes, duração útil e cobertura espacial.
- Mapa dos índices físicos dos canais após filtragem/tiling, máscaras e metadados.
- Store canônico com referências de splits, evitando duplicar amostras por fold.
- Normalização/escape versionados, compatíveis entre treino, inferência e avaliação.
- Auditoria de faixa dinâmica residual após escape: preservação de variação e erro
  encode/decode identidade em canais ordinários, não só bit-exatidão dos extremos.
  Políticas alternativas exigem máscara de comparação comum e cobrança de side stream.
- Testes de roundtrip, extremos, regiões constantes, dados não finitos e arquivos inválidos.

**Critério de passagem:** nenhuma perda silenciosa de informação ou vazamento entre
partições; custos de representação mensuráveis. Auditorias e proteção dos relatórios
7–8 são referências para ampliar a cobertura de testes e integrar o codec completo.

### Etapa 3 — Baselines e isolamento dos gargalos

**Objetivo:** distinguir bugs, capacidade de ajuste, otimização, quantização e
transferência para configurações reservadas.

**Entregáveis a completar/validar:**
- Baselines reproduzíveis: checkpoint transferido, adaptação uniforme e referências
  simples como reconstrução zero no espaço normalizado.
- Distinguir o offshore original sem adaptação, a referência adaptada e o modelo
  aleatório treinado em FEBUS. Comparar taxas com a mesma precisão de entrada,
  shapes, padding e contabilização de side; redução de stride não é fine-tuning
  a bitrate constante.
- Curvas de treino/validação e testes de capacidade individual/compartilhada.
- Diagnóstico por família/configuração de amplitude, fase, espectro e uso do codebook.
- Diagnósticos de fase com pesos fixados pelo alvo e cobertura explicitada, ao lado
  de alinhamento ponderado por magnitudes, evitando confundir recuperação de energia
  com mudança de fase ou retirar silenciosamente componentes difíceis.
- Evidência de convergência e ablações que separem encoder, quantizador e decoder.

**Critério de passagem:** uma hipótese testável para a limitação, apoiada em resultados
por grupo. Os relatórios 1–6 fornecem controles; nenhum teste de memorização serve
como demonstração de generalização ou de bitrate competitivo.

### Etapa 4 — Representação temporal, espacial e dinâmica

**Objetivo:** adaptar a representação à física FEBUS sem destruir bandas ou regiões
difíceis do alvo para aproximá-lo artificialmente do domínio OOI.

**Entregáveis a desenvolver/validar:**
- Comparação de duração física das janelas, contexto multiescala e tiling espacial.
- Normalização global versus alternativas por canal/região, com parâmetros reversíveis.
- Tratamento de espaçamentos, gauge lengths e lacunas de canais, inclusive eventual
  condicionamento por geometria.
- Comparações sobre o mesmo alvo físico, contabilizando padding, bordas e overhead.
- Distribuição do orçamento de latentes entre canais e tempo, comparando grades
  de mesma taxa real/nominal quando possível. Padding compensatório deve ser
  versionado e restaurável, sem retirar canais/bandas do alvo.

**Critério de passagem:** ganho de fidelidade sem redução oculta do conteúdo avaliado.
Resampling exige antialiasing e uma especificação explícita de quais bandas devem
ser preservadas; aumentar overlap não equivale a aumentar aquisições independentes.

### Etapa 5 — Adaptação generalista e preservação da representação

**Objetivo:** melhorar transferência sem sacrificar famílias ou esquecer capacidades
úteis da inicialização.

**Entregáveis a desenvolver/validar:**
- Comparação de ajuste completo, decoder-only, congelamento parcial e replay.
- Ablações de LR, regularização, seleção/parada macro e objetivos de reconstrução.
- Amostragem controlada, exposição por grupo e testes de interferência entre famílias.
- Repetição dos candidatos promissores entre folds e seeds, com validação completa.
- Inicializações específicas de cada split, com lineage verificada: um checkpoint
  treinado em outro fold pode ter visto os grupos que agora são reservados.
  Mudança de seed de adaptação não implica repetição independente do pré-treino.

**Critério de passagem:** ganhos robustos na validação das quatro famílias, respeitando
as tolerâncias de regressão da etapa 1. Avaliar a inicialização como candidata;
mais updates não obrigam a escolher um checkpoint posterior. Em mudanças de LR,
distinguir branch com optimizer novo de resume, que restaura seu estado salvo.

Os relatórios 3–4, 6 e 9 orientam as ablações, sem fixar uma loss, sampler ou política
de congelamento como solução universal.

### Etapa 6 — Diversidade dos dados e ampliação condicional do corpus

**Objetivo:** testar se limitações de cobertura justificam novas aquisições FEBUS-like.

**Entregáveis a desenvolver/validar:**
- Curvas com subconjuntos crescentes de grupos independentes do treino, validação fixa
  e orçamento/convergência controlados, evitando apenas sortear mais janelas correlacionadas.
- Identificação de lacunas por família, geometria, variável, espectro e condição instrumental.
- Inventário de datasets externos com licença, origem, unidades e compatibilidade física.
- Piloto FEBUS-only versus externos+FEBUS, incluindo pré-treino/adaptação e mistura,
  com custos e transferência negativa medidos por família.
- Curvas de escala de dados por aquisições/configurações inteiras, inicialmente
  25%, 50% e 100% dos grupos disponíveis no treino, com estratos preservados quando
  viável; não partir arquivos correlacionados só para atingir essas frações.
  Comparar pelo menos o controle e a rede maior, validação fixa e exposição
  registrada; controlar updates e analisar também passes/convergência.
- Medir duração única, canais/metros úteis, eventos/regimes e diversidade espectral,
  além de janelas. Não converter contagens de imagens ou horas de áudio da literatura
  em uma regra de amostras por parâmetro para DAS. Novas aquisições independentes
  são prioridade de generalização, sem bloquear um piloto de capacidade no corpus atual.

**Critério de passagem:** evidência de diversidade adicional útil ou justificativa
explícita para preencher uma lacuna. GB brutos, inclusive de OOI, não diagnosticam
escassez: contabilizar duração, canais, redundância e processamento efetivo.
Escalar downloads somente após validar acesso, qualidade, licença e armazenamento.

### Etapa 7 — Especialização, MoS e fallback generalista

**Objetivo:** investigar especialização se os trade-offs entre famílias persistirem.

Não é pré-requisito da comparação estrutural. Famílias/configurações não vistas
não exigem, por si só, OWR ou um decoder novo: mudança de domínio é distinta de
novidade de classe definida por uma tarefa de identificação. OWR/incremental só
entra como hipótese se houver requisito explícito de classes/catalogação dinâmica.

**Entregáveis a desenvolver/validar:**
- Comparação do generalista com especialistas por família e/ou geometria.
- Roteamento inicial por metadados; gating aprendido apenas com dados de desenvolvimento.
- Fallback testado em famílias/configurações desconhecidas e metadados ausentes.
- Comparar primeiro com decoder compartilhado; múltiplos heads/adapters só
  avançam com benefício demonstrado e custos de rota/modelos versionados. Congelar
  redes antigas preserva suas funções condicionais, não garante representação
  de novos domínios ou retenção do sistema após mudança de rota/preprocessamento.
- Avaliação do sistema completo: fidelidade, cobertura, modelos, latência e overhead.
- Controle compartilhado com dose por família/contagem de updates comparável ao
  conjunto de experts, distinguindo efeito de exposição, ordem e otimização de
  vantagem estrutural. Parear updates não equivale a parear wall-clock/armazenamento.
- Roteamento por metadados fixado antes da avaliação, registry e fallback explícitos;
  teste funcional de rota desconhecida não demonstra fidelidade fora do domínio.

**Critério de passagem:** o sistema roteado cobre todas as famílias e melhora o
trade-off custo/fidelidade. Não usar seleção oracle pelo erro contra o original
no teste, nem tratar pesos de memorização dos relatórios 5–6 como especialistas
generalizáveis. Média de reconstruções pode suavizar o sinal e exige validação.

### Etapa 8 — Arquitetura, quantização e treino do zero

**Objetivo:** testar limites de capacidade/taxa quando dados, representação e
otimização estiverem adequadamente controlados.

**Entregáveis a desenvolver/validar:**
- **Escala de capacidade a taxa fixa:** controle ~28M e alvo ~100M (faixa de projeto
  80–120M); ponto intermediário ~50M se necessário para diagnosticar tendência/custo.
  Esses tamanhos são hipóteses de engenharia, não ótimos demonstrados para DAS.
  Começar no ponto compacto já controlado, preservando grade por shape, K=2.048,
  dimensão 128, 11 bits por índice, normalização, filtro, objetivo e cobertura.
  Mais largura interna não pode aumentar silenciosamente o número de tokens.
- Escalar primeiro profundidade/largura de uma mesma família residual, investindo
  em processamento em resolução reduzida; avaliar uma expansão do decoder ou
  blocos de contexto como ablação posterior. Atenção, se testada, deve operar em
  tokens reduzidos/janelas/eixos fatorizados, não em atenção global de todos os
  512×1.000 pontos. Contar parâmetros por encoder/decoder/quantizador/modelo de
  probabilidade; aumentar o preditor entrópico não equivale a aumentar o reconstrutor.
- Separar capacidade de inicialização: controles pequeno/grande do zero para o
  contraste causal, mantendo A transferido como referência prática. Uma expansão
  inicializada dos pesos offshore pode ser outro contraste apenas após validar
  preservação funcional/compatibilidade; não comparar só grande scratch versus
  pequeno transferido e atribuir a diferença ao tamanho.
- Verificar gradiente, AdamW, replay apenas de índices, shapes máximos, memória e
  throughput antes de treino longo. Em GPU de 8 GiB, 100M não tem viabilidade
  garantida: checkpointing, precisão validada e acumulação são recursos possíveis,
  mas não justificam reduzir contexto/alvo ou mudar a taxa sem um controle próprio.
  Parear exposição inicial e dar extensões limitadas por convergência predefinida;
  reportar também FLOPs/tempo, pois updates iguais não igualam compute.
- Inicialização aleatória versus transferência, com orçamento suficiente e comparável.
- Comparação controlada de referência transferida, mesma arquitetura do zero e
  variante arquitetural do zero no corpus fixo. Parear taxa, objetivo, exposição
  adicional e capacidade quando possível; declarar compute histórico não pareado.
- Para a variante FEBUS, testar processamento espacial/temporal anisotrópico e
  contexto multiescala, respeitando contiguidade, organização de núcleos, posições
  e shapes variáveis. Não importar um decoder denso de ciclo fixo como se canal–tempo
  fosse uma sequência elétrica periódica de dimensão equivalente.
- Especificar diferenças concretas de C frente à CNN residual nativa e à variante
  temporal dilatada já testada: blocos, campo receptivo e redução/reconstrução por
  eixo. “Deep CNN” ou mudança de nome não são evidência de arquitetura nova.
- Atenção CNN–Transformer é ablação opcional após redução de tokens ou fatorização
  por eixo. Comparar à CNN sem atenção, com taxa/capacidade/compute explícitos;
  bons resultados de identificação não demonstram ganho de reconstrução.
- Preservar K/dimensão do quantizador na primeira comparação sempre que viável e
  usar pontos de taxa comuns, incluindo regime compacto. O decoder não pode
  receber skips/features contínuos do encoder sem codificá-los e cobrar seu payload.
- Inicialização do codebook somente com treino e diagnóstico de perplexidade,
  utilização e perdas separadas. Colapso de otimização não prova falta de capacidade.
- Ablações de reativação/atualização de códigos com controle de mesma arquitetura,
  taxa, source e optimizer; intervenções usam apenas treino, preservam estado
  retomável e são avaliadas no codebook exato salvo. Uso de códigos não é aceitação.
- Triagem com orçamento mínimo sem parada precoce seguida de análise de convergência:
  candidatos scratch ainda em melhora não são descartados por um piloto curto.
- Ablações de strides, capacidade, codebook, quantização residual/multiescala e
  condicionamento físico, sem alterar todos os componentes simultaneamente.
- Curvas rate-distortion que incluam custos reais de codificação e decodificação.
- **Codificação entrópica dos índices:** estimar redundância marginal e condicional
  com dados de treino; comparar empacotamento fixo e entropy coding em índices
  congelados antes de alterar o quantizador. Exigir bitstream decodificável e
  índices idênticos. Contexto usa somente símbolos/metadados disponíveis ao decoder;
  contabilizar tabelas, modelos, headers e latência. Hiperprior é etapa posterior
  com side cobrado, não atalho sem custo.
- **Quantização, somente em ablação própria:** manter VQ atual no estudo de escala;
  depois comparar K=1.024/2.048/4.096 (10/11/12 bits por índice) com curvas de taxa,
  não como taxas iguais. Dimensão do embedding não é quantidade de bits transmitidos.
  FSQ é candidato para estabilidade, não solução comprovada para FEBUS; RVQ é
  candidato a refinamento, cobrando cada índice de cada estágio. Não repetir
  reativação de códigos apenas porque a utilização é baixa.
- Controles clássicos históricos podem contextualizar as curvas, sem condicionar
  a investigação neural a um novo benchmark clássico ou substituí-la por codec clássico.
- Fronteira não dominada de checkpoints em erro, amplitude, fase, piores canais
  e taxa; seleção por RMSE isolado não autoriza ignorar as outras tolerâncias.

**Critério de passagem:** ganho na pior família e nas métricas de aceitação à mesma
taxa, ou melhoria demonstrada da curva rate-distortion. A vantagem de um latente
contínuo não demonstra vantagem de compressão sem um codec/taxa equivalentes.
Dados e arquitetura podem ser investigados conjuntamente, mas com ablações separáveis.

### Etapa 8B — ROI espaço-temporal e refinamento neural com orçamento limitado

**Objetivo:** melhorar a distribuição dos bits sem abandonar a reconstrução de
toda a região útil. ROI significa priorizar sinal válido; não substitui nem
altera a máscara heurística de canais inadequados.

**Entregáveis a desenvolver/validar:**
- Camada neural básica global e refinamento residual quantizado em tiles
  selecionados. Alternativamente, mapas de qualidade condicionam o codec;
  essa alternativa deve ser comparada separadamente, sem assumir equivalência.
- Primeira política de ROI baseada em dificuldade de reconstrução medida **no
  encoder**, que dispõe do original e da reconstrução básica; combinar erro local
  relativo limitado e absoluto para não favorecer apenas regiões de alta energia
  nem explodir em canais fracos. Nenhum rótulo/evento ou oracle de teste é necessário.
  Fixar regra, limites e orçamento no desenvolvimento antes de avaliar.
- Distinguir essa seleção implementável da escolha retrospectiva do melhor modelo
  por erro no teste. O decoder recebe somente stream básico, mapa/endereços,
  refinamentos e metadados codificados; nenhum acesso ao original ou skip gratuito.
- Sob o mesmo orçamento total por janela/arquivo, comparar uniforme versus
  adaptativo. Medir cobertura/fração ROI, custos de endereços/máscaras e os ganhos
  dentro e fora de ROI, além de todas as guardas por família/configuração/canal.
  Refinar tudo é controle de maior taxa, não vitória da alocação adaptativa.
- Proteger eventos fracos, regiões sem ROI e transições/bordas; testar ROI vazia,
  saturada, sinal fraco, metadados ausentes e padrão fora do domínio. Definir um
  piso de fidelidade global e fallback explícito quando o orçamento não bastar.
- Se um seletor aprendido for considerado depois, usar somente treino e comparar
  com a política determinística; medir falsos negativos e custo de inferência.
  Não adotar saliência visual, GAN ou geração de textura para completar DAS.

**Critério de passagem:** melhor curva fidelidade–bits no sistema completo e
respeito ao piso global/caudas; ganho em ROI com degradação não autorizada fora
dela não satisfaz o objetivo generalista. A mudança da distribuição de qualidade
deve ser explícita e compatível com a aplicação.

### Etapa 9 — Seleção robusta, auditoria externa e modelo final

**Objetivo:** transformar um candidato de pesquisa em um sistema validado e reproduzível.

**Entregáveis a desenvolver/validar:**
- Protocolo escolhido por validação cruzada, múltiplas seeds e análise por grupo.
- Treino final nos dados de desenvolvimento com orçamento previamente definido,
  ou validação por configuração preservada para seleção/parada.
- Auditoria em novas aquisições/configurações fora da escolha de dados e hiperparâmetros.
- Pacote encoder/decoder e roteamento, se houver, formato versionado, testes de compatibilidade,
  documentação de taxa, normalização, geometria e limites conhecidos.

**Critério de passagem:** atendimento dos requisitos no sistema completo e no holdout
externo. Treinar em todos os dados de desenvolvimento não autoriza usar os mesmos
dados como evidência de generalização; o holdout externo permanece fora do treino.

## 4. Regras comuns de avaliação e decisão

1. **Mostrar várias agregações:** macro por configuração dentro da família, macro
   entre famílias, ponderação por janela, pior família e piores configurações.
   Declarar quais famílias cada fold contém; um fold parcial não valida o objetivo completo.
2. **Avaliar forma de onda e escala:** RMSE/MAE, correlação, SNR, amplitude/RMS,
   fase e erro espectral por banda física. Correlação alta não basta se amplitude
   estiver errada; magnitude espectral melhor não garante fase correta.
3. **Separar regiões e custos:** região ordinária em unidades físicas, escapes
   bit-exatos, máscaras/cobertura e overhead. Evitar RMSE global artificialmente
   favorável quando extremos dominam o denominador.
4. **Separar pesquisa e auditoria:** resultados de testes já consultados nos estudos
   históricos são exploratórios para decisões subsequentes. Reservar um novo
   holdout para a avaliação final e não reutilizá-lo na seleção de candidatos.
5. **Medir robustez com unidades independentes:** repetir seeds e usar aquisição,
   arquivo ou configuração como cluster; janelas correlacionadas não são réplicas
   independentes para intervalos de confiança.
6. **Medir o codec, não só o modelo:** incluir índices VQ, codificação entrópica,
   escapes, normalização, mapas espaciais, roteamento e container na taxa efetiva.
7. **Registrar decisões verificáveis:** uma alteração avança quando seu entregável
   e critério são satisfeitos; resultados negativos justificam mudar a hipótese,
   não aumentar automaticamente o tempo de treino.

## 5. Dependências e caminhos condicionais

```text
Requisitos e avaliação → integridade/pipeline → baselines e diagnóstico
                                               ↓
                     controles A/B/C → escala ~28M / ~100M em taxa compacta fixa
                                               ↓
                     ┌──────────────────────────┼──────────────────────────┐
                     ↓                          ↓                          ↓
          Lacuna de diversidade      Trade-off entre grupos       Limite de capacidade/taxa
                     ↓                          ↓                          ↓
          Dados FEBUS-like           ROI + piso global             Capacidade/contexto/quantização
                     └──────────────────────────┼──────────────────────────┘
                                               ↓
                             seleção robusta → auditoria externa → modelo final
```

As ramificações não são mutuamente exclusivas. Retornar ao diagnóstico sempre que
uma mudança de corpus, representação ou arquitetura alterar os pressupostos da
comparação. Se houver falha de integridade, corrigir e revalidar a pipeline antes
de atribuir o problema à arquitetura ou à quantidade de dados.
Especialização/roteamento são caminhos condicionais adicionais, não a resposta
automática a novas famílias ou um desvio obrigatório para reconhecimento OWR.

### Caminho de decisão para representações com maior densidade

1. Confirmar ganhos por configuração e inspecionar amplitude/fase e canais robustos,
   preservando inicialização e checkpoints alternativos como referências.
2. Confrontar arquiteturas/inicializações neurais no mesmo teto de bits por janela;
   comparar também um ponto compacto, não apenas a melhor fidelidade em taxa alta.
3. Se o ganho respeitar as tolerâncias concorrentes, priorizar repetição entre seeds
   e grupos reservados antes de escalar treino ou mudar várias peças da arquitetura.
4. Se RMSE melhorar com regressões físicas, investigar o mecanismo e realizar uma
   ablação isolada; mais épocas com o mesmo seletor não são a resposta automática.
5. Se o ganho depender de transmitir muito mais índices, investigar eficiência da
   representação, não apenas aumentar densidade. Quantização residual/multiescala,
   especialização e novos dados são hipóteses separadas, com payload/custos próprios;
   não substituem a comparação estrutural por mais continuações inalteradas.

Esse caminho refina as etapas 3, 5 e 8; não substitui requisitos de aceitação nem
a auditoria externa da etapa 9.

### Investigação estrutural sem mudança simultânea de corpus

Quando ganhos incrementais não aproximarem a fidelidade dos requisitos da aplicação,
investigar a etapa 8 sem exigir novas continuações inalteradas da etapa 5. Manter
uma referência fixa e comparar arquitetura/inicialização no mesmo corpus; ampliar
dados em estudo separado da etapa 6. Após os controles A/B/C, a ordem preferida é:

1. Testes de shape, gradiente, memória, compatibilidade e taxa dos candidatos.
2. Escala da família residual até ~100M, preservando o ponto compacto e separando
   capacidade de inicialização; A transferido permanece referência prática.
3. Orçamento adicional ou ablação de otimização quando não houver convergência,
   antes de concluir inferioridade estrutural.
4. Confirmação por seed/grupo de candidatos que melhorem a fronteira de fidelidade.
5. Ampliação de aquisições quando houver evidência de lacuna de diversidade,
   preservando protocolo fixo e holdout externo.

Codificação entrópica congelada pode ser investigada independentemente da escala,
sem alterar os símbolos nem o contraste de capacidade. ROI vem em estudo próprio
após uma base codificável controlada; novos quantizadores e atenção não são
acoplados ao primeiro teste de 100M. Se houver ganho compacto, construir curva
com pontos progressivamente mais comprimidos, incluindo a geometria offshore
original como controle. **700× é uma meta exploratória, não um resultado previsto.**

Não alterar arquitetura, inicialização, objetivo e dataset ao mesmo tempo nem
apresentar vantagem de uma triagem como prova de fidelidade suficiente.

### Requisitos operacionais transversais

- Planejar disco, GPU e duração antes de escalar; preservar originais e referências.
- Usar protocolos versionados, logs, locks, saves atômicos e resume verificável.
- Executar trabalhos longos em ambiente independente de uma sessão interativa;
  documentar a unidade real de retomada e a repetição de trabalho interrompido.
- Validar equivalência antes de deduplicar/migrar armazenamento; não apagar conteúdo
  distinto apenas por ser antigo. Caminhos compartilhados exigem gravação atômica.
- Manter comandos e procedimentos de recuperação nos relatórios ou na documentação
  operacional, sem transformar este roadmap em um diário de execução.

## 6. Base bibliográfica e prioridades derivadas

Revisão dirigida de artigos e projetos públicos, consultados em **29/10/2026**;
não é revisão sistemática exaustiva. Fontes com texto/código acessível têm maior
força para escolhas de implementação do que resumos. Não foi identificada, nesta
busca, demonstração de um codec neural único com avaliação conjunta nas quatro
famílias FEBUS, nossos alvos físicos e nossas guardas de canais. Isso não prova
inexistência de trabalhos equivalentes.

### 6.1 Convergência com compressão DAS

| Fonte e acesso | Evidência relevante | Convergência e limite para FEBUS |
|---|---|---|
| [Wang et al., Applied Optics, 2024 — RAE](https://doi.org/10.1364/AO.527625). Resumo e disponibilidade de dados na página da editora; texto integral restrito. | Autoencoder recorrente/LSTM; resumo relata SNR 40,60 dB em CR=8 e CR máximo 512. Dados remetem a Brady Hot Springs. | Apoia explorar contexto temporal; **não** assumir SNR 40,60 dB em CR=512. Não foi possível auditar tamanho de rede, quantidade efetiva de treino ou contabilidade integral de bits; 512× não é meta transferível diretamente. |
| [Chen et al., TGRS, 2025 — Deep Learning for Seismic Data Compression in DAS](https://doi.org/10.1109/TGRS.2025.3526933). Resumo público consultado; texto integral não auditado. | Encoder ViT e decoder; armazenamento de features e pesos do decoder; resumo anuncia ~50× em dados de escala GB. | Contexto não local é uma hipótese pertinente, não prova de generalização multifibra ou de superioridade de Transformer. Distinguir ajuste por registro de codec compartilhado. Não transplantar ~50× sem decompor downsampling, features e pesos. |
| [Ni et al. — Wavefield Reconstruction of DAS](https://doi.org/10.1029/2024JH000247). Resumo indexado; acesso integral bloqueado nesta revisão. | Compara INR e SHRED. Resumo relata generalização espaço-temporal de SHRED numa fibra com 20% dos canais e dificuldade em transientes de menor comprimento de onda; INR não generaliza sobre espaço/tempo. | Reforça verificar transientes/alta frequência e separar reconstrução de canais omitidos de codec que codifica todas as amostras. Não adotar INR ajustada por janela como generalista sem cobrar pesos e treino. |
| [AI4EPS/ZipDAS](https://github.com/AI4EPS/ZipDAS). README e código `train.py`/`zipdas/model.py` consultados. | Codec neural baseado em Ballé 2017, quantização escalar e prior fatorizado; custo taxa–distorção e empacotamento entrópico. Defaults do trainer: 128 filtros e patches 1.024×1.024. | Demonstra implementação neural DAS com stream, não necessidade de um VQ específico. Defaults não são ótimo validado. O loader de treino/validação usa a mesma fonte de dados no código consultado; não importar o split como evidência de generalização. Auditar normalização/escala física ao adaptar. |
| [Cullison, Li e Almomin — Stanford SEP-194, 2024](https://sep.sites.stanford.edu/publications/sep-reports/sep-194-2024), [projeto CAE](https://github.com/Haipeng-ustc/CS230-DAS-Compression). Resumo institucional e README. | CAE para DAS urbano; resumo descreve ~30× preliminares, redução de ruído e alterações em sinais quase estáticos. | Próximo no formato canal–tempo, mas denoising não é fidelidade ao bruto. Dataset disponível por contato, sem escala de treino reproduzível estabelecida aqui. |

Artigos clássicos recentes servem **apenas como contexto dos protocolos**:
[DASPack](https://doi.org/10.1093/gji/ggaf397) discute diversidade de 15 datasets
e custos/erros; [Ma et al., 2026](https://doi.org/10.1093/gji/ggag288) distinguem
regimes dominados por eventos e ruído em oito datasets. Não são alternativa de
produto, pré-requisito de benchmark, nem um limite teórico da compressão neural.
Diferenças de sinal, banda e tolerância impedem ranquear artigos só por CR.

### 6.2 Evidência transferível de codecs neurais próximos

**Escala de rede — evidência favorável, mas não uma lei universal.**
[Li et al., Scaling Learned Image Compression Models up to 1 Billion,
arXiv:2508.09075v1](https://arxiv.org/html/2508.09075v1) estudam redes HPCM de
**68,50M, 120,08M, 246,43M, 543,57M e 1.002M**. Texto integral consultado:
treino em Flickr2W, crops 256×256, batch 32 e dois milhões de updates;
ganhos taxa–distorção ao escalar. O modelo de 1B chega a ~0,3 dB a mais em altas
taxas frente aos melhores modelos comparados. A escala intermediária é analisada
principalmente num único ponto de lambda; a arquitetura inclui transformadas,
hiperprior e preditor entrópico. É um **preprint**, não resultado DAS e não
equivale a expandir só encoder/decoder com VQ fixo. Sua comparação também encontra
instabilidade ao escalar variantes Swin. A pequena inclinação do ajuste de escala
desaconselha prometer ganho proporcional ao número de parâmetros.

Como contraponto, [Luo, Rethinking Learned Image Compression: Context is All You
Need, arXiv:2407.11590v1](https://arxiv.org/html/2407.11590v1), também preprint,
relata retornos pequenos/não monotônicos ao alargar o backbone, e destaca o modelo
de contexto. Usa 8.000 imagens ImageNet e testa expansão para 16.000 sem ganho
expressivo nessa configuração. Há experimentos deliberados de overfitting em
Kodak: não importar seus resultados como validação independente nem suas
interpretações como teorema. A convergência útil entre os dois trabalhos é:
**onde colocar capacidade e como treiná-la importa; 100M é hipótese, não ótimo.**

**Bits por símbolo — aproveitar redundância, não só reduzir tokens.**
[Ballé et al., ICLR 2018 — Scale Hyperprior](https://arxiv.org/abs/1802.01436)
fundamentam modelagem probabilística com informação auxiliar cobrada. Em sinais,
[Défossez et al., EnCodec, 2022/2023](https://arxiv.org/abs/2210.13438) e
[código público](https://github.com/facebookresearch/encodec) usam RVQ: até 32
codebooks de 1.024 entradas, 10 bits por estágio/símbolo; número de estágios
controla taxa. Um modelo de probabilidade reduz o stream em até 40% naquele
contexto. Isso **não prevê 40% em FEBUS**, mas justifica medir entropia dos nossos
índices e depois bytes reais. Treino de EnCodec: 600.000 updates, batch 64 de
segmentos de um segundo, fontes diversas e oito A100; são exposições, não uma
receita de tamanho mínimo do dataset DAS. Seus objetivos perceptuais/adversariais
não são transferidos para reconstrução científica do bruto.

**Quantização — estabilidade separada de capacidade.**
[Mentzer et al., FSQ, ICLR 2024](https://arxiv.org/abs/2309.15505) substituem VQ
aprendida por poucas dimensões quantizadas em níveis fixos, com codebook implícito
igual ao produto dos níveis. É evidência para uma ablação de estabilidade/uso de
códigos, não prova de melhor RMSE DAS. Comparações exigem cardinalidade/bits
explícitos, incluindo estados não usados e projeções; utilização elevada por si
só não promove um codec. RVQ de áudio inspira refinamento progressivo, mas cada
estágio transmite índices adicionais: não confundir com ganho gratuito de taxa.

**ROI — alocação adaptativa com objetivo explícito.**
[Song, Choi e Han, ICCV 2021 — Spatially-Adaptive Feature Transform](https://arxiv.org/abs/2108.09551)
e [QmapCompression](https://github.com/micmic123/QmapCompression) condicionam um
codec em mapas de qualidade e distribuem bits espacialmente. Texto do artigo
consultado: melhorar uma tarefa pode piorar PSNR global; o mecanismo não garante
fidelidade uniforme. [Flexible Neural Image Compression via Code Editing,
NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/4f3820576130a8f796ddbf204c841487-Paper-Conference.pdf)
oferece precedente de ajuste de latentes/quantização para taxa e ROI com decoder
compartilhado, ao custo de otimização na codificação. Convergência: usar mapas ou
refinamentos, medir custos e separar dentro/fora da ROI. **Não importar saliência
visual, LPIPS/GAN ou texturas plausíveis como objetivos do sinal FEBUS.**

### 6.3 Escolhas prioritárias para o projeto

| Eixo | Prioridade proposta | O que a literatura não permite concluir |
|---|---|---|
| Capacidade | Comparar ~28M com ~100M na mesma família residual e taxa compacta; ~50M é ponto diagnóstico opcional. | Não há tamanho ótimo publicado para nosso corpus; não prever recuperação de 700× por quadruplicar parâmetros. |
| Arquitetura | CNN residual escalável; contexto em baixa resolução e multiescala. Atenção fatorizada é ablação posterior. | ViT/LSTM bem-sucedidos em uma aquisição não estabelecem vencedor entre as quatro fibras. |
| Dados | Corpus auditado para o piloto; curvas por grupos inteiros e novas aquisições para robustez externa. | As 28.780 janelas preparadas dos 207 HDF5 não são 28.780 realizações independentes, nem todas pertencem ao treino. Não há fórmula validada de janelas por parâmetro. |
| Quantização | Primeiro K=2.048, dimensão 128, mesma grade; depois entropia e ablações K/FSQ/RVQ isoladas. | Mais códigos usados ou maior dimensão latente não garantem mais informação útil por bit. |
| ROI | Base global + refinamento residual local sob orçamento total, com mapa decidido no encoder e piso global. | Não garante economia se todas as regiões forem difíceis; ROI não deve esconder sinais fracos ou aumentar custo sem contabilização. |

Uma regra para comparar eficiência é separar **contagem de índices**, **bits por
índice** e **custos auxiliares**. No empacotamento fixo:

`B_total = N_indices × ceil(log2 K) + B_normalization + B_escapes + B_maps + B_container`.

Em múltiplos estágios/níveis, somar seus custos; em entropy coding, substituir o
termo fixo pelos bytes efetivos dos símbolos. Pesos compartilhados são custo de
implantação/amortização, não payload obrigatório de toda janela; pesos ou adapters
ajustados por aquisição devem ser cobrados. Para ~100M, só pesos FP32 representam
~400 MB decimais, fora ativações, gradientes, optimizer e codebooks.

O offshore ~744,73× com correlação ~0,9017 é referência motivadora do artigo
fornecido, não controle pareado: alvo, bandas, agregação, overhead e unidade física
precisam ser harmonizados. Não estabelecer equivalência com RMSE denormalizado ou
com a macro atual. A ordem de decisão é **capacidade a taxa fixa → confirmação
por grupos/seeds → ROI com taxa total controlada**, com estudo independente de
entropia e diversidade. Requisitos físicos e auditoria externa acompanham essa
investigação; não a substituem e continuam obrigatórios para promoção final.
