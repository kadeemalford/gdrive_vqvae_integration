# PROGRESS CHECKPOINT 3 — Fidelidade por bit: do VQ ao uso seletivo de contexto

**Consolidação: 19/11/2026.** Continuidade do [checkpoint 2](PROGRESS_CHECKPOINT_2.md).
Cobre os **45 relatórios de 80 a 124**, pela numeração atual do [índice](README.md).

**Resultado geral:** a VQ madura continua sendo a referência neural. Nenhum
novo reconstrutor demonstrou maior fidelidade generalista no mesmo orçamento
de bytes. O avanço mais consistente foi responder à pergunta inversa: **fixada
a precisão do waveform, quando é possível transmitir menos bytes?** No FEBUS,
decidir pelo ajuste quando usar os canais e tempos vizinhos economizou
**21,75% e 31,82%** sobre o marginal no painel de desenvolvimento, já cobrando
a instalação. O método também economizou em um projeto externo e empatou em
outro. É um caminho de implantação seletiva; não demonstra alta fidelidade a
700×, nem substitui a VQ no seu ponto de operação.

## 1. Onde houve ganho de taxa sem mudar a reconstrução

O controle **marginal** codifica cada valor quantizado sem consultar vizinhos.
O **espaço-temporal** usa vizinhos já reconstruídos pelo decoder para atribuir
probabilidades aos mesmos valores. Nos dois modos chegam os mesmos índices e
o mesmo waveform quantizado, inclusive nos trechos ruidosos. Isso não significa
restaurar o float32 original sem perdas: o erro absoluto por célula é limitado
por ε. As precisões 0,03 e 0,1 são frações da escala fixa de calibração da
configuração, não percentuais da amplitude instantânea nem tolerâncias de uso.

| FEBUS: economia contra o marginal sob o contrato de cada linha | ε/σ = 0,03 | ε/σ = 0,1 |
|---|---:|---:|
| [117 — Contínuo](117-FEBUS_CONTINUOUS_F2_PROBE.md): só transporte | 30,38% | 41,27% |
| Mesmos trechos e tabelas; instalação cobrada uma vez | 0,83% | 27,41% |
| [119 — Instalação compactada](119-FEBUS_LOSSLESS_INSTALLATION_RESULTS.md): contextual sempre instalado, marginal também reempacotado | 10,19% | 32,36% |
| [121 — Instalação escolhida só no ajuste](121-FEBUS_FIT_ONLY_INSTALLATION_ROUTING_RESULTS.md): economia líquida da política | **21,75%** | **31,82%** |

**O custo das tabelas muda a decisão.** Na precisão estrita do 117, o transporte
encolheu em todas as famílias, mas entregar objetos autocontidos aumentou o total
em **8,57%**. O [envelope reversível](118-FEBUS_LOSSLESS_INSTALLATION_ENVELOPE.md)
restaurou as 243 instalações byte a byte e reduziu o custo sem mudar o modelo.
Ainda assim, instalar contexto em toda parte penalizou algumas configurações.
Os percentuais da tabela **não são cumulativos**.

**A solução mais útil foi escolher antes de avaliar.** No [120](120-FEBUS_FIT_ONLY_INSTALLATION_ROUTING.md),
uma configuração só recebia contexto se ele economizasse pelo menos 3% dos
bytes completos em **cada** arquivo retirado do ajuste. Após 40 comparações
desse tipo, foram congeladas 18 decisões. A 0,03, só *single-mode/Configuration 8*
recebeu contexto; a 0,1, também *single-mode/Configuration 3* e
*multimode/Configuration 7*. As economias equivaleram a **1,263 e 1,293
bits/célula**, sem regressão material por configuração. A política abriu mão
de parte do melhor agregado a 0,1 em troca de evitar perdas locais.

**Alcance:** nove configurações de quatro famílias, com multicore C2/C3,
mas sem C1/C4. Famílias inteiras ficaram no marginal e economizaram zero;
logo, o critério generalista do [107](107-FEASIBILITY_CONTRACT_AND_EXECUTION_HANDOFF.md)
não passou. As avaliações anteriores já tinham usado esse painel. O
[122](122-FEBUS_ROUTING_CONFIRMATION_PREFLIGHT.md) identificou por hash as
18 decisões e os 22 envelopes para confirmação futura; não admitiu sessões novas.

## 2. O que se repetiu fora do FEBUS

Os relatórios [123 — EGS](123-EXTERNAL_DAS_ROUTING_EGS.md) e
[124 — Utah FORGE](124-EXTERNAL_DAS_ROUTING_FORGE.md) replicaram **o método
de seleção**: ajustaram tabelas no próprio projeto e congelaram a escolha
antes de ler os sinais de avaliação. Não aplicaram a instalação FEBUS a
outro sensor. A tabela mostra bytes completos poupados pela política contra
o marginal, à mesma reconstrução **dentro de cada acervo**.

| Produto e avaliação | ε/σ = 0,03 | ε/σ = 0,1 | O que decidiu o resultado |
|---|---:|---:|---|
| EGS/Terra15, velocidade nativa em m/s a 36 kHz; duas gravações de ajuste e duas posteriores de avaliação | **0%** | **0%** | A regra manteve o marginal. A 0,1 o contextual teria poupado 3,56% e 8,84% nas avaliações, mas a vantagem não foi estável nos dois folds de ajuste. |
| Utah FORGE, SEG-Y proporcional a *strain-rate*; dois estágios de ajuste e dois de avaliação, com o mesmo evento observado em dois poços | **4,36%** | **15,83%** | Economia nos dois eventos em três das quatro combinações de poço e precisão. Em 16B a 0,03, ficar no marginal evitou aumentos de 9,53% e 10,73%. |

O contraste do FORGE explica por que vale manter a escolha por configuração:
**para os mesmos eventos**, a 0,03 o poço 78B economizou **7,95% e 9,64%**
usando contexto, enquanto 16B perderia bytes. Já o Terra15 mostra o custo
da prudência: proteger contra regressão pode deixar passar uma economia real.
Não se reajustou a regra pelos resultados. Dois poços simultâneos não são
duas repetições independentes; poço, fibra, montagem e instrumento podem
variar juntos. Não agregar os dois projetos como uma só população.

## 3. Sinais neurais que esclareceram o caminho

| Iniciativa | Resultado pareado | Decisão |
|---|---|---|
| **Âncoras espaciais e inovação** ([93](93-NEURAL_ANCHOR_INNOVATION_STUDY.md), [95](95-FROZEN_DEVELOPMENT_VALIDATION.md)) | Na validação de desenvolvimento, H2 teve **8,48% menos erro e 1,50% menos bytes** que o controle independente. Contra a VQ madura, teve **25,12% mais erro e 12,07% mais bytes**. | A dependência espacial é aproveitável; o reconstrutor testado não melhora a referência prática. |
| **Compatibilidade entre treino e inferência** ([108](108-FEASIBILITY_DIAGNOSTICS_EXECUTION.md)–[111](111-FEBUS_TRAIN_INFERENCE_CONTEXT_ALIGNMENT.md)) | 414 inferências em 20 arquivos mostraram sensibilidade à largura do contexto. Em F4-B, treinar na geometria completa reduziu o erro **40,18%** contra treinar em recortes, com **4,40% mais bytes**. Ainda ficou **27,01% pior em erro e 2,74% maior em bytes** que a madura. | Usar geometria representativa em estudos futuros. O diagnóstico explicou parte do mau desempenho do controle; não justificou estender F4-B. |
| **Pré-treino externo PoroTomo** ([83](83-POROTOMO_PRETRAINING_BRIDGE_RESULTS.md)) | Com os mesmos 8.192 updates finais em FEBUS, a ponte reduziu RMSE **4,54%**, com mais treino e **4,09% mais bytes**. A orçamento total de 24.576 updates, perdeu para treino direto em FEBUS: **5,49% mais RMSE e 1,73% mais bytes**. | Dados externos ajudaram a inicialização com pouca adaptação ao alvo; não foram o melhor uso do orçamento medido. |

O ponto de referência de F4-B é **NRMSE\* 0,49034 a 0,69969 bits/célula**.
As economias de F2 são contra um **marginal de precisão explícita**, não
contra essa VQ. Também não se deve comparar diretamente o RMSE macro dos
primeiros relatórios com o NRMSE\* prospectivo, que usa um piso por grupo
nos canais de variação pequena.

## 4. Demais iniciativas e mapa dos relatórios

| Relatórios | O que acrescentaram e por que não lideram a próxima etapa |
|---|---|
| [80](80-LEARNING_RATE_AND_EXTERNAL_PRETRAINING_REVIEW.md)–[83](83-POROTOMO_PRETRAINING_BRIDGE_RESULTS.md) | A mudança de agenda de aprendizado e a ponte QuantX não justificaram extensão; PoroTomo distinguiu benefício de inicialização de vantagem a orçamento pareado. |
| [84](84-FIXED_ID_SYNTHESIS_AND_CONTEXT_PROBES.md)–[89](89-JOINT_PRODUCT_VQ_LONG_TRAINING_PROTOCOL.md) | Síntese dos mesmos IDs, contexto antes/depois do VQ, atenção com treino conjunto e troca de objetivo não mudaram o patamar. MSE ganhou 0,27% de RMSE sobre o controle, pagando 0,83% mais bytes. Escalas locais encareceram os pacotes; o produto VQ não justificou treino longo. |
| [90](90-FIDELITY_PER_BIT_NEXT_RESEARCH_REVIEW.md)–[95](95-FROZEN_DEVELOPMENT_VALIDATION.md) | Preparação prospectiva e comparação espacial. No lifting H1, recodificar coeficientes poupou cerca de 53–60% nos pontos destacados, sem superar a VQ. H2 foi o contraste espacial mais informativo; ampliar a observação do encoder pouco acrescentou. |
| [96](96-FULL_TILE_BOTTLENECK_DIAGNOSTIC.md)–[102](102-DIRECT_DISCRETE_INDEX_STUDY.md) | Diagnósticos de resíduos/latentes, produto denso, metadados e clocks, contexto multicore, normalização local e auditoria de orçamento. O produto trocou bytes por erro; a decisão discreta melhorou o NRMSE\* em 0,994%, mas piorou o erro quadrático físico em oito configurações. |
| [103](103-ERROR_BOUNDED_RESIDUAL_STUDY.md)–[107](107-FEASIBILITY_CONTRACT_AND_EXECUTION_HANDOFF.md) | H4 verificou o limite de erro, mas a correção neural poupou só 0,378% à mesma reconstrução. Composição semissintética, curva 4/6/10 e condicionamento por hardware não demonstraram ganho generalista. A [revisão 106](106-FEBUS_FEASIBILITY_AND_RESEARCH_REDIRECTION.md) redirecionou a investigação para precisão, taxa e diversidade identificável. |
| [108](108-FEASIBILITY_DIAGNOSTICS_EXECUTION.md)–[111](111-FEBUS_TRAIN_INFERENCE_CONTEXT_ALIGNMENT.md) | Replay e F4-B separaram efeito de contexto de ganho de codec. Partir o FiLM em oito canais reduziu erro em 48,52% contra seu modo completo, mas custou 140,33% mais bytes. Cinquenta controles sintéticos verificaram o método de F2; a execução contínua dependia da interpretação dos clocks. |
| [112](112-FEBUS_SHORT_WINDOW_ENTROPY_TRIAGE.md)–[117](117-FEBUS_CONTINUOUS_F2_PROBE.md) | Em janelas de 50 ms, 19 arquivos adicionais deram **5,20% / 14,10%** de economia líquida; não são 19 sessões documentadas. Compartilhar tabelas espaciais aumentou o total; o transporte VQ compacto poupou 2,9944%, concentrados em dois tiles. Sob a premissa aceita de relógio centrado e comparável, F2 contínuo mediu nove precisões em nove configurações. |
| [118](118-FEBUS_LOSSLESS_INSTALLATION_ENVELOPE.md)–[122](122-FEBUS_ROUTING_CONFIRMATION_PREFLIGHT.md) | Compactação reversível da instalação, roteamento decidido no ajuste e preparação da confirmação FEBUS. O ganho líquido e a ausência de vantagem universal estão discriminados na seção 1. |
| [123](123-EXTERNAL_DAS_ROUTING_EGS.md)–[124](124-EXTERNAL_DAS_ROUTING_FORGE.md) | Replicações externas da seção 2. No EGS, misturar duas gravações para ajustar as contagens, mantendo fixo o número de células, poupou apenas **0,008–0,276%**. Não foi uma curva de diversidade de treino neural. |

## 5. Estado da pesquisa e próximos passos

- **Preservar** a VQ madura como referência neural e a política marginal/contextual
  como candidata de implantação seletiva. H1–H5 e F4-B estão encerrados nos
  desenhos testados. Os 356.352 updates da sequência consolidada em 101/106
  não são ensaios independentes nem prova de limite informacional.
- **Para confirmar o resultado FEBUS**, priorizar uma nova aquisição de
  *single-mode/Configuration 8* com a mesma geometria, usando a rota congelada
  sem reajuste. C1/C4 precisa de evidência própria. Preservar teste fold0 e
  reservas EDAS Polyimide/Acrylate. Para separar efeitos entre fibras,
  procurar pares monomodo/multimodo sob a mesma excitação e com proveniência;
  o 124 não avaliou tais pares.
- **Enquanto faltam sessões**, fechar F0 com quem usa os dados: que erro de
  amplitude, fase ou pico muda uma conclusão científica, por quanto tempo se
  amortiza a instalação e que granularidade de acesso é necessária? Os marcos
  do [114](114-FEBUS_DEPLOYMENT_DECISION_AUDIT.md) — 2–3% menos NRMSE\* sem mais
  bytes ou 3–5% menos bytes à mesma fidelidade — orientam investimento; não
  substituem tolerâncias físicas nem alteram os critérios anteriores.
- **F3 e F4-A:** a curva por sessões independentes no FEBUS permanece aberta.
  O modelo neural de probabilidades dos valores quantizados continua condicional:
  o ganho seletivo justifica investigar onde há economia repetível, mas não
  satisfaz o critério de investir num modelo para as quatro famílias.

Síntese documental, sem nova execução. Protocolos, populações e artefatos
nos relatórios vinculados são a referência para reproduzir cada contraste.
