# Progress reports — índice por etapa

Os relatórios 1–89 não foram alterados. A consolidação inicial reuniu os 32 registros históricos de 90–121 em **13 documentos**; novas etapas foram acrescentadas depois. A numeração atual segue em sequência a partir do 90. Os números dos registros históricos e suas âncoras foram preservados.

| Documento | Registros anteriores | Etapa e conclusão |
|---|---|---|
| [90 — Revisão de fidelidade por bit](90-FIDELITY_PER_BIT_NEXT_RESEARCH_REVIEW.md) | 90 | Direção e hipóteses da sequência; não é resultado de codec. |
| [91 — Preparação prospectiva e oportunidade espacial](91-PROSPECTIVE_PREPARATION_AND_SPATIAL_OPPORTUNITY.md) | 91 | Preparação e contrato de métricas/taxa, com diagnóstico inicial. |
| [92 — H1 — lifting, coder e alocação por subbanda](92-LEARNED_LIFTING_STUDY.md) | 92, 93, 94, 95 | Quatro braços concluídos; economia lossless do coder não promoveu H1 sobre o VQ. |
| [93 — H2 — âncoras neurais e inovação](93-NEURAL_ANCHOR_INNOVATION_STUDY.md) | 96, 97, 98 | Preflight congelado e treino conjunto separados; ganho contra o controle, não contra o VQ maduro. |
| [94 — Observação completa no encoder](94-ENCODER_OBSERVATION_STUDY.md) | 99, 100 | Ganho pequeno; gate de extensão reprovado, sem promoção. |
| [95 — Validação de desenvolvimento congelada](95-FROZEN_DEVELOPMENT_VALIDATION.md) | 101, 102 | Cobertura integral dos cinco modelos; sem reseleção ou confirmação independente. |
| [96 — Diagnóstico de gargalos full-tile](96-FULL_TILE_BOTTLENECK_DIAGNOSTIC.md) | 103, 104 | Descritores de erro, índices e covariância; não são limites informacionais. |
| [97 — Produto VQ denso e curva de taxa congelada](97-DENSE_PRODUCT_AND_RATE_MATCH_STUDY.md) | 105, 106, 107, 108 | Economia de bytes com mais erro; a grade medida não demonstrou dominância RD. |
| [98 — Metadados físicos, clocks e proveniência](98-PHYSICAL_METADATA_AND_PROVENANCE.md) | 109, 110, 111, 112 | Cabeçalhos e relógios distinguem recortes, mas não certificam independência de aquisições. |
| [99 — Oportunidade multicore nominalmente alinhada](99-CLOCK_ALIGNED_MULTICORE_OPPORTUNITY.md) | 113, 114 | Linear e magnitude condicional concluídos sem oportunidade: 24/24 fallbacks por braço; modelo acrescentou 0,135% aos bytes. Sem extensão. |
| [100 — Normalização interna local](100-LOCAL_INTERNAL_NORMALIZATION_STUDY.md) | 115, 116 | Via residual ativa, sem avanço prático; sem extensão. |
| [101 — Consolidação dos budgets e decisões](101-RESEARCH_SEQUENCE_CONSOLIDATION.md) | 117 | Registro inicial de 245.760 updates preservado; fechamento posterior totaliza 356.352, com CNN/MLP, tipos de exposição e coortes discriminados. |
| [102 — Decisão discreta de índices e auditoria ampliada](102-DIRECT_DISCRETE_INDEX_STUDY.md) | 118, 119, 120, 121 | Redução escalar com regressões físicas; sem promoção ou distilação. |
| [103 — H4: correção com erro controlado](103-ERROR_BOUNDED_RESIDUAL_STUDY.md) | — | Garantia verificada; residual neural causal economizou 0,378% com bits físicos idênticos, abaixo do gate. Sem extensão. |
| [104 — Dados: composição e diversidade](104-SEMISYNTHETIC_AUGMENTATION_STUDY.md) | — | Composição reprovada; curva 4/6/10 concluída, não monotônica e sem ganho de fidelidade por bit demonstrado. Sem extensão ou promoção. |
| [105 — H3: condicionamento físico](105-PHYSICAL_CONDITIONING_STUDY.md) | — | Três braços concluídos; hardware falhou nos quatro critérios do gate, com mais bytes e pior erro que categoria nas combinações excluídas. Sem extensão. |
| [106 — Viabilidade e novo direcionamento](106-FEBUS_FEASIBILITY_AND_RESEARCH_REDIRECTION.md) | — | Revisão crítica de dados, modelo, taxa e literatura após H1–H5. Verificação de metadados/artefatos; recomenda diagnóstico de viabilidade e sessões documentadas antes de outra candidata neural. Sem treino ou leitura de waveforms. |
| [107 — Contratos e execução delegável](107-FEASIBILITY_CONTRACT_AND_EXECUTION_HANDOFF.md) | — | Definições F0–F4, métricas prospectivas, planejamento de contexto e painel F1 de 20 arquivos. Pool F2 reduzido a 32 arquivos por vínculos temporais; ordens de execução e gates explícitos. Sem inferência ou treino FEBUS. |
| [108 — Diagnósticos de viabilidade](108-FEASIBILITY_DIAGNOSTICS_EXECUTION.md) | — | F1 concluído: 414 forwards e sensibilidade de contexto, sem ganho de codec demonstrado. D2b validado em 50 casos sintéticos, roundtrip bit-exato e zero violações; D2c bloqueado por semântica dos clocks. |
| [109 — Contexto com cobertura e taxa completas](109-FEBUS_FULL_COVERAGE_CONTEXT_CONTRAST.md) | — | `part8` reduz o erro do FiLM em 48,5%, com 140,3% mais bytes; não supera a referência madura. |
| [110 — Sensibilidade do contrato temporal](110-FEBUS_TEMPORAL_CONTRACT_SENSITIVITY.md) | — | Envelopes condicionais tornam o plano candidato mais restrito e retiram o contraste multicore; D2c não foi liberado. |
| [111 — Compatibilidade de treino e inferência](111-FEBUS_TRAIN_INFERENCE_CONTEXT_ALIGNMENT.md) | — | F4-B melhora o controle de crop, mas perde para a VQ madura em erro e bytes. |
| [112 — Redundância em janelas de 50 ms](112-FEBUS_SHORT_WINDOW_ENTROPY_TRIAGE.md) | — | Triagem F2: economia de transporte, mas não no objeto autocontido. |
| [113 — Amortização e transferência](113-FEBUS_FROZEN_INSTALLATION_AMORTIZATION.md) | — | Economia líquida em outros arquivos, sem passar o critério nas duas precisões. |
| [114 — Decisão de implantação](114-FEBUS_DEPLOYMENT_DECISION_AUDIT.md) | — | Taxas com instalação e marcos provisórios de progresso. |
| [115 — Tabelas contextuais compartilhadas](115-FEBUS_POOLED_CONTEXT_TABLES.md) | — | Menor instalação, mas mais bytes totais. |
| [116 — Wire VQ compacto](116-FEBUS_COMPACT_VQ_WIRE_PROTOCOL.md) | — | Economia de 2,9944%, concentrada em dois tiles literais. |
| [117 — Curva F2 contínua](117-FEBUS_CONTINUOUS_F2_PROBE.md) | — | Economia de 30,38–41,27% no transporte nas precisões primárias; com instalação, o critério conjunto por família/configuração não passou. |

## Rastreabilidade

- As referências do roadmap usam os números atuais da primeira coluna. Registros históricos dentro das etapas conservam seus identificadores originais.
- Dentro de cada etapa, protocolo e resultado continuam explicitamente separados.
- Os números em protocolos, artefatos e relatórios antigos não foram renumerados.
- [Originais byte a byte, incluindo README/roadmap anteriores](archive/post_89_consolidation_20261115/originals.tar.gz).
- [Manifesto SHA-256 da consolidação](archive/post_89_consolidation_20261115/manifest.json).

Resultados, checkpoints, fontes congelados e datasets não foram movidos ou reescritos. Para próximas etapas, atualizar o mesmo documento com protocolo e resultado, em vez de criar um progress report por atualização operacional.
