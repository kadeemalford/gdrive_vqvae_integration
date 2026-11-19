# Consolidação 90–116: evidência preservada, sem promoção ou novo sweep

**15/11/2026. Consolidação somente de artefatos, sem sinais ou inferência.**

## Orçamento efetivamente concluído

Históricos científicos foram conferidos para endpoint e monotonicidade.

| Campanha | Updates científicos de otimização |
|---|---:|
| H1 lifting, quatro braços | 131.072 |
| H2 conjunto, dois braços | 65.536 |
| Observação completa do encoder | 16.384 |
| Produto denso full-tile | 16.384 |
| Normalização interna local | 16.384 |
| **Total desde a revisão90** | **245.760** |

Inclui otimização dos passos no H1 fixo, não apenas pesos neurais.
Não inclui pré-treino histórico, smokes, testes de retomada, iterações Lloyd ou
inferência. Updates de arquiteturas distintas não equivalem a mesmo FLOP/tempo.

Os históricos/auditoria registram **19.243.252.000 apresentações de células**
nessas otimizações. Não são 19 bilhões de amostras novas ou independentes:
há reuso dos mesmos dados entre updates, braços e campanhas.

## O que foi demonstrado e o que não foi

- **Coder:** economia lossless importante nos coeficientes de H1, sem promover
  sua representação. Predição inteira posterior acrescentou pouco.
- **H2:** predição/inovação conjunta melhorou seu controle e transferiu esse
  ganho para a validação de desenvolvimento congelada; continuou inferior à
  referência prática em erro e bytes.
- **Observação do encoder:** ganho pequeno, gate reprovado; nenhuma extensão.
- **Produto:** reduziu bytes com mais erro. A grade congelada de taxa não
  demonstrou dominância; não interpolar para declarar vencedor.
- **Normalização interna local:** via ativa, sem avanço da fronteira e com
  regressões físicas. Não prolongar por pequena melhora de endpoint.
- **Proveniência:** dez horários de início de cabeçalho não são dez arquivos
  idênticos; existem 81 clocks distintos no treino e nenhuma coincidência nos
  fingerprints não constantes amostrados. Isso não certifica independência.
- **Multicore:** o diagnóstico linear nominalmente alinhado falhou no gate.
  Não inferir dele que toda dependência entre núcleos seja inexistente.

Nenhuma dessas conclusões prova um limite informacional do DAS. O VQ maduro
continua a referência de alta fidelidade disponível, não um ótimo universal.

## Populações que não podem ser misturadas

O registro legível por máquina preserva três coortes distintas:

1. 129 checks P1 train-only: cinco pontos selecionados/comparador.
2. 136 checks train-only com tiles pequenos: seis modelos/selecionados/endpoints
   e 12 pontos da curva congelada.
3. 7.220 janelas de validação de desenvolvimento: cinco checkpoints previamente
   congelados, **não** confirmação independente.

O script confere arquivos, grupos e contagens de células dentro de cada coorte
e **não cria ranking global entre coortes**. Os novos modelos reprovados não
foram enviados à validação para buscar um resultado favorável.
Os 28 pontos catalogados não são 28 ensaios independentes.

Artefatos: `results/experiments/research_decision_registry_20261115/registry.json`.
Reprodução:

```bash
# Use destino novo: o script rejeita sobrescrever a consolidação.
.venv/bin/python -m finetuning.consolidate_research_decisions \
  --output results/experiments/research_decision_registry_recheck_20261115
```

## Decisão de continuação

Manter a referência madura e preservar todos os negativos e seus códigos.
**Não iniciar P3, prolongar pilotos reprovados ou lançar outro sweep de
LR/gates/seed por completar esta sequência.** Isso seria reuso adaptativo do
mesmo desenvolvimento sem evidência de avanço.

Prioridade de maior valor: novas sessões FEBUS com proveniência, unidades e
excitação documentadas, ou uma hipótese estrutural que explicite a diferença
frente aos mecanismos já testados e tenha controle de taxa/capacidade.
Derivados FFT/processados não substituem novas aquisições.
O template/validador de 110 organiza essa entrada, mas não fabrica evidência.

A proteção do teste fold0 e EDAS continua. Tampouco se reclassificam o gate
patológico de 89 ou a validação consumida como confirmação independente.
Não reduzir tolerâncias/guardas desconhecidas da aplicação para aprovar um
modelo por média. Mais autonomia não implica autorização para inventar
proveniência, redefinir o alvo ou gastar orçamento sem hipótese identificável.

Verificação final: **463 testes passaram**. Uma reprodução da consolidação
em destino novo produziu `registry.json` **idêntico byte a byte**, sem
reabrir waveforms. Código/documentação e auditorias foram sincronizados no
remoto, preservando divergências; sem commit/push.

### Diagnóstico posterior, sem updates de treinamento

Os probes118–120 de índices congelados estão fora da população/versão deste
registro. A redução escalar não passou pelas guardas físicas; decisão em
[121](102-DIRECT_DISCRETE_INDEX_STUDY.md#report-121).
Não adicioná-los silenciosamente às coortes/28 pontos já congelados aqui.

## Continuação posterior — fechamento em 16/11/2026

Esta seção usa a **numeração atual dos documentos** e não reescreve o registry
original nem mistura populações. Depois do orçamento histórico acima:

| Etapa posterior | Updates novos | Decisão |
|---|---:|---|
| [103 — residual neural com precisão controlada](103-ERROR_BOUNDED_RESIDUAL_STUDY.md) | 8.192 | Ganho de bytes de 0,378%, insuficiente; sem extensão. |
| [104 — composição semissintética](104-SEMISYNTHETIC_AUGMENTATION_STUDY.md) | 16.384 | Mais erro que controle real; sem promoção. |
| [104 — diversidade de configurações](104-SEMISYNTHETIC_AUGMENTATION_STUDY.md) | 24.576 | Curva não monotônica, sem ganho de fidelidade por bit demonstrado. |
| [105 — hardware/categoria/constante](105-PHYSICAL_CONDITIONING_STUDY.md) | 49.152 | Gate reprovado em todos os quatro critérios; sem extensão. |
| [99 — magnitude condicional multicore](99-CLOCK_ALIGNED_MULTICORE_OPPORTUNITY.md) | 12.288 | Todos os checks em fallback; modelo apenas acrescentou custo. |
| **Adicional** | **110.592** | Nenhum candidato promovido. |

Total nominal desde a revisão 90: **356.352 updates científicos de otimização**.
Inclui otimização dos passos H1 fixos e treino de pequenos MLPs, não apenas
backbones CNN; não é uma unidade uniforme de compute nem de aprendizagem.
Smokes, pré-treino histórico, Lloyd e inferência continuam excluídos.

Contagens separadas: **21.144.266.000 apresentações de células físicas de
treinamento**, incluindo portadores compostos, e **83.886.080 apresentações
de alvos de coeficientes** nos MLPs de H4/H5. Não somá-las como novas amostras
ou aquisições, nem esconder a diferença de representação e contexto.
Endpoints/históricos foram conferidos e os hashes registrados em:
`results/experiments/remaining_hypotheses_closeout_20261116/registry.json`.
Essa consolidação não treinou e não abriu waveforms.

Coortes de avaliação continuam distintas:

- 136 checks / 43 arquivos / dez grupos para os estudos full-tile;
- subconjuntos H3 de 115 janelas vistas e 21 de combinações excluídas;
- 24 recortes / oito pares multicore / 16 arquivos / 3.360.000 células conjuntas
  no novo contraste H5;
- validação de desenvolvimento histórica consumida não foi reaberta e nunca
  se tornou confirmação independente.

H3 teve um teste restrito real; H5 teve triagem linear e outro mecanismo de
magnitude neural, sem rescindir o gate original. Isso fecha os desenhos
registrados, **não todas as variantes possíveis dessas hipóteses**.
Novas sessões independentes continuam pendentes. Sem candidato elegível,
não lançar P3 nem outro sweep para salvar os negativos. A referência madura
permanece default e os sinais teste/EDAS continuam reservados.

Fechamento de engenharia desta continuação: **498 testes passaram**,
`git diff --check` limpo e 294 links locais válidos. Numeração sequencial
1–105 preservada, relatórios 1–89 intactos e nenhum JSON ativo de mapeamento.
