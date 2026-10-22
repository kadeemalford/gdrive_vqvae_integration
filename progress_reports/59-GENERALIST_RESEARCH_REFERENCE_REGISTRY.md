# Registro consolidado 4x: referências disponíveis, sem promoção final

## Escopo e integridade

Runner: `finetuning/summarize_generalist_frontier.py`.
Evidência: `results/experiments/generalist_4x_registry_20261020/`:
`profiles.csv`, `research_choices.json`, hashes/protocolo e `COMPLETE`.
**11 candidatos 4x** dos estudos recentes, sete no fold 0 e quatro no fold 1.
Não é inventário exaustivo de todas as taxas/arquiteturas históricas.

Sem novas inferências, otimização, datasets ou modificações de checkpoint.
Os 11 checkpoints estão disponíveis e seus **SHA-256 coincidem com o registrado
na avaliação**. Um checkpoint ausente seria marcado como evidência histórica,
não escolhido como candidato utilizável.

Comparações permitidas apenas dentro de coortes com fold, coordenadas de
validação, referência/normalização, geometria, K, parâmetros e custos de
índices/side idênticos. Não fazer pooling de scores robustos do fold 0 com
denominadores escape-v1 do fold 1 ou taxas pooled de corpus diferentes.

## Referências de pesquisa sob a guarda retrospectiva

Menor RMSE macro disponível entre candidatos que preservam RMSE/correlação/p90
por configuração contra o source da própria coorte:

| Coorte | Referência de pesquisa | RMSE macro | Correlação | p90 macro por canal | Compressão nominal |
|---|---|---:|---:|---:|---:|
| Fold 0 / robust | Auxiliar 0,05, fase 8 | 0,476394 | 0,870237 | 0,712982 | 42,811x |
| Fold 1 / escape-v1 | **Continuação histórica epoch 39** | 0,477692 | 0,871184 | 0,810105 | 42,133x |

Arquivos de referência, sem copiar/promover para um modelo de produção:

- Fold 0: `results/experiments/channel_tail_pilot_20261019/channel_tail/best_macro.pt`.
- Fold 1: `results/experiments/high_rate_continuation_20261015/rate_4x/best_macro.pt`.

**São duas redes/partições distintas, não um único modelo validado nos dois folds.**
Não montar roteador por fold/configuração, nem avaliar uma rede em grupos que
estiveram no seu treino e chamar isso de generalização independente.

Este ranking é **retrospectivo**, com histórias de compute/optimizer diferentes.
Epoch 39 histórico preservou AdamW/RNG e usou mais updates adicionais que o
piloto de oito épocas; sua vantagem não prova superioridade causal de LR,
optimizer ou de um objetivo. Não reescrever os seletores dos estudos anteriores.

## Verificação específica do candidato histórico

`generalist_guard.assess` não encontrou violações frente ao source epoch 24 nas
seis configurações do fold 1. Isso não resgata a confirmação negativa do termo
auxiliar: o candidato histórico usa o objetivo base e outra trajetória de treino.

Pareamento físico também concluído, sem inferência:
`results/experiments/historical_rate4_fold1_vs_baseline_20261020/`.
Usa **`continued_rate4.windows.attempt_002.csv` completo (5.820 janelas)**;
attempt_001 tinha 2.503 e foi preservado, não misturado ou confundido com completo.
Referência: baseline do piloto atual, mesmo source/referência/coordenadas/custos.

A satisfação dos três critérios é relativa ao source. Limites físicos persistem:
por exemplo, p90 por canal de single-mode Configuration 11 ainda é **1,169678**.
Nenhuma tolerância de aplicação permite classificá-lo como aceitável/final.

## Fronteira de compromisso, não um vencedor universal

O registro calcula dominância componente a componente em **18 métricas de grupo**:
RMSE/correlação/p90 nas seis configurações, epsilon numérico 1e-9. Não permite que
ganho macro esconda perda em um componente. Checkpoints indisponíveis não dominam
candidatos utilizáveis; métricas históricas continuam registradas.

Nos candidatos considerados, ambos os sources são dominados por ao menos uma
alternativa na sua coorte. Os demais mantêm algum trade-off e permanecem na
fronteira desses 18 critérios. Isso não significa equivalência física nem que
todos devam ser promovidos: a fronteira não incorpora todos os diagnósticos,
tolerâncias ou custo de treinamento.

Amplitude/fase ficam visíveis, mas **não fazem parte da guarda de três critérios**.
Fold 0 auxiliar tem RMS macro 0,896346, pior que 0,901455 do source, apesar de
passar pela guarda. Filtros de aceitação finais podem ser mais amplos e devem
ser definidos antes de aplicá-los como critério de promoção.

## Decisão operacional e requisitos para avançar

**Nenhum novo treino longo foi iniciado após a confirmação negativa.** Não
continuar épocas, variar pesos ou relaxar a guarda indefinidamente sobre as
mesmas configurações. Defaults permanecem inalterados; nenhum modelo final
promovido, nenhum artefato ou raw apagado.

Antes de uma decisão de modelo final, fixar requisitos que ainda faltam:

| Requisito | Decisão necessária |
|---|---|
| Fidelidade física | Limites de erro em unidades raw, RMSE relativo, amplitude/fase/bandas relevantes por configuração. |
| Preservação de canais | Limites para caudas e canais fracos, além de médias de grupo; não inferir tolerância de epsilon numérico. |
| Bitrate | Orçamento aceitável com container/modelo/side e implementação real; taxas atuais são nominais de índices+side. |
| Generalização | Aquisições independentes não usadas para motivar/selecionar estes estudos; separar por aquisição/configuração antes do próximo treino. |
| Protocolo final | Baseline, hipóteses, candidatos, critérios de admissão e budget pré-fixados, sem substituir controle depois de ver resultados. |

Isso não impede novos estudos com hipótese isolada, mas não há evidência para
promover a nova loss ou justificar outra extensão automática desta mesma trilha.
A prioridade é reduzir incerteza de aceitação/independência, não acumular mais
compute para ganhos macro pequenos.

## Reprodução e testes

```bash
.venv/bin/python finetuning/summarize_generalist_frontier.py \
  --output-dir results/experiments/generalist_4x_registry_NEW
```

Saída nova obrigatória; fontes somente leitura. **189 testes passaram**,
incluindo direção das métricas e dominância sem ocultar regressão individual.
`git diff --check` sem erros. Não houve commits ou pushes.
