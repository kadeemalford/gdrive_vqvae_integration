# Modelo congelado: efeito localizado da proteção adaptativa

> Auditoria adicional do fold 1 concluída e cobertura combinada documentadas no
> [relatório 35](35-DYNAMIC_RANGE_COVERAGE_CONCLUSION.md).

## 1. Resultado concluído

Fonte: `results/experiments/frozen_range_policies_20261014/`, protocolo,
windows/groups/summary e `COMPLETE`. Runtime **6min21s**. Modelo Temporal fold 0
congelado, **zero updates**, mesmas 5.920 janelas de validação e uma referência
ordinária comum entre políticas; tails preservadas/cobradas separadamente.

**Confirmou-se melhora de processamento nas 40 janelas afetadas**, sem mudar
pesos ou alterar as 5.880 entradas restantes. Isso não é ganho de aprendizagem.

## 2. Mesma região ordinária nas 40 janelas

| Métrica | Escape absoluto 1e20 | Candidato adaptativo |
|---|---:|---:|
| Correlação | 0,014496 | **0,487711** |
| RMSE relativo | 2,394e10 | **0,884192** |
| p90 relativo por canal | 3,971e10 | **1,207171** |
| Erro raw fora da referência | ~9,017e15 | **0** |

P90/estatísticas são médias de valores por janela no grupo, não um quantil
pooled único. O baseline produz erros gigantes no sinal ordinário quando o
decoder restaura previsões usando a escala dominada pelas caudas remanescentes.
Antes, denominadores que incluíam essas caudas escondiam parte da falha.

Na candidata, células fora da referência são escapes exatos: nenhum valor
original foi removido/clippado. O custo extra é contabilizado. Os scores gigantes
não devem ser comparados diretamente com RMSE legado ~0,738: máscara/escala
de referência são diferentes. **Não concluir que o modelo aprendeu bilhões
de vezes melhor**, nem que todos os valores numéricos grandes sejam corrompidos.

## 3. População completa e invariância do restante

Para robust-candidate, referência ordinária comum completa:

- **RMSE macro 0,734098**, correlação **0,658977**.
- Pior família RMSE **0,843958**.
- Mediana relativa de canais **0,870442**, p90 **1,078875**.
- Compressão nominal pooled índices+side **163,144x**, contra **166,316x** do
  escape absoluto; diferença de **303.232 bytes**, já verificada na auditoria.

Nas **5.880 janelas não alteradas**, ambas as políticas produziram exatamente os
mesmos scores: RMSE macro **0,733380**, correlação **0,660136**, p90 **1,077037**.
O enorme score comum agregado do baseline é dominado pelas 40 janelas, não uma
falha generalizada em toda a população.

Mesmo após proteção, os 40 casos têm RMSE ~0,884 e o restante ~0,733: **a correção
numérica não resolve a reconstrução de alta fidelidade**. Preservar separação
entre pipeline incorreta em parte do corpus e limitações de modelo/taxa/transferência.

## 4. Próximo passo iniciado: verificar onde a política altera o treino

No fold 0, identidade auditada mostrou nenhuma entrada de treino alterada; portanto
repetir treino longo só para essa correção seria injustificado. Antes de integrar
ao pipeline geral, aplicar a mesma auditoria determinística ao **train/val fold 1**.
Nesse split, Configuration 6 tem papel diferente; não assumir invariância do treino.

Runner existente agora recebe `--fold`, default 0 preservado.
Saída: `results/experiments/remaining_dynamic_range_fold1_20261014/`.
Serviço: **`febus-dynamic-range-fold1-20261014`**. **Auditoria pendente nesta atualização**.
São 17.960 janelas de treino/validação, sem GPU, updates ou score de teste.

Se a política alterar apenas os casos numericamente problemáticos, definir uma
integração experimental versionada, com comparação comum e guards de policy no
resume. Adaptação de pesos, se necessária em split com treino alterado, deve ser
pareada e separada da avaliação congelada. Não promover globalmente sem nova
cobertura de configurações/aquisições e requisitos físicos.

Nenhum novo treino iniciou nesta atualização. O framework principal continua
usando as políticas anteriores; candidata robusta ainda é experimental isolada.
