# Faixa dinâmica adaptativa: auditoria concluída e comparação congelada

> Comparação congelada concluída; resultados e próximo teste de cobertura no
> [relatório 34](34-FROZEN_DYNAMIC_RANGE_RESULTS.md).

## 1. Auditoria concluída

Fonte: `results/experiments/remaining_dynamic_range_audit_20261014/`, protocolo,
windows/groups/summary e `COMPLETE`. **15.640 janelas**, fold 0 train/val,
sem modelo, updates ou score de teste. Runtime **7min12s**.

Compare escape absoluto 1e20 com candidato adaptativo da mesma família de streams:
threshold limitado por p99 absoluto x 1.000, cap 1e20 e fallback positivo para
sinais esparsos. Não é classificação física de invalidade; nenhuma célula é
descartada do alvo completo. Ambas as identidades são pontuadas na máscara
ordinária comum do candidato robusto.

## 2. Resultado numérico e custos

| População/política | RMSE relativo identidade macro | p90 canal identidade macro | Side bytes | Células escapadas |
|---|---:|---:|---:|---:|
| Treino / escape-v1 | 2,295e-8 | 2,622e-8 | 388.800 | 0 |
| Treino / adaptativo | 2,295e-8 | 2,622e-8 | 388.800 | 0 |
| Val / escape-v1 | 4,363e-5 | 2,851e-4 | 699.656 | 57.857 |
| Val / adaptativo | **2,224e-8** | **2,558e-8** | **1.002.888** | **95.761** |

**Nenhuma das 9.720 janelas de treino mudou a máscara.** Na validação, somente
as **40 janelas problemáticas** receberam novos escapes: 37.904 células adicionais,
**303.232 bytes adicionais** (~296 KiB), correspondendo aos registros índice+valor
de oito bytes. Escapes foram restaurados bit-exatos em todas as identidades.

Na taxa nominal temporal-4x (sem container/modelo), compressão val pooled passa
de **166,316x para 163,144x**: melhora de proteção não é informação gratuita.
No treino, ratio e erro de identidade são iguais entre políticas.

O erro macro antigo parece pequeno porque 40 janelas são diluídas entre milhares.
Nas janelas afetadas, o diagnóstico anterior mostrou erros de roundtrip por canal
de alguns pontos percentuais e muito mais erro no modelo. Ganho de identidade
**não é ganho neural**, nem prova de que os valores extremos sejam corrompidos.

## 3. Próxima comparação iniciada: mesmas weights, mesma referência

Runner: `finetuning/evaluate_frozen_range_policies.py`.
Saída: `results/experiments/frozen_range_policies_20261014/`.
Serviço: **`febus-frozen-range-eval-20261014`**. **Resultados pendentes**.

Modelo: melhor Temporal do fold 0, congelado, sem optimizer/updates.
Avaliar as **5.920 janelas completas** sob entrada escape-v1 ou candidata robusta,
decodificando com o side stream correspondente. Quando entradas normalizadas
são idênticas, reutilizar o mesmo forward determinístico; não mudar pesos.

### Evitar denominadores artificialmente favoráveis

- Máscara, média e escala de referência **iguais entre políticas**, derivadas da
  região ordinária robusta. Esses scores não são diretamente comparáveis aos
  scores legados com outra máscara/escala.
- Pontuar canais/correlação/RMSE/espectro na mesma região modelada de referência.
  Spectral diagnostics usam o mesmo preenchimento zero nas regiões escapadas.
- Registrar separadamente erro raw fora da referência, fração ordinária, células
  escapadas e side bits. Alvo completo permanece reconstruído, com escapes cobrados.
- Mostrar scopes completo / 40 janelas alteradas / 5.880 não alteradas; um ganho
  localizado não demonstra solução do restante do corpus.
- Não integrar automaticamente o candidato ao trainer ou reclassificar as
  janelas como inválidas para facilitar a reconstrução.

## 4. Decisão posterior

Se o modelo congelado melhorar somente nas 40 janelas sem afetar o restante,
priorizar proteção numérica versionada, não outra adaptação longa: seu treino
nesse fold não mudou. Se houver regressão/custo injustificado, preservar baseline
e investigar outro limite de faixa/cobertura sem mascarar o resultado.

Política geral ainda precisa de auditoria ampliada em configurações/aquisições
fora desse split. O restante da reconstrução continua limitado por representação,
taxa e transferência; corrigir o caso extremo não estabelece fidelidade alta.

**149 testes passaram**, incluindo máscara/denominador comum, não mutação,
escapes bit-exatos e proteção de eventos esparsos. Nenhum novo treino iniciado.
