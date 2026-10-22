# Cobertura da proteção adaptativa e conclusão sobre retreinamento

## 1. Auditoria fold 1 concluída

Fonte: `results/experiments/remaining_dynamic_range_fold1_20261014/`, protocolo,
summary/groups/windows e `COMPLETE`. Runtime **8min55s**; 12.140 janelas de
treino e 5.820 de validação, sem modelo, updates ou score de teste.

**Zero janelas/células adicionais escapadas** em ambos os splits. Máscaras,
normalização/roundtrip e tamanho de side stream foram iguais às da política
escape-v1. A fórmula de threshold transmitido pode diferir, mas inputs numéricos
e bytes totais não mudam na ausência de novos escapes.

| População | RMSE identidade macro, ambas | Side bytes, ambas |
|---|---:|---:|
| Treino fold 1 | 2,141e-8 | 485.600 |
| Validação fold 1 | 2,296e-8 | 232.800 |

Configuration 6 não está no treino/validação do fold 1; seu papel é teste nesse
split e nenhum score de modelo desse teste foi consultado nesta auditoria.

## 2. Cobertura canônica combinada

Deduplicando train/val dos folds 0 e 1 por `drive_id`, bloco, início temporal,
canais e tamanho temporal, as duas auditorias cobrem **23.880 janelas canônicas**.
São dados preparados já estudados, não novas aquisições independentes. A união
passa por coordenadas com outros papéis em outros folds, mas esta operação é
integridade/roundtrip, não avaliação de modelo no teste desses folds.

Nas duas auditorias, a alteração ficou localizada nas 40 janelas multimode do
fold 0 val. Dados brutos fora dos crops preparados e novas aquisições ainda
exigem auditoria; a origem física/export dos valores grandes não foi confirmada.

## 3. Decisão

**Não retreinar os modelos atuais só para essa proteção**, pois inputs dos dois
treinos auditados não mudaram. O modelo congelado do relatório 34 comprovou
melhora localizada quando a entrada problemática muda, preservando os demais
outputs e os tails exatamente com custo adicional.

Manter proteção adaptativa como **política experimental validada nessa cobertura**,
não default de produção sem integração versionada, guards, identificação de modelo
e audit externa. Ela não explica nem resolve o RMSE ~0,7 no restante da população.

O próximo teste volta ao gargalo de representação/taxa: aumentar posições latentes
na grade espacial e medir uma curva rate-distortion com preço explícito. Não
atribuir ganho neural à correção localizada ou anunciar a mesma taxa em grades
mais densas. Protocolo no [relatório 36](36-NEURAL_RATE_DISTORTION_SWEEP_PROTOCOL.md).
