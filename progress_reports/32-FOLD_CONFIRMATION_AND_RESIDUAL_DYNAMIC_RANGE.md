# Confirmação entre folds e faixa dinâmica residual

> A auditoria de identidade foi concluída. Resultados/custos e comparação de
> modelo congelado no [relatório 33](33-DYNAMIC_RANGE_AUDIT_AND_FROZEN_COMPARISON.md).

## 1. Confirmação concluída: efeito de geometria não é universal

Fonte: `results/experiments/rate_allocation_fold0_seed2027/`, `COMPLETE`,
histories e `full_validation/summary.csv`/tabelas. Três braços, 24 épocas cada;
validação completa nas mesmas **5.920 janelas** do fold 0, source específico
desse fold e seed de adaptação 2027. Nenhum score de teste.

| Grade | Epoch selecionado | RMSE macro | Correlação macro | Pior família RMSE | Erro canal não fraco |
|---|---:|---:|---:|---:|---:|
| Temporal | 24 | **0,738526** | **0,654386** | **0,861669** | 0,886936 |
| Balanced grid | 24 | 0,752330 | 0,639786 | 0,882873 | 0,861061 |
| Spatial | 23 | 0,755173 | 0,633108 | 0,866151 | **0,846948** |

O ganho macro espacial do fold 1 **não se confirmou**. Source/qualidade de warm-start
e configurações diferem entre folds; isso não é atribuição causal exclusiva a
domínio ou stride. Balanced/Spatial preservam trade-offs favoráveis de canais,
mas nenhuma geometria deve ser promovida como solução generalista única.

Temporal/Balanced ainda selecionaram a última época, mas antes de mais exposição
é necessário esclarecer os valores enormes nas métricas de canal desta população.

## 2. Anomalia localizada: exatamente 40 janelas

A inspeção das tabelas por janela encontrou **40/5.920 janelas com p90 > 10**,
todas também com p90 > 1.000, no multimode Configuration 6. Nas demais 5.880,
essas caudas gigantes não aparecem. Correspondem às janelas numéricas problemáticas
já auditadas, não a descoberta de arquivos corrompidos novos.

Em uma das janelas mais afetadas, após retirar abs>1e20 do cálculo:

- Máximo ainda ordinário ~**7,98e16**.
- Média de normalização ~**3,89e9**, escala ~**9,24e14**.
- **510 canais** classificados como fracos relativamente a essa escala enorme.
- P90 do modelo ~**6,84e10**.
- P90 de roundtrip encode→decode **sem modelo** ~**0,0363**.

Portanto escape-v1 protegeu os extremos mais altos, mas não eliminou toda a
desproporção de escalas. Há erro mensurável do próprio roundtrip, e a rede pode
injetar variação de escala enorme nos canais que têm amplitude ordinária pequena.
Classificá-los como "fracos" relativamente ao global não significa que sejam
fisicamente constantes ou inválidos. A origem física/export dos valores não está
confirmada; preservar os valores completos continua obrigatório.

## 3. Diagnóstico de cobertura, sem exclusão do score principal

Reagregação exploratória das **5.880 janelas sem escapes originais**:

| Grade | RMSE macro diagnóstico | Correlação | p90 canal macro |
|---|---:|---:|---:|
| Temporal | **0,733380** | **0,660136** | 1,077037 |
| Balanced grid | 0,747264 | 0,646047 | 1,028855 |
| Spatial | 0,752448 | 0,637727 | **1,018090** |

Isso **não substitui** os scores completos nem declara as 40 janelas inválidas.
Mostra que a reversão do ranking de geometria não é explicada apenas por esses
casos, enquanto médias p90 gigantes são dominadas por eles. Não remover os
problemas para melhorar artificialmente a aceitação generalista.

## 4. Próximo passo iniciado: auditoria de proteção adaptativa

Priorizar faixa dinâmica antes de outro treino longo. Comparar encode/decode
identidade de escape-v1 com candidato adaptativo, pontuando ambas as políticas
na **mesma máscara ordinária de referência**, preservando e cobrando todos os
escapes bit-exatos. Não treinar/selecionar modelo usando melhora por denominação
alterada ou pelos extremos agora reconstruídos exatamente.

Auditoria CPU no fold 0 train/val: `finetuning/audit_remaining_dynamic_range.py`.
Política isolada em `finetuning/robust_escape.py`: cap 1e20, p99 absoluto x 1.000,
com fallback positivo para dados esparsos. É hipótese numérica versionada, não
threshold de validade física. Nenhum loader/trainer passou a usar a política
experimental nesta atualização.

Saída: `results/experiments/remaining_dynamic_range_audit_20261014/`.
Serviço: **`febus-dynamic-range-audit-20261014`**. **Resultado completo pendente**.
15.640 janelas de treino/validação, sem arrays de teste, GPU ou updates.

## 5. Critérios antes de integrar ou treinar

1. Conferir redução de erro de roundtrip e manutenção da variação ordinária.
2. Medir quais janelas/células mudam, inclusive no treino saudável e sinais esparsos.
3. Cobrar novos escapes e bitrate nominal; exatidão não é informação gratuita.
4. Se confirmar, avaliar modelos congelados em máscara comum e regiões separadas
   antes de adaptar pesos. Não concluir ganho neural a partir de identity roundtrip.
5. Só então definir piloto de adaptação pareado; mudanças no codec/normalização
   exigem versão, metadata, guards e testes de restaurabilidade.

**148 testes passaram**, incluindo não mutação, escapes bit-exatos, proteção de
sinal esparso, constantes e rejeição de payload inválido. Dados brutos permanecem
intactos. Sem novo treino iniciado: a auditoria é pré-requisito para evitar otimizar
com uma representação numericamente inadequada em parte da população.
