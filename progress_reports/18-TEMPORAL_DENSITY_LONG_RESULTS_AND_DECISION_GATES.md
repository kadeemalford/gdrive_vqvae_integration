# Densidade temporal: resultados longos e próximos critérios de decisão

> Auditoria completa e controle DCT concluídos: resultados no
> [relatório 19](19-TEMPORAL_AUDIT_AND_MATCHED_DCT_RESULTS.md).

## 1. Conclusão

Os dois braços completaram oito épocas adicionais no corpus inteiro, com validação
completa por época. Ambos selecionaram a época 8. **4x é a linha neural prioritária
para investigação**, não um modelo final: melhorou RMSE e correlação nas seis
configurações frente à própria inicialização. O ganho de RMSE macro foi ~3,63%
relativo e ~6,67% frente ao standard do piloto.

2x tem ganho menor e uma regressão de configuração oculta pela média de família.
Nos dois braços, a amplitude melhora, mas o alinhamento de fase FFT ponderado
declina. Portanto, não estender automaticamente o treino nem declarar fidelidade
suficiente com base apenas no seletor RMSE.

## 2. Evidência e população

- Fonte: `results/experiments/temporal_density_long_20261008/`, marcador `COMPLETE`,
  `protocol.json`, históricos e summaries dos dois braços.
- Inicialização: best_macro do piloto por geometria; AdamW novo, LR 3e-6,
  normalização global escape-v1, ajuste completo e sampling uniforme.
- Treino: 12.140 janelas/época; oito épocas por braço, sem parada antecipada.
- Validação: 5.820 janelas, quatro famílias/seis configurações do fold 1.
- Comparação gerada em `comparison.json` e `REPORT.md`; inicial/selecionado/final
  mantidos. Epochs são relativos à continuação, não ao piloto precedente.
- Nenhum teste consultado. Mesmos grupos de desenvolvimento usados em decisões
  anteriores: não é evidência independente de generalização.

## 3. Resultado global e limites

| Braço | RMSE inicial → final | Correlação inicial → final | Pior família RMSE inicial → final |
|---|---:|---:|---:|
| 2x | 0,856491 → 0,848946 | 0,471902 → 0,490283 | 0,961211 → 0,952566 |
| 4x | **0,840263 → 0,809797** | **0,502038 → 0,559971** | **0,951289 → 0,920879** |

| Braço | RMS ratio inicial → final | Alinhamento fase FFT inicial → final |
|---|---:|---:|
| 2x | 0,465033 → 0,489055 | 0,812256 → 0,801028 |
| 4x | 0,499811 → 0,575644 | 0,820360 → 0,796833 |

Multimode ainda é a pior família. Amplitude continua atenuada. Alinhamento FFT é
ponderado pelo produto das magnitudes original/reconstruída: recuperar componentes
adicionais pode mudar os pesos. A queda pede diagnóstico por grupo/banda, mas não
prova que todo canal perdeu fase; tampouco permite ignorar o trade-off.

As curvas de RMSE ainda melhoram na última época: 4x cai de 0,811855 para 0,809797
e 2x de 0,849434 para 0,848946. **Não há demonstração de saturação**. Contudo, a
curva escalar favorável não é suficiente para autorizar outra longa continuação
sem examinar custos e métricas concorrentes.

## 4. Configurações: não esconder regressões

| Família/configuração | 2x inicial | 2x final | 4x inicial | 4x final |
|---|---:|---:|---:|---:|
| Doped / 5m SMF | 0,808211 | 0,799444 | 0,791866 | 0,761650 |
| Multicore / C1 and C4 | 0,909930 | 0,899188 | 0,890618 | 0,852390 |
| Multimode / Configuration 7 | 0,961211 | 0,952566 | 0,951289 | 0,920879 |
| Single-mode / Configuration 11 | 0,707206 | 0,699927 | 0,703355 | 0,682060 |
| Single-mode / Configuration 3 | 0,786004 | **0,787702** | 0,736593 | 0,723748 |
| Single-mode / Configuration 7 | 0,746619 | 0,746130 | 0,741888 | 0,706999 |

2x regride ~0,001698 absoluto em Configuration 3. 4x melhora todas as configurações
em RMSE e correlação e mantém vantagem sobre 2x em todas elas. Isso não garante
vantagem em cada canal, nem equivalência de taxas.

## 5. Taxa e canais: auditoria antes de nova otimização

A razão nominal média por configuração/família do treino é **321,72x / 164,66x**
para 2x/4x, respectivamente. Esses valores **não são pooled**. A geometria e side
stream permanecem iguais aos dos respectivos pilotos; a auditoria recalcula os
totais e razões pooled a partir de cada janela, sem fazer média de razões.

O erro relativo médio por canal permanece enorme e sensível a canais quase
constantes (~15.918 / ~22.786); não usar essa média isolada para classificar 4x
como pior ou declarar canais fiéis. Mediana/p90, não fracos, RMSE raw e fase por
grupo complementam o RMSE global.

**Próxima execução iniciada:**

1. Reavaliar standard selecionado e os dois candidatos longos em todas as 5.820
   janelas, sem updates, com métricas robustas por canal e contagem exata de índices.
   Saída: `results/experiments/temporal_density_long_audit_20261010/`.
2. Após essa avaliação, comparar DCT com cada rede em **102 janelas idênticas**,
   duas por arquivo de validação, seed 2026. O teto de bytes do DCT é calculado
   individualmente a partir dos índices reais da geometria + side de cada candidato,
   não por um multiplicador aproximado do standard. Saída:
   `results/experiments/temporal_density_matched_dct_20261010/`.

Resultados dessas duas tarefas **ainda pendentes nesta atualização**. O DCT tem
stream real com header/coeficientes/normalização/escapes; a taxa neural é nominal
índices+side. Pareamento de teto e janelas não equivale a codecs completos com
taxas efetivas idênticas. Não comparar a amostra DCT com a média neural completa.

## 6. Decisões seguintes, condicionais

- **Ganho consistente com canais e fase aceitáveis:** repetir 4x com outra seed
  e confirmar transferência por configuração/fold; não pular direto para produção.
- **Ganho de RMSE com regressões relevantes:** manter candidatos não dominados,
  diagnosticar por banda/canal e ensaiar uma alteração controlada de objetivo ou
  representação. Não repetir automaticamente a loss STFT de magnitude já rejeitada.
- **DCT competitivo ao mesmo teto:** investigar residual/híbrido ou quantização,
  cobrando seleção, residual e metadados; nenhum roteamento oracle gratuito.
- **Limite de dados identificado:** ampliar aquisições/configurações independentes,
  não simplesmente duplicar janelas. Novo holdout externo antes de promoção final.

Critérios físicos/limiares de aceitação ainda precisam ser definidos para o uso
pretendido. Sem eles, as decisões acima são pesquisa, não certificação de fidelidade.

## 7. Reprodução e operação

```bash
.venv/bin/python finetuning/summarize_numeric_adaptation.py \
  --experiment-dir results/experiments/temporal_density_long_20261008

.venv/bin/python finetuning/run_temporal_density_audit.py \
  --evaluation-dir results/experiments/temporal_density_long_audit_20261010 \
  --control-dir results/experiments/temporal_density_matched_dct_20261010
```

A auditoria foi lançada como serviço de usuário systemd
`febus-temporal-audit-20261010`, independente do processo interativo do OpenCode.
Não executar outra cópia enquanto esse serviço estiver ativo. Serviço de usuário
não implica sobrevivência garantida a logout/reboot; o runner retoma candidatos
já completos, preservando tentativas parciais. O benchmark CPU incompleto exige
novo diretório de saída, sem sobrescrever evidência.

Log dos subprocessos:
`results/experiments/temporal_density_long_audit_20261010.pipeline.log`.

```bash
systemctl --user status febus-temporal-audit-20261010
```

**114 testes passaram**, incluindo detecção de regressão de configuração escondida
pela média de família, além do smoke de codec/métricas do controle pareado.
GPU e driver estão funcionais. Nenhum checkpoint ou dado bruto foi alterado.
