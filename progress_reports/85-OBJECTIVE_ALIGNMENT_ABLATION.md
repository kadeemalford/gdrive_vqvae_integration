# Redirecionamento: objetivo de reconstrução, com controle histórico pareado

**Protocolo registrado em 12/11/2026, antes dos resultados novos.** Continuidade
dos negativos dos [relatórios 81](81-LR_SCHEDULE_PAIRED_RESULTS.md)–
[84](84-FIXED_ID_SYNTHESIS_AND_CONTEXT_PROBES.md). Sem promoção.

## 1. Decisão de direção

Não continuar automaticamente com dados externos, tabelas de síntese ou
adaptadores congelados: os contrastes delimitados não atingiram investimento.
Isso não demonstra que as famílias estejam esgotadas. Antes de outra arquitetura
longa, isolar uma escolha de treinamento que permaneceu constante: a combinação
de erro quadrático e absoluto, apesar de a decisão principal privilegiar RMSE.

A mistura histórica é válida, mas **não é o mesmo objetivo que MSE**. O efeito
prático de removê-la não está isolado pelos contrastes espectrais ou de canais
limitados anteriores. MSE favorece médias condicionais; não recupera informação
ausente nem a realização exata de ruído imprevisível. Não prometer que eliminar
L1 recuperará amplitude, fase ou fidelidade geral.

## 2. Contraste congelado

| Braço | Objetivo | Updates adicionais | Estado inicial |
|---|---|---:|---|
| Controle histórico | 0,5 MSE + 0,5 L1 + VQ | 16.384 | A compacto, época 24, AdamW em 98.304 |
| Novo MSE | MSE + VQ | 16.384 | Exatamente os mesmos pesos/momentos/contadores/RNG |

- Controle: `lr_schedule_fold0_v2_20261109/constant_high`, execução corrigida,
  concluída e auditada, **não** a execução rejeitada por alias de contadores.
- Mesma sequência congelada de batches, oito blocos de 2.048; mesmos 11.760
  exemplos elegíveis de treino, grade `spatial-4x`, K2048, corpus/escapes/normalização.
- Mesmo AdamW herdado sem reset, LR constante `1e-4`, weight decay/betas/epsilon
  herdados, clip1, FP32/TF32 desligado. Toda a rede é treinável.
- VQ não muda de coeficiente. Portanto, muda também sua escala relativa frente
  à reconstrução: este é um teste de **objetivo prático**, não uma separação
  causal entre média/mediana mantendo todas as normas de gradiente idênticas.
  MSE normalizada por janela tampouco é exatamente o RMSE macro físico.
- Fim esperado do optimizer em **114.688**, com origem em 98.304. Fonte em memória
  e arquivo imutáveis. Retomada somente em fronteiras completas de bloco.

## 3. Reutilização sem treinamento redundante

A referência só é admissível se baterem hashes de fonte, código, vendors,
inputs, sequências e checkpoints, contadores e proveniência de optimizer.
Também serão congelados resultados por grupo, bundles e pesos de implantação.
A GPU física e a versão PyTorch precisam coincidir com o controle.

O score inicial nas mesmas 130 janelas históricas deve reproduzir o controle
antes de começar o braço completo. Essa checagem não garante equivalência de
todo ambiente numérico ou independência estatística, mas evita uma comparação
com origem silenciosamente diferente. Treinar apenas o challenger economiza
16.384 updates, sem contá-los como avanço científico novo.

## 4. Seleção, avaliação e limites

- Inicialização elegível; seleção por menor RMSE relativo macro na mesma triagem
  histórica. Preservar endpoint separadamente, sem seleção usando teste.
- Depois do treino, avaliar selecionado/endpoint nas **7.220 janelas de validação**,
  com FAK1/FSB1, mesmo replay discreto e reader independente. Calibração de
  probabilidades somente nas mesmas 32 janelas de treino usadas no controle.
- Reutilizar streams/grupos do controle apenas após verificar hashes e identidade
  de inputs/metadata/calibração/implementação. Pesos compartilhados discriminados;
  mesmo número nominal de índices não garante os mesmos bytes reais.
- **Sem extensão automática de treino.** Só considerar confirmação posterior se
  houver **≥5% menos RMSE físico macro** que o controle misto pareado, bytes
  completos não maiores e nenhuma regressão nas guardas predefinidas por
  configuração: RMSE, correlação, p90, FFT magnitude/fase e distância RMS a 1.
- Não retirar canais quase constantes nem relaxar guardas se aparecerem caudas
  patológicas. Registrar negativos e benefícios parciais sem chamar de promoção.

O desenvolvimento já foi reutilizado muitas vezes. Um positivo seria evidência
exploratória, ainda dependente de confirmação em outra coorte e requisitos
físicos operacionais. Não é um novo teste externo independente.

## 5. Execução prevista

Somente uma RTX 4090 remota, GPU física 0 de `aumo-gpu-001`, autenticação por
`SSH_PASSWORD` sem registrar seu valor. Testes CPU locais não treinam modelos.
Primeiro smoke de dois updates, com maior tile; depois o braço completo e a
avaliação, sequencialmente. Diretórios próprios `objective_ablation_*_20261112`.

Implementação: `finetuning/objective_ablation.py`,
`finetuning/run_objective_ablation.py`, `finetuning/evaluate_objective_ablation.py`
e `tests/test_objective_ablation.py`. Nenhum resultado dessa nova ablação foi
observado ao registrar este protocolo.

## 6. Resultado concluído e auditado

Campanha `results/experiments/objective_ablation_fold0_20261112/`, precedida
por smoke e replay discreto. Treino completo: **16.384 updates / ~727,1 s** de
updates, fora avaliação/I/O, numa única RTX 4090. O estado final do AdamW está
em **114.688**; hashes do código/fonte/checkpoints e linhagem foram conferidos.
O score inicial reproduziu a triagem histórica antes do treino completo.

| Candidato | RMSE macro ↓ | Correlação ↑ | Bytes completos |
|---|---:|---:|---:|
| Origem congelada | 0,667586 | 0,727657 | 15.658.668 |
| Controle misto, dose pareada | 0,661818 | 0,731712 | **15.646.620** |
| MSE + VQ | **0,660039** | **0,733764** | 15.776.804 |

MSE melhora **0,2687% contra controle**, não os 1,1305% contra a origem sem
esses updates. Transmite **130.184 bytes / 0,8320% a mais**. Selecionado e
endpoint coincidiram, ao bloco 8. Os custos compartilhados são discriminados
nos JSONs; não é um ganho a bitrate igual ou uma promoção.

Magnitude FFT macro melhorou `0,500832 → 0,496479`, fase
`0,839992 → 0,841429` e RMS ratio `0,764956 → 0,774133`. Porém, houve **14 vetos**:
cinco de p90, três de fase, dois de distância RMS a 1, dois de RMSE e dois de
correlação. RMSE piorou em single-mode 2/6. O p90 macro continua patológico
em canais de pequena variação; não foi removido ou relaxado para favorecer MSE.
Mesmo ignorá-lo não resolveria ganho insuficiente, bytes maiores e outros vetos.

No último bloco, as médias de treino foram MSE ~0,37915, L1 ~0,40187 e
VQ ~0,29578; ~0,44% dos batches atingiram clip de gradiente. Esses componentes
caracterizam a execução, não isolam o mecanismo causal da pequena melhora.

**Decisão:** encerrar sem extensão, confirmação ou adoção de MSE como default.
O objetivo histórico não explica principalmente o patamar observado neste
contraste. Isso não esgota outros objetivos nem estabelece convergência ótima.

O próximo diagnóstico não retreina essa CNN. Examina a distribuição condicional
dos coeficientes dos codecs escalares funcionais anteriores, com reconstrução
bit-exata e controles de probabilidade/framing comuns, antes de considerar o
hyperprior proposto no relatório 77. Não confundir o diagnóstico de probabilidades
com novo ganho de fidelidade ou autorização para repetir os treinos escalares.
