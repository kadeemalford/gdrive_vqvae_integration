# Normalização treinada: resultados do piloto e validação ampliada

> Resultados da validação completa e a rota de controle rate-distortion estão no
> [relatório 14](14-FULL_NORMALIZATION_VALIDATION_AND_RATE_CONTROL.md).

## 1. Decisão com base no piloto

Experimento concluído: `results/experiments/channel_representation_pilot_20261008/`.
**Não promover a normalização por canal como padrão generalista.** Ela melhora
relativamente à própria inicialização deslocada, mas perde para o controle global
no RMSE macro do mesmo alvo restaurado e custa mais bits de side information.

Antes de outro treino, verificar os candidatos na validação completa e distinguir
erro em canais de baixa variância de melhora consistente da forma de onda. Não
prolongar o braço channel somente porque sua loss de treino continua caindo.

## 2. Protocolo e integridade

- Fonte comum: full epoch 6 do estudo de preservação, com fresh AdamW em ambos.
- Fold 1, LR 1e-5, weight decay 1e-5, seed 2026, todos os parâmetros ajustáveis.
- **6×2.048 updates por braço**, 12.288 cada; epochs 1–6 sem lacunas nos CSVs.
- Mesmo conjunto de **102 janelas de validação, 51 arquivos, quatro famílias**.
- Global escape-v1 versus channel-escape-v1; escapes preservados e avaliação física
  comum após restauração. A política channel muda input e ponderação efetiva do treino.
- Inicialização como candidata; nenhuma avaliação de teste ou promoção automática.
- Tempos de epochs completos: **16,40 min global / 15,98 min channel**, excluindo
  custos não incluídos nessa medida, como parte da inicialização e dos saves.

## 3. Resultado no alvo comum

| Braço | RMSE macro inicial | Melhor RMSE macro | Epoch | Correlação macro selecionada | Pior família RMSE |
|---|---:|---:|---:|---:|---:|
| Global | 0,869262 | **0,867924** | 6 | 0,448475 | 0,964664 |
| Channel | 0,932653 | **0,889123** | 4 | 0,396314 | 0,976736 |

Channel reduz bastante o erro da sua própria entrada nova, mas continua **~2,44%
pior que global** no RMSE macro. O mesmo ponto de partida em pesos não implica
mesma qualidade inicial porque as representações de entrada diferem.

| Família | Global selecionado | Channel selecionado | Channel − global |
|---|---:|---:|---:|
| Doped | 0,832464 | 0,901371 | +0,068907 |
| Multicore | 0,924606 | 0,938404 | +0,013798 |
| Multimode | 0,964664 | 0,976736 | +0,012072 |
| Single-mode | 0,749964 | 0,739981 | −0,009983 |

A melhora de single-mode não é uniforme: Configuration 3 e 7 ganham RMSE,
enquanto Configuration 11 perde ligeiramente:

| Configuração single-mode | Global | Channel |
|---|---:|---:|
| 3 | 0,796119 | 0,772494 |
| 7 | 0,750059 | 0,742699 |
| 11 | 0,703713 | 0,704750 |

No total, channel vence em **2/6 configurações**. Famílias sem regressão contra a
própria inicialização channel não significam famílias melhores que o baseline global.
O comparador automático mostra a primeira comparação; a tabela acima explicita a segunda.

## 4. Erro por canal e rate-distortion

| Métrica selecionada, agregação macro | Global | Channel |
|---|---:|---:|
| Side bits por célula | 0,004979 | 0,066988 |
| Média das razões nominais índices+side | 611,26× | 278,79× |
| RMSE relativo médio por canal | 14.239,17 | 0,87621 |

O número enorme do global é sensível a canais quase constantes; não substitui o
RMSE raw/global nem demonstra que a maioria dos canais tem esse erro. A auditoria
do relatório 11 já documentou esse problema de denominador. Uma representação pode
preservar melhor a escala de canais fracos sem reconstruir melhor os dominantes.

As razões acima são médias macro de razões por janela, **não razão total do corpus**.
Incluem índices fixos e parâmetros/escapes explícitos, não container, mapas espaciais,
pesos dos modelos, roteamento ou codificação entrópica. Não alegar taxa de codec
completo ou comparar somente o score por canal ignorando o aumento de side bytes.

## 5. Próxima etapa iniciada: validação completa dos candidatos

Runner de avaliação: `finetuning/evaluate_physical_candidates.py`.
Saída: `results/experiments/channel_full_validation_20261008/`.
Log: `results/experiments/channel_full_validation_20261008.log`.

Avaliação em background, **sem treinamento**, das mesmas **5.820 janelas de validação**:

1. Full best_macro epoch 6 do estudo de preservação, normalização global.
2. Global best_macro epoch 6 do piloto de representação.
3. Channel best_macro epoch 4 do piloto de representação.

Mesmo hash de validação do estudo longo; SHA-256 de cada checkpoint registrado no
protocolo. Não acessar partição de teste nem alterar pesos/arquivos de origem.
Os resultados completos ainda não integram as tabelas de piloto acima.

### Métricas adicionais

- Erro raw e métricas globais no mesmo alvo após restauração.
- Mediana e p90 dos erros relativos por canal, além da média histórica.
- Diagnóstico de canais de pequena variância: std temporal <
  `max(1e-6, RMS ordinário centrado da janela × 1e-6)`.
- Score adicional dos canais fora desse grupo diagnóstico, sem removê-los do
  score principal. Esse limiar é heurístico de análise, **não validade física,
  critério de denoising, regra de exclusão ou nova tolerância de aceitação**.
- Totais de células, bits de índices e side bytes, permitindo razão nominal
  agregada correta por soma dos bits, não média de razões.
- CSV por janela e por configuração, para verificar distribuição/outliers.

Métricas robustas são opt-in: a extensão não muda as colunas históricas do treino
quando `detailed_channels=False`. A avaliação pode retomar candidatos concluídos;
tentativas parciais permanecem em CSVs separados, sem sobrescrita silenciosa.

```bash
.venv/bin/python finetuning/evaluate_physical_candidates.py \
  --output-dir results/experiments/channel_full_validation_20261008 --resume
```

Background depende do serviço OpenCode. Em interrupção, um candidato incompleto é
reavaliado e seus arquivos parciais preservados; não há epoch/optimizer a retomar.

## 6. Decisões condicionais após a expansão

- Se a regressão generalista de channel persistir, não escalar essa normalização
  uniformemente para todas as fibras. Investigar representação com contexto físico
  ou custo intermediário, em ablações controladas.
- Se o ganho single-mode for consistente por grupo, avaliar roteamento por metadados
  como hipótese exploratória, com fallback global e custos de dois modelos explícitos.
  Não escolher o modelo pelo erro contra o alvo de cada janela de teste.
- Se a discrepância vier principalmente dos denominadores pequenos, priorizar
  critérios físicos e distribuição por canal antes de decidir arquitetura/taxa.
- Confirmar qualquer candidato entre folds/seeds e em nova aquisição/configuração
  holdout antes de promover. A expansão da mesma validação usada na seleção não é
  auditoria externa independente.

Os estudos permanecem nas etapas de representação e diagnóstico do roadmap. Nenhum
novo treino longo foi iniciado nesta transição, nenhum modelo foi promovido e dados/
checkpoints foram preservados. Os testes incluem alvo comum, side costs, canais de
baixa variância e emissão das métricas por janela durante a validação.
