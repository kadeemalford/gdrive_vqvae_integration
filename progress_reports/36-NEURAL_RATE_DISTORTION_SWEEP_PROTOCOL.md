# Curva neural de taxa-fidelidade com grade espacial fixa

> Sweep e avaliação completa concluídos; resultados e controle DCT pareado no
> [relatório 37](37-NEURAL_RATE_DISTORTION_RESULTS.md).

## 1. Hipótese e controles

Depois dos trade-offs entre geometrias/folds e da correção numérica localizada,
testar se mais orçamento de índices aproxima a reconstrução de fidelidade alta.
Preservar a densidade espacial da melhor candidata macro no fold 1, mudando
somente os últimos strides temporais e seus correspondentes no decoder.

Três braços partem das mesmas weights Spatial fold 1 epoch 23, com **AdamW novo**
em todos, LR 3e-5. Source original fica intacto e é referência congelada ao final.
Mesmo corpus/ordem de treino, objetivo temporal+VQ, codebook K=2.048 e
**27.958.081 parâmetros**; sem refresh, especialistas ou novo dataset.

## 2. Taxas diferentes, explicitamente cobradas

| Braço | Geometria | Grade para 512x1.000 | Índices | Ratio de índices vs control |
|---|---|---:|---:|---:|
| rate_1x / control | spatial-4x | 128x63 | 8.064 | 1,000 |
| rate_2x | spatial4-temporal2 | 128x125 | 16.000 | **1,984127** |
| rate_4x | spatial4-temporal4 | 128x250 | 32.000 | **3,968254** |

Rótulos 2x/4x são aproximações. Cada índice custa 11 bits nominais; side/padding
serão recalculados por janela. A compressão nominal esperada cai de ~165x para
~84x/~42x, não ganha gratuitamente. Conclusão deve ser de **curva rate-distortion**,
não superioridade à mesma taxa. Strides também mudam receptive field/compute;
mais bits não demonstram um limite de informação ou arquitetura ótima.

Padding espacial compensatório é o mesmo nas três grades; alvo completo permanece
intacto e decoder recorta somente padding. Não há downsampling externo, filtragem,
clipping ou remoção de canais do alvo. Source e política escape-v1 continuam iguais:
auditoria mostrou nenhuma mudança de input train/val pela política robusta nesse fold.

## 3. Orçamento e avaliação

- **24 x 4.096 = 98.304 updates adicionais por braço**, 294.912 totais.
- Seed 2026, sampling window, weight decay 1e-5, grad clip 1.
- Patience 25: orçamento mínimo de 24 épocas sem parada precoce.
- Seleção nas mesmas 102 janelas por RMSE macro, incluindo epoch 0.
  Geometrias distintas podem ter inicializações de score diferentes.
- Avaliar source congelado e os selecionados nas **5.820 janelas completas**.
- Medir RMSE/correlação, amplitude/fase, cauda por canal, pior família/configuração,
  taxa pooled e tempos. Não promover um candidato só porque transmite mais bits.
- Nenhuma avaliação de teste ou auditoria externa; protocolo permanece exploratório.

Se ganhos de densidade forem pequenos, reconsiderar capacidade, contexto e
diversidade; se forem fortes, confirmar curvas em configurações adicionais e
confrontar controles clássicos nos respectivos tetos. Ainda faltam tolerâncias
físicas e bitrate aceitável definidos pela aplicação para uma seleção final.

## 4. Testes e memória

**150 testes passaram**, com forwards/índices reais, contagens em canais variados,
roundtrip de metadata/padding e compatibilidade histórica.

Smoke GPU em 512x1.000 com três updates **descartados**, nenhum checkpoint salvo:

| Grade | Pico alocado GPU | Índices reais |
|---|---:|---:|
| Control | ~4,00 GB | 8.064 |
| Temporal 2x | ~4,31 GB | 16.000 |
| Temporal 4x | ~5,37 GB | 32.000 |

Shapes/gradientes/loss finitos; mesmo número de parâmetros. Evidência:
`results/experiments/neural_rate_sweep_smoke_20261014/`.

## 5. Execução e recuperação

Runner: `finetuning/run_neural_rate_sweep.py`.
Saída: `results/experiments/neural_rate_sweep_20261014/`.
Log: `results/experiments/neural_rate_sweep_20261014.pipeline.log`.
Serviço **`febus-neural-rate-sweep-20261014`**, sequencial na GPU.
**Resultados pendentes nesta atualização**; estimativa inicial **9–14 h**,
sujeita ao maior compute/variação de shapes. Atualizar com épocas medidas.

```bash
systemctl --user status febus-neural-rate-sweep-20261014

# Somente sem cópia ativa:
.venv/bin/python finetuning/run_neural_rate_sweep.py \
  --output-dir results/experiments/neural_rate_sweep_20261014 --resume
```

Source/manifest/código/smoke/auditoria são fixados por hashes; locks, saves atômicos
e retomada de epoch completo. Resume restaura optimizer/RNG sem ampliar orçamento.
Serviço não garante sobrevivência a logout/reboot; originais e referências não
são sobrescritos. Nenhum download ou modelo final promovido.
