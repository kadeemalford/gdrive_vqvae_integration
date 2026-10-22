# Alocação de índices entre tempo e canais: teste controlado

> Experimento concluído. Resultados e decisão de confirmação no
> [relatório 30](30-RATE_ALLOCATION_RESULTS.md).

## 1. Hipótese

Testar o orçamento de representação, não novos dados, normalização ou MoE:
uma grade mais densa em canais pode preservar heterogeneidade espacial, à custa
de menos posições temporais. Nenhuma banda/canal do alvo original é removida.

Todos começam dos mesmos pesos native transferidos epoch 13. Manter esse anchor
estável evita herdar a regressão multimode do controle Balanced de sampling.
Esse **balanced de geometria não é o balanced de sampling** do relatório 28.

## 2. Três braços

| Braço | Geometria | Posições para 512x1.000 | Relação com taxa temporal |
|---|---|---:|---:|
| Temporal/control | temporal-4x: mais densidade temporal | 32 x 250 = **8.000** | 1,000 |
| Balanced grid | balanced-2x: 2x espacial + 2x temporal | 64 x 125 = **8.000** | **exatamente igual** |
| Spatial | spatial-4x: 4x espacial, temporal padrão | 128 x 63 = **8.064** | **1,008** |

Igualdade de posições Temporal/Balanced vale para todos os tamanhos suportados;
Spatial tem 0,8% a mais para T=1.000, mas essa diferença muda com duração. K=2.048,
11 bits por índice e **27.958.081 parâmetros** em todos. Índices reais, side e
razões pooled serão recalculados na validação, não estimados só por stride.

### Padding e rastreabilidade

Kernel 4 com stride 1/padding 1 reduz uma célula. Geometrias espaciais recebem
padding zero inferior compensatório antes do encoder: **oito linhas para quatro
níveis**, após o padding nativo. Decoder produz essas linhas adicionais e as
recorta, junto ao padding original, mantendo o alvo original completo.

- Nenhuma linha física é removida/interpolada nem o alvo reamostrado.
- Parametrização permanece intacta; strides, receptive fields e contexto de borda mudam.
- Hook de padding é versionado pela geometria e reinstalado pelo loader; não existe
  no state_dict. Aplicação repetida remove o hook anterior, evitando acumulação.
- Comparação não isola strides de efeitos de contexto/padding, nem mantém receptive
  field idêntico. Testa a hipótese completa de alocação codificável nesse modelo.
- Evaluadores históricos sem suporte devem rejeitar a geometria não padrão.

Implementação: `finetuning/latent_geometry.py` e factory versionada existente.

## 3. Treino e seleção

- Mesmo fold 1, alvo FEBUS/escape-v1 e sampling **window uniforme** nos três braços.
- AdamW **novo** em todos, LR 3e-5, weight decay 1e-5, grad clip 1.
- Objetivo 0,5 MSE + 0,5 L1 temporal + VQ; refresh desligado.
- **24 x 4.096 = 98.304 updates por braço**, 294.912 totais.
- Patience 25: orçamento mínimo fixo sem parada antes de 24 épocas.
- Seleção nas mesmas 102 janelas por RMSE macro, epoch 0 como candidata.
  Scores iniciais não precisam coincidir: geometria muda mesmo com pesos iguais.
- Avaliar anchor congelado e selecionados nas **5.820 janelas completas**.
- Medir pior grupo, mediana/p90/nonweak de canais, amplitude/fase e bitrate.
- Nenhum acesso ao teste, auditoria externa ou promoção automática.

Orçamento suficiente para triagem não demonstra convergência. Uma variante que
melhore apenas uma família ou ainda esteja em melhora pede análise posterior,
não adoção automática. Ampliar taxa total permanece outra hipótese, isolada desta.

## 4. Testes e smoke

**142 testes passaram**: shapes/índices reais, igualdade de taxa em 23/45/70/512
canais, alternância de geometrias sem acumular hooks, pesos invariantes e
roundtrip de checkpoint restaurando padding/strides sem intervenção manual.

Smoke GPU em 512x1.000, três updates descartados, nenhum modelo salvo:

| Geometria | Pico alocado GPU | Índices reais |
|---|---:|---:|
| Temporal | ~3,96 GB | 8.000 |
| Balanced grid | ~3,77 GB | 8.000 |
| Spatial | ~4,00 GB | 8.064 |

Gradientes/loss finitos e shape do alvo preservado. Evidência:
`results/experiments/rate_allocation_smoke_20261012/`.

## 5. Execução e recuperação

Runner: `finetuning/run_rate_allocation_experiment.py`.
Saída: `results/experiments/rate_allocation_pilot_20261012/`.
Log: `results/experiments/rate_allocation_pilot_20261012.pipeline.log`.
Serviço de usuário **`febus-rate-allocation-20261012`**. **Resultados pendentes**.
Estimativa inicial **8–11 h**, incluindo avaliação final; atualizar com épocas medidas.

Retomada em 2026-10-13: após interrupção, havia três épocas completas do braço
Temporal e nenhum dos outros braços. GPU e protocolo/checksums foram verificados;
resume iniciado no serviço **`febus-rate-allocation-resume-20261013`**, mantendo
optimizer/RNG e orçamento restante. Uma mensagem de sucesso da sessão de espera
com D-Bus encerrado não substitui os marcadores `COMPLETE` do experimento/avaliação.

```bash
systemctl --user status febus-rate-allocation-20261012
# Serviço da retomada:
systemctl --user status febus-rate-allocation-resume-20261013

# Somente sem cópia ativa:
.venv/bin/python finetuning/run_rate_allocation_experiment.py \
  --output-dir results/experiments/rate_allocation_pilot_20261012 --resume
```

Fonte, corpus, protocolo e código fixados por hashes; locks e saves atômicos.
Resume mantém geometria, optimizer/RNG e orçamento restante; serviço não garante
sobrevivência a reboot/logout. Nenhum checkpoint anterior ou HDF5 foi apagado.
