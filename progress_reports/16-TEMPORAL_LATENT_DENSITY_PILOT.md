# Ablação neural de densidade temporal dos latentes

> Resultados do piloto, validação completa e protocolo de continuação estão no
> [relatório 17](17-TEMPORAL_DENSITY_FULL_RESULTS_AND_LONG_CONTINUATION.md).

## 1. Hipótese e estado do experimento

Após o [controle DCT de taxa](15-SPARSE_DCT_RATE_DISTORTION_RESULTS.md), comparar
a rede na geometria padrão com compressão temporal de features menos agressiva.
Mais posições latentes podem preservar detalhes perdidos no domínio FEBUS a 20 kHz.
O resultado é hipótese, não ganho presumido.

Piloto iniciado em background:
`results/experiments/temporal_density_pilot_20261008/`.
Log: `results/experiments/temporal_density_pilot_20261008.log`.
Runner: `finetuning/run_temporal_density_experiment.py`.

## 2. O que muda e o que permanece controlado

- **Padrão:** quatro downsamplings com stride (2,2).
- **Temporal-2x:** último downsampling tem stride (2,1); primeiro upsampling
  correspondente do decoder também (2,1).
- **Temporal-4x:** últimos dois downsamplings temporais e primeiros dois upsamplings
  correspondentes usam stride (2,1).

Preservam-se kernel 4×4, padding de convolução, níveis, stride espacial, número e
forma dos parâmetros, embedding/codebook, alvo original, normalização global e
padding da entrada a múltiplos de 16. Não há decimação do sinal raw, novo tiling,
remoção de bandas, canais ou mudança de duração da janela.

**Mais bits e geometria/receptive field temporal mudam juntos.** Não chamar essa
ablação de comparação à mesma taxa ou efeito causal somente do bitrate. Mesmos
updates não significam mesmos FLOPs ou runtime. A rede não recebe uma informação
extra gratuita: mais índices precisam ser cobrados no codec.

### Quantidade real de índices

Para 512×1.000, padding temporal 1.008:

| Política | Posições latentes | Bits de índices (11 por código) | Razão nominal índices+40 bytes de side |
|---|---:|---:|---:|
| Standard | 2.016 | 22.176 | ~728,31× |
| Temporal-2x | 4.000 | 44.000 | ~369,68× |
| Temporal-4x | 8.000 | 88.000 | ~185,17× |

2x/4x são nomes aproximados; kernels pares/padding afetam a contagem, não bastando
dividir o tempo por 8/4. `latent_positions()` reproduz a topologia, e a validação
confere a previsão contra `indices.numel()` do forward denso. Taxa continua nominal
parcial, sem container, mapas físicos/model IDs e codificação entrópica.

## 3. Protocolo pareado

| Item | Protocolo |
|---|---|
| Fonte | global best_macro epoch 6 do piloto de normalização |
| Braços | standard, temporal-2x, temporal-4x |
| Fold / dados | fold 1; pool de 12.140 janelas de treino, sampler uniforme |
| Orçamento | **6×2.048 updates por braço**, máximo 36.864 totais |
| Inicialização | mesmas formas/pesos, AdamW novo em cada braço |
| LR / weight decay | 1e-5 / 1e-5 |
| Seeds e ordem | seed 2026, mesmo sampler e orçamento |
| Validação | duas janelas por arquivo: 102 janelas, quatro famílias, seis configurações |
| Objetivo | temporal MSE/L1+VQ, sem perda espectral |
| Seleção | macro RMSE no mesmo alvo restaurado; inicialização candidata |
| Parada | patience 7; não interromper antes do orçamento pareado de seis epochs |
| Teste | não consultado |

Os pesos iguais não garantem outputs iniciais iguais porque strides mudam a
representação. Se o denso apenas recuperar uma inicialização pior, comparar sua
qualidade absoluta com o standard, não somente ganho relativo ao próprio passo 0.

## 4. Verificações antes de treinar

`smoke_temporal_geometry.py` executou forward, backward, grad clip e **um update
descartado por política** na maior forma de treino, 512×1.000. Nenhum peso/checkpoint
desse teste foi salvo ou usado para iniciar o piloto.

| Política | Posições verificadas | Pico alocado GPU |
|---|---:|---:|
| Standard | 2.016 | 3.507.850.752 bytes |
| Temporal-2x | 4.000 | 3.587.918.848 bytes |
| Temporal-4x | 8.000 | 3.963.848.704 bytes |

Objectives finitos, mesma forma de saída, contagens corretas; viável na RTX 4060
de 8 GiB nas condições do smoke. Não tratar runtime do primeiro forward com
warmup CUDA como comparação de velocidade; benchmark operacional virá do treino.
Artefatos: `results/experiments/temporal_geometry_smoke_20261008/`.

**102 testes passaram** antes do lançamento, incluindo modelo nativo pequeno,
preservação bit-exata de parâmetros ao mudar/reverter strides, índice/padding,
treino/resume com geometria herdada, recusa de troca explícita no resume e
checagem de metadata conflitante.

## 5. Compatibilidade e operação

Strides não estão no `state_dict`. Checkpoints guardam `latent_geometry` em config
e finetune; o loader deve aplicar `finetuning/latent_geometry.py` ao construir a rede.
O treinador e o avaliador físico fazem isso; ferramentas históricas locais que
assumem stride padrão **rejeitam checkpoint denso**, em vez de avaliá-lo incorretamente.
O loader externo original de smartgrids, sem esse adapter, não é suficiente.

`evaluate_physical_candidates.py` também aceita candidatos explícitos
`--candidate label=escape-v1=caminho/checkpoint.pt` e restaura a geometria gravada.
Isso permite avaliar candidatos do piloto na validação completa sem alterar o
alvo/split. Os resultados densos exigem esse caminho stride-aware.

Logs/protocolos/locks e saves atômicos preservam checkpoints anteriores. Disco
tinha ~15 GiB livres; preflight exige 5 GiB. Background depende do serviço OpenCode.
Se interrompido:

```bash
.venv/bin/python finetuning/run_temporal_density_experiment.py \
  --output-dir results/experiments/temporal_density_pilot_20261008 --resume
```

Retoma último epoch completo, repetindo updates do epoch interrompido.
Após conclusão, comparar com:

```bash
.venv/bin/python finetuning/summarize_numeric_adaptation.py \
  --experiment-dir results/experiments/temporal_density_pilot_20261008
```

O orçamento deve levar da ordem de uma hora ou mais, sujeito a formas, validação,
I/O e maior custo dos braços densos. Isso é estimativa, não ETA medido do piloto.

## 6. Critério de decisão posterior

- Comparar RMSE/correlação, fase/amplitude, canais e pior família no alvo comum,
  cobrando taxa efetiva de índices/side por geometria.
- Não promover ganho restrito a uma fibra ou baseado somente em VQ loss.
- Se houver melhora consistente, confirmar na validação completa e em outras
  seeds/folds antes de ampliar o orçamento ou reduzir o stride ainda mais.
- Se não houver, investigar inicialização/otimização e arquitetura mais apropriada,
  ou representação/codec híbrido controlado. O teste negativo não prova impossibilidade
  de reconstrução nem falta de dados como causa única.
- Definir tolerâncias físicas e manter novo holdout externo antes de modelo final.

Dados, baselines e todos os checkpoints anteriores permanecem intactos; nenhuma
promoção de modelo ou aquisição/download foi feita nesta etapa.
