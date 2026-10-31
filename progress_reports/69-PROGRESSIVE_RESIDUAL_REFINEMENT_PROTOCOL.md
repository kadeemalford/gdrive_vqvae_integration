# Piloto local de refinamento residual neural progressivo

**Piloto VQ concluído sem extensão:** ganho de fidelidade insuficiente, sem
promoção. Resultado e ablação FSQ separados no
[relatório 70](70-RESIDUAL_RESULTS_AND_FSQ_ABLATION.md).

## Hipótese e diferenças frente à ROI anterior

A troca por modo denso substitui 8.064 por 32.000 índices na janela máxima.
Testar **base compacta congelada + codec menor do erro de reconstrução**:
8.064 índices básicos + **2.016 de refinamento** em 512×1.000. São 25% mais
índices antes de headers, tabelas e metadados, não ganho gratuito de compressão.

O resíduo é **no sinal normalizado**, não RVQ dos embeddings nem a mesma troca
de modo do relatório 68. A hipótese é melhorar fidelidade por bit transmitido.

## Base, refiner e dados

- Base: `compact_A/best_macro.pt`, época 24, de `spatial_abc_20261024_v2`;
  **27.958.081 parâmetros congelados**, sem gradientes ou updates.
- Refiner: CNN nativa, base channels 32, **7.355.873 parâmetros**, quatro níveis
  padrão, K=2.048, dimensão 128 e commitment 0,25; inicialização aleatória.
- Corpus `spatial_band_v1_fold0_20261024`: 11.760 janelas de treino, 7.220 de
  validação; treino folds 2/3, validação 1, teste 0 sem inferência.
- Filtro e normalização originais inalterados; não reaplicar detector ao resíduo.
  HDF5 e NPZ originais preservados.
- Seed **2035**, reservatório de 64 janelas de treino, 16 por família, até 256
  vetores por janela. Codebook semeado só por resíduos de treino.

Para entrada normalizada `x`, a base produz `b` dos seus índices. O refiner recebe
`r = x - b` nas células válidas; escapes são zerados no input/perda e restaurados
bit-exatamente pelo side básico. **Não há normalização extra do resíduo nem
transmissão de escala contínua adicional.**

O decoder recebe códigos básicos e residuais quando admitidos, reconstrói
`b + r_hat` e aplica uma vez a normalização/escapes do frame básico. O original
fica no encoder/treino/avaliação, nunca é entrada gratuita do decoder. Checkpoints
residuais têm papel versionado e hash da base; não são codecs diretos de janelas
brutas intercambiáveis com os checkpoints anteriores.

## Orçamento e avaliação

- **RTX 4060 local apenas**, FP32, cuDNN determinístico, activation checkpointing
  no refiner e batch de uma janela; nenhuma execução remota de treino.
- AdamW LR 1e-4, weight decay 1e-5, clipping 1; 0,5 MSE + 0,5 L1 no resíduo
  válido + VQ. Com base constante, erro residual equivale ao erro da composição
  no espaço normalizado das células modeladas.
- **16×1.024 = 16.384 updates iniciais**, sem parada precoce nesse orçamento.
  É piloto de menor exposição que o estudo de capacidade, não prova de convergência.
- Extensão única de **8×1.024**, máximo **24.576 updates**, se melhora recente
  entre os dois últimos blocos de quatro épocas for ≥0,2% e o melhor RMSE com
  guarda for ≥1% melhor que a base. Sem prolongamento automático adicional.
- Validação a cada época: amostra fixa de até duas janelas por arquivo/região;
  seleção pelo menor RMSE macro com guarda. Registrar também composição **sem
  guarda** e quantidade de janelas admitidas para não esconder falhas no fallback.
- Avaliação completa dos selecionados nas fases 16 e, quando elegível, 24, com
  métricas físicas, escala/fase/espectro e canais fracos/não fracos por grupo.
- Tabelas de símbolos ajustadas só no treino. Todos os refinamentos são realmente
  codificados, decodificados para índices idênticos e somados à base decodificada
  de seu frame. O frame residual não tem side de normalização.
- Controles: base; todos os tiles refinados; todos os refinamentos admitidos;
  tetos por arquivo de **1,25× e 1,5× dos bytes básicos**.
- Admitir refinamento sem piora de RMSE/correlação/p90; priorizar ganho fracional
  limitado de RMSE² e p90 com pesos iguais, células e custo extra. É regra no
  encoder, não demonstração independente de generalização.
- **Alocação aninhada:** 1,5× preserva todos os refinamentos de 1,25×, para não
  retirar uma melhoria já transmitida ao expandir o orçamento.
- Bundles cobram frames, fronteiras, mapas, política, metadados, tabelas, sequência
  e flags de composição. Pesos FP32 compartilhados são reportados separadamente.
  Comparar taxa efetiva com relatório 68, não apenas contagem de índices.

## Execução e segurança

Runner: `finetuning/run_residual_refinement.py`.
Diretório: `results/experiments/residual_refinement_fold0_20261030/`.
Unidade: `febus-residual-refinement-20261030.service`.

Smoke inicial: três passos AdamW descartáveis no shape máximo real, gradientes
finitos, nenhuma atualização da base e 2.016 índices residuais. Depois reiniciar
seed/modelo e semear codebook. Suite: **233 testes** antes do treino.

Protocolo/fontes/metadados fixados por hashes/snapshot; checkpoints atômicos com
optimizer/RNG e orçamento retomável. Interromper com menos de 10 GiB livres, sem
limpeza automática; não alterar código durante a fila.

`metrics.csv`, `smoke.json`, `full_phase*`, bundles, `REPORT.md`, guardas,
`STATUS.json` e `COMPLETE` são fontes de estado. Não promover pela proteção
construída no encoder: tolerâncias físicas e validação externa seguem pendentes.

Se o refinamento não melhorar a fronteira erro–bytes, encerrar sem estender
automaticamente nem aumentar a rede. Se melhorar, investigar seeds/grupos e
continuidade nas bordas antes de integrar ao codec.

### Início verificado

O smoke passou na RTX 4060 local: **1.450.918.912 bytes (~1,35 GiB) de pico
alocado** no shape máximo, três passos AdamW descartados e 2.016 índices
residuais. Nenhum parâmetro da base recebeu gradiente. A fila entrou em treino,
época 1/16, em **30/10/2026 às 14:11 UTC**. Estado e métricas posteriores devem
ser consultados nos artefatos, não inferidos deste registro de início.
