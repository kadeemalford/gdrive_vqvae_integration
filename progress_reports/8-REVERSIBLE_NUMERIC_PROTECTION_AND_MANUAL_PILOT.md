# Proteção numérica reversível e próximo piloto manual

Data: 2026-10-07. Continuação da auditoria do relatório 8. **Nenhum treino real
executado nesta etapa**: testes unitários usam modelo sintético minúsculo; checkpoint
real foi usado somente em oito forwards, sem gradiente, de quatro janelas de validação.

## Resultado e limites

Implementada `escape-v1`, opcional e versionada. Fonte HDF5/NPZ não foi modificada.
Nas 23.880 janelas canônicas:

- 40 janelas com 57.857 células acima de `abs(valor)>1e20`.
- Todos os valores escapados restaurados **bit a bit**, incluindo sinal, após roundtrip.
- **Zero instâncias de canais inteiramente não extremos perderam toda a variação**
  na normalização, ao contrário do protocolo antigo nos recortes afetados.
- Máximo RMSE relativo do roundtrip da região ordinária: **3,7484×10⁻⁸**.
  É erro de normalização/conversão float32, **não qualidade de reconstrução VQ-VAE**.
- Side stream total não comprimido: **1.418.056 bytes**, **0,004584 bit/célula**.
  Inclui header de 40 bytes por janela e 8 bytes por escape (índice uint32 + float32).
  Não inclui índices VQ, arquivo/container, mapas espaciais nem demais custos de codec.
  O custo nos recortes afetados é bem maior que a média global.

Artefatos: `results/experiments/reversible_normalization_20261007/`.

## Formato e semântica

`finetuning/reversible_normalization.py`:

1. Requer array 2D float32 finito, sem alterar o original.
2. Separa células `abs>1e20`; esse threshold identifica risco numérico no corpus
   atual, **não define validade física** dos dados e ainda precisa de validação externa.
3. Calcula média/desvio float64 apenas sobre as demais células. Escala = std + 1e-6.
4. Células escapadas recebem zero no input normalizado; modelo não precisa estimá-las.
5. Side stream little-endian `FNE1` guarda shape, média, escala, threshold, contagem,
   índices ordenados e valores originais float32. Decoder valida header, comprimento,
   índices, finitude e faixa; escapes sobrescrevem a predição do modelo.
6. Região ordinária é restaurada em float32; **não é lossless**. A região escapada é.

Não houve clipping, exclusão de janelas ou classificação dos valores como inválidos.
O side stream foi materializado em memória nos testes; não foi criado novo corpus
nem anunciado um formato completo de compressão. Overhead deve entrar na futura
comparação rate-distortion. Corrupção de stream gera erro, mas checksum/transporte
e container completo ainda precisam ser definidos para produção.

## Integração do treino

`finetune_vqvae.py --normalization escape-v1`:

- Mantém `legacy` como default, preservando os experimentos anteriores.
- Exclui células escapadas do MSE/L1 e das métricas temporais de seleção; os demais
  canais continuam no alvo. VQ loss permanece ativa sobre a representação zero-filled.
- Registra fração escapada na validação e política nos checkpoints/resumos.
- Resume rejeita troca de política. Mudar política requer **novo braço, optimizer novo**.
- Requer `--validation-details`; spectral-weight deve permanecer zero. Espectros
  registrados são somente diagnósticos do array com escapes zerados, não espectros
  do alvo físico original. Não usar esses números para alegar preservação de banda
  nas regiões escapadas.
- Checkpoints passaram a ser gravados com arquivo temporário + replace atômico:
  falha de serialização preserva o arquivo anterior. Isso não torna o conjunto de
  três checkpoints uma transação única nem permite retomar exatamente no meio do epoch.

**Limite importante:** `evaluate_checkpoints.py`, probes antigos e ferramentas de
diagnóstico históricas continuam usando normalização antiga. Não usá-los para avaliar
um checkpoint escape-v1 sem adaptar explicitamente o loader/restore/métricas. Treinar
com escape não implementa automaticamente um codec de produção. A comparação futura
precisa avaliar região ordinária em unidades físicas e preservação/overhead do escape,
além dos seletores normalizados. RMSE global dominado por extremos continua inadequado.

## Smoke com o checkpoint real

`smoke_escape_inference.py`: fold 0, checkpoint uniform `best.pt`, dois arquivos
afetados Configuration 6, bloco 0 e bloco 1, janela inicial de cada bloco; todos val.
Seleção deliberada para integração, **não amostra representativa nem critério de promoção**.

| Arquivo / bloco | RMSE relativo raw da região ordinária, legacy | escape-v1 |
|---|---:|---:|
| 09-21-52 / 0, sem escapes | 0,9855 | 0,9855 |
| 09-21-52 / 1, afetado | 1,549×10¹⁹ | 0,9921 |
| 09-22-11 / 0, sem escapes | 0,9546 | 0,9546 |
| 09-22-11 / 1, afetado | 2,095×10⁷ | 0,9971 |

Os números legacy incluem inversão da sua escala global e mostram dano de dynamic
range na região ordinária. Os escapes do novo caminho ficaram bit-exatos após forward
e decode. Nos blocos sem escapes, inputs normalizados são iguais às operações antigas;
diferença residual de saída raw é apenas arredondamento da conversão final.
**RMSE ~1 continua ruim:** a proteção remove a falha numérica, não demonstra capacidade
de reconstruir as ondas. Nenhum ganho generalista foi inferido desse smoke.

Artefatos: `results/experiments/escape_inference_smoke_20261007/`.

## Próxima etapa: treino pareado, executar em terminal independente

Runner: `finetuning/run_numeric_adaptation_experiment.py`.

- Fold **2**: arquivos afetados pertencem ao treino; validação não contém nenhum dos
  três arquivos com alertas do relatório 8. Assim, alvos/métricas de validação são
  idênticos entre os braços, sem comparar denominadores incompatíveis.
- Dois braços, `legacy` e `escape-v1`, mesma inicialização uniform fold 2 `best.pt`,
  fresh AdamW, seed 2026, janela uniforme, LR **1e-6**, weight decay **1e-5**.
- **3 epochs × 2.048 updates por braço**, total máximo 12.288 updates. Duas janelas
  de validação por arquivo, estratificadas deterministicamente; nenhum forward de teste.
- Inicialização epoch 0 retida como candidata. `best.pt`, `best_macro.pt`, `last.pt`,
  métricas por configuração/macro e progresso/log streaming. Sem promoção automática.
- É ablação conservadora **exploratória**, não treino final. Baixa fração dos recortes
  afetados e orçamento limitado podem produzir poucos updates neles; resultado nulo
  não prova inutilidade da proteção. A cobertura/exposição exata deverá ser auditada
  na análise posterior. Configurações completas e novo holdout continuam necessários.
- ~3,1 GB de checkpoints estimados para dois braços; runner exige ao menos 5 GiB
  livres para headroom. Disco tinha ~14 GB livres ao preparar o comando.
- `--dry-run` passou e não criou output nem iniciou processos de treino.

No terminal independente, a partir da raiz do projeto (preferencialmente em tmux):

```bash
tmux new -s febus-numeric
.venv/bin/python finetuning/run_numeric_adaptation_experiment.py \
  --output-dir results/experiments/numeric_adaptation_pilot_20261007
```

Detach: `Ctrl+b`, depois `d`. Retomar sessão: `tmux attach -t febus-numeric`.
Se o processo terminar/interromper, retomar checkpoints com:

```bash
.venv/bin/python finetuning/run_numeric_adaptation_experiment.py \
  --output-dir results/experiments/numeric_adaptation_pilot_20261007 --resume
```

Resume retoma o último epoch completo; updates do epoch interrompido serão repetidos.
Concluir braços não significa que houve melhora. Depois, comparar inicialização/melhor
validação, cada família e pior família, sem selecionar peso por teste.

## Testes e pendências

**60 testes passaram**: roundtrip/bit pattern, preservação de variação, compatibilidade
sem escapes, streams inválidos, máscara temporal, treino/resume sintético, salvamento
atômico sob falha e preflight/dry-run do runner. `git diff --check` passou.

Rastreamento de canais físicos, store canônico para eliminar repetição por fold,
normalização por canal e MoS/fine-tuning parcial continuam pendentes. Não foram
feitas migrações, exclusões, novos downloads, treino real ou mudanças de arquitetura.
**Ponto de parada atingido: próximo experimento envolve treino e aguarda execução manual.**
