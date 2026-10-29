# Corpus espacial congelado e comparação neural A/B/C em duas taxas

## 1. Decisão e evidência já concluída

Avançar da seleção espacial para uma **comparação estrutural controlada**, não
para outra continuação isolada do 4× histórico. A política
`persistent-amplitude-v1` foi mantida sem ajuste; a faixa próxima de 60 m não
recebeu uma regra especial. Os quatro tipos de fibra participam do mesmo corpus.

Preparação: `data/finetune/spatial_band_v1_fold0_20261024/`.
Auditoria: `results/experiments/spatial_corpus_audit_20261024/`, `COMPLETE`.

| Papel | Janelas preparadas |
|---|---:|
| Treino, folds 2 e 3 | 11.760 |
| Validação, fold 1 | 7.220 |
| Teste reservado, fold 0 | 9.800 |
| Total | **28.780** |

**Todas as 28.780 janelas foram conferidas contra os 207 HDF5 originais**:
igualdade bit a bit dos valores retidos, finitude, shapes, mapas físicos,
ausência de sobreposição espacial entre tiles e cobertura completa dos canais
retidos em cada bloco. As configurações não cruzam splits. Os 9.800 exemplos
reservados foram inspecionados apenas para integridade/preparação, **não receberam
inferência neural e não entram na seleção experimental**.

Hash agregado dos arrays:
`328d993cfc3ec9a3de457966e45a5fb213f1e73b84d23e974cbfcc21578829eb`.
Hashes dos brutos, manifestos, mapas e preparação estão no `summary.json` da auditoria.
Nenhum dataset histórico foi sobrescrito.

O preflight adicional de normalização passou em **todas as 11.760 janelas de
treino e 7.220 de validação**: valores normalizados finitos e restauração exata
das células escapadas (`normalization_preflight.json` na auditoria). Não houve
escapes no treino; a validação ainda exigiu 40.742 escapes. Portanto, o filtro
espacial **não elimina toda patologia numérica remanescente** e o mecanismo
reversível de proteção continua necessário, com o mesmo custo e máscara de
referência para todos os candidatos. Essa checagem não usou os modelos.

O detector selecionou faixas em **40 dos 207 arquivos**; nos demais, não houve
excesso persistente que atendesse à regra. A seleção espacial total, incluindo
remoção de canais não finitos, preservou as seguintes frações de instâncias
canal × bloco do bruto:

| Família | Cobertura espacial retida |
|---|---:|
| Doped | 90,14% |
| Multicore | 92,11% |
| Multimode | 97,21% |
| Single-mode | 95,66% |

Essas frações **não são porcentagens de remoção exclusivamente por amplitude**,
nem frações de bits economizados pelo codec. A seleção não cria aquisições novas.

## 2. Tratamento dos tiles e seleção de validação

Faixas separadas são amostras separadas, sem juntar posições não contíguas.
Regiões maiores que 512 canais são divididas em tiles sem perder o restante.
Há tiles pequenos, inclusive de um ou dois canais, preservados deliberadamente.

Para não atribuir a um tile de um canal o mesmo peso de um tile de 512 canais,
a **nova validação pondera os tiles por número de células dentro da configuração**,
mantendo peso igual para configurações dentro de cada família e para famílias.
Os resultados novos não são diretamente intercambiáveis com médias históricas
por janela. Métricas relativas ainda usam a referência normalizada de cada tile;
o p90 agregado é uma média ponderada de quantis por tile, **não um quantil pooled
de todos os canais do arquivo**.

A triagem por época usa **130 janelas fixas**, sorteadas com seed `2031`: até duas
por arquivo e índice de região retida, para não omitir uma região pequena apenas
por limitar a amostra por arquivo. A avaliação de cada checkpoint selecionado
usa **todas as 7.220 janelas de validação**. Essa validação é de desenvolvimento,
reutilizada nas escolhas, não um holdout externo independente.

A loss de treino permanece `0,5*MSE + 0,5*L1 + VQ`, sem loss espectral ou auxiliar
por canal. Sampling uniforme por janela, mesma seed e mesma ordem de exposição
entre braços; o efeito do novo tiling é comum e explicitamente parte do protocolo.

## 3. Braços e arquitetura C implementada

| Braço | Arquitetura | Inicialização |
|---|---|---|
| A | `native-v1` | Offshore original, sem carregar nenhum fine-tuning FEBUS |
| B | `native-v1` | Aleatória; treino somente FEBUS |
| C | `anisotropic-multiscale-v1` | Aleatória; treino somente FEBUS |

O original `models/febus_finetune_base/best.pt` permanece referência de origem.
O 4× histórico da época 40 é **controle adicional**, não inicialização de A.

C não é a variante que apenas troca um kernel 3×3 por 1×9. Implementação em
`finetuning/anisotropic_vqvae.py`:

- Stem temporal 1×9 e espacial 3×1.
- Redução por eixo em estágios separados: convoluções temporais 1×4 e espaciais
  4×1, com strides independentes e a mesma grade final de índices dos controles.
- Blocos multiescala com ramos depthwise temporal local 1×9, temporal dilatado
  1×9 e espacial 5×1, seguidos por fusão pontual e residual interno.
- Contexto adicional antes do gargalo, com dilatação temporal 8.
- Reconstrução por eixo com convoluções transpostas separadas e um decoder
  compartilhado. **Nenhum skip encoder→decoder atravessa o gargalo.**
- Mesmo VQ de 2.048 vetores, dimensão 128 e commitment cost 0,25.

C tem **26.302.273 parâmetros** versus **27.958.081** na nativa, diferença de
aproximadamente −5,9%. Capacidade é próxima, não exatamente pareada. Atenção,
roteamento por família e OWR não participam desta primeira comparação.

B/C usam reservatório de latentes de **32 janelas somente do treino**, oito por
família e até 256 vetores por janela, para inicializar o mesmo tipo de codebook.
Não há calibração em validação/teste nem atualização de gradientes nessa etapa.

## 4. Dois orçamentos de bits comuns

| Ponto | Geometria | Índices reais em 512×1.000 | Bits por índice |
|---|---|---:|---:|
| Compacto | `spatial-4x` | 8.064 | 11 |
| Maior fidelidade | `spatial4-temporal4` | 32.000 | 11 |

As contagens são **idênticas entre A/B/C para cada shape real**, verificadas pelo
forward e pelo cálculo da geometria. A taxa de todo o corpus será recalculada
por soma de bits, incluindo side de normalização/escapes. Mapas espaciais,
container/modelo e demais custos de implantação não estão integralmente incluídos
nessa taxa nominal. Exclusão de cobertura não é ganho de compressão neural.

### Replay discreto exato e memória

O smoke identificou que a expressão straight-through de treino
`z + (q-z).detach()` pode diferir numericamente do vetor `q` do dicionário,
por arredondamento. A nova avaliação ativa `exact_eval_quantization=True`:
reconstitui **exatamente os vetores associados aos índices**, mantendo o mesmo
quantizador e os gradientes straight-through no treinamento. Checkpoints
históricos não têm seu comportamento alterado por padrão; os controles desta
coorte também são avaliados com replay discreto exato.

Um primeiro teste C excedeu a memória da GPU. Foi adotado checkpointing de
ativações, recomputando blocos durante o backward, sem reduzir canais, trocar
precisão ou mudar o objetivo. A opção também foi ativada na nativa.

Smoke aprovado: `results/experiments/anisotropic_rate_smoke_checkpointed_20261024/`,
`COMPLETE`. Houve forward, backward, AdamW, contagem de índices e replay pelo
decoder apenas dos códigos em **512×1.000**, nas duas arquiteturas e taxas.
Picos CUDA alocados: nativa ~2,63/3,28 GiB; C ~4,44/5,21 GiB, respectivamente
compacto/denso. Essas medidas são de preflight, não benchmark de throughput final.

**215 testes passaram** antes do lançamento, incluindo shapes, gradientes,
checkpoint replay, igualdade dos códigos, cobertura de regiões na triagem,
ponderação de tiles e regra de extensão.

## 5. Protocolo longo e autonomia limitada

Runner: `finetuning/run_spatial_abc_study.py`.
Avaliador: `finetuning/evaluate_spatial_study.py`.

1. Avaliar integralmente o offshore original, o 4× histórico e as inicializações
   A nas duas taxas, com o mesmo corpus corrigido e política de avaliação.
2. Executar os seis braços, priorizando a nova hipótese C no ponto denso:
   `high_C → high_A → high_B → compact_C → compact_A → compact_B`.
3. **16 épocas × 4.096 updates = 65.536 updates por braço**, AdamW novo,
   LR `1e-4`, weight decay `1e-5`, seed `2031`. Sem parada antecipada nesse orçamento.
4. Retenção do melhor macro RMSE da triagem e auditoria completa do selecionado;
   guardar snapshot da fase 16 antes de qualquer extensão.
5. Depois dos seis braços iniciais, estender **apenas os que ainda melhoram**:
   redução ≥0,2% entre o melhor RMSE das épocas 9–12 e o das épocas 13–16 implica
   **mais oito épocas**, sem outra extensão. Máximo **24 épocas/98.304 updates
   por braço**, seis braços no total. Não exigir que scratch já vença A para
   merecer essa extensão. Modelo em melhora no teto é inconclusivo sobre convergência.
6. Novos snapshots e avaliações completas após extensão. Comparação principal
   pareada de compute adicional na fase 16; fase 24 é seletiva e será identificada
   separadamente. Pré-treino offshore não equivale a compute total gratuito/pareado.
7. Guardas por configuração para RMSE/correlação/p90 contra A à mesma taxa, e,
   no ponto denso, também contra o 4× histórico reavaliado no corpus corrigido.
   Pequenas médias favoráveis não promovem automaticamente um codec.

Corpus, código, fontes, seeds, smoke e regras são registrados por hash antes dos
treinos. O runner interrompe se houver mudança nos arquivos fixados, falta de
espaço ou erro de subprocesso. Logs, checkpoints atômicos, status e marcadores
permitem auditoria/retomada. **Nenhum modelo final é promovido automaticamente.**

## 6. Execução e acompanhamento

### Estudo concluído

`COMPLETE` e o status final confirmam término em **2026-10-26 23:12 UTC**:
seis braços com 24 épocas e 12 selecionados avaliados integralmente (fases 16/24).
Nenhum modelo foi promovido. A síntese e a confirmação proposta estão no
[relatório 63](63-SPATIAL_ABC_RESULTS_AND_FOLD_CONFIRMATION.md). As retomadas
abaixo são histórico de execução, não indicação de que este estudo segue ativo.

### Retomada final: B compacto

Na checagem seguinte, cinco braços já tinham completado as 24 épocas e suas
avaliações completas. `compact_B` tinha checkpoint e CSVs coerentes até a
**época 22**, mas nenhum processo estava ativo e ainda não havia `COMPLETE` do
estudo. Código e metadados continuavam com os hashes originais. A fila foi
retomada da época 22 para executar somente **as duas épocas restantes**, a
avaliação completa de B compacto e a síntese das guardas.

Resultados compactos já concluídos: A, selecionado na época 24, atingiu RMSE
**0,667586** e correlação **0,727658**; C, selecionado na época 23, atingiu
**0,700077 / 0,697539**. Ambos operam a **160,211× nominais** no mesmo corpus.
Não se encerrou a comparação antes do último braço nem se ampliou o teto.

### Retomada das extensões restantes — 26/10

Na nova checagem não havia processo ativo nem unidade transitória disponível;
o estudo ainda não tinha `COMPLETE`. Os seis braços tinham encerrado a fase 16,
e a regra fixada autorizou extensão para todos. Os checkpoints confirmaram
**24 épocas nos três braços densos**, **20 em `compact_C`** e **16 em
`compact_A`/`compact_B`**. Hashes do código e dos metadados do corpus continuavam
iguais ao protocolo.

A fila foi retomada com `--resume`, sem refazer os braços já completos: faltavam
quatro épocas de C compacto e oito de cada controle compacto, seguidas pelas
avaliações completas e síntese. O teto de 24 épocas por braço não foi ampliado.

Resultados completos dos selecionados após a extensão densa, à mesma taxa:

| Braço denso | Época selecionada | RMSE macro completo | Correlação macro completa | Compressão nominal pooled |
|---|---:|---:|---:|---:|
| A — offshore + adaptação | 23 | **0,445187** | **0,888093** | 41,577× |
| B — nativa do zero | 23 | 0,482705 | 0,867561 | 41,577× |
| C — anisotrópica do zero | 22 | 0,495379 | 0,859700 | 41,577× |

A mantém vantagem agregada, mas **não é uma substituição generalista aprovada**:
contra o 4× histórico reavaliado na mesma coorte, regride em RMSE/correlação/p90
de multicore C1 and C3, em RMSE/p90 de single-mode Configuration 10 e no p90 de
doped 5m LSF. Os três candidatos densos têm algum veto contra essa referência.
Não se alterou a guarda para acomodar a média favorável.

Na fase compacta inicial, os RMSE completos eram A **0,684070**, C **0,707319**
e B **0,715659**. Essa ordem ainda é provisória, pois faltam as extensões compactas
e a análise consolidada por configuração. Nenhum novo protocolo ou arquitetura
foi iniciado antes de concluir a comparação predefinida.

### Retomada e resultados parciais — 25/10

Na checagem de **2026-10-25 11:44 UTC**, não havia trainer/orquestrador ativo e
a unidade transitória não estava disponível. Não havia `COMPLETE` do estudo.
Os checkpoints confirmaram os três braços densos concluídos e `compact_C` na
**época 13**, com CSVs de treino/validação coerentes até essa época. A notificação
cancelada pelo reinício do servidor não foi usada como prova de término nem
como diagnóstico da causa da interrupção do treinamento.

Hashes do código continuavam iguais ao protocolo. A fila foi retomada com
`--resume`, na mesma unidade e diretório, e o novo status confirmou
`compact_C`, **13 → 16 épocas**. Não se reiniciaram os braços densos nem se
mudou o orçamento; optimizer e RNG vêm da última época completa.

Avaliações completas já disponíveis, mesma fase de 65.536 updates e mesma taxa:

| Braço denso | Época selecionada na triagem | RMSE macro completo | Correlação macro completa | Compressão nominal pooled |
|---|---:|---:|---:|---:|
| A — offshore + adaptação | 16 | **0,458709** | **0,881087** | 41,577× |
| B — nativa do zero | 16 | 0,490299 | 0,863342 | 41,577× |
| C — anisotrópica do zero | 14 | 0,506769 | 0,853744 | 41,577× |

A é melhor nesses dois agregados na fase densa inicial; isso **não encerra a
comparação**: faltam os pontos compactos, as extensões predefinidas e a síntese
das guardas por configuração/canal. Não se promoveu um candidato nem se alterou
a arquitetura/otimização em função desses resultados parciais.

### Lançamento e controles congelados

**Controles congelados completos e primeiro treinamento iniciado:** a unidade
passou para `high_C` em **2026-10-24 18:06 UTC**, com meta inicial de 16 épocas.
O subprocesso do trainer foi confirmado ativo. As quatro referências foram
avaliadas nas mesmas 7.220 janelas:

| Referência | Geometria | RMSE macro | Correlação macro | Compressão nominal pooled |
|---|---|---:|---:|---:|
| Offshore original sem adaptação | Standard | 0,992525 | 0,155285 | 574,143× |
| 4× histórico, época 40 | Spatial4-temporal4 | 0,464043 | 0,880575 | 41,577× |
| A antes do treino, ponto compacto | Spatial-4x | 1,000009 | 0,091461 | 160,211× |
| A antes do treino, ponto denso | Spatial4-temporal4 | 1,009581 | 0,097107 | 41,577× |

As inicializações A mantêm os pesos offshore, mas mudam a grade latente; não
se espera a mesma saída do offshore standard. O desempenho do 4× histórico não
foi transferido para A. Nenhum resultado de treino longo está estabelecido por
essa tabela; o avanço é registrado nos checkpoints e métricas dos braços.

A primeira referência completa já medida no corpus novo, **offshore original**,
teve RMSE macro **0,992525**, correlação **0,155285** e razão pooled nominal
**574,143×**. Esses números usam o alvo/ponderação/replay exato desta coorte,
não devem ser subtraídos diretamente de médias históricas de outro protocolo.

O p90 macro por canal dessa referência ficou ~**9,50e6**, concentrado em
multimode Configuration 6, enquanto o erro médio de canais fora do grupo fraco
diagnóstico ficou **1,0975**. Isso confirma uma limitação remanescente de faixa
dinâmica/denominadores locais, não uma melhora ou piora uniforme de todos os
canais. Os casos não foram apagados da avaliação nem o filtro foi ajustado
após observar esse score. Guardas, métricas físicas, estratos e caudas devem ser
interpretados em conjunto; não haverá promoção automática por média ou por
redução dessa cauda isolada.

Execução concluída: `results/experiments/spatial_abc_20261024_v2/`.
Unidade independente do chat: **`febus-spatial-abc-20261024-v2.service`**, via
systemd do usuário, com teto de 48 h e 22 GiB de RAM. O teto operacional não
substitui o orçamento científico; interrupção não implica conclusão do estudo.
Não há reinício automático em caso de erro nem retomada automática após desligar
o computador. Checkpoints preservam a última época completa.

A primeira tentativa de inicialização parou por configuração do caminho de importação
de `smartgrids`, antes de treinar; o caminho foi corrigido e uma nova execução
com protocolo próprio foi iniciada. A tentativa anterior não é evidência de treino.

Estado e resultados autoritativos:

- `STATUS.json` / `STATUS.md`: etapa e timestamps.
- `protocol.json` / `initializations.json`: hashes e inicializações.
- `baselines/`: controles congelados na validação completa.
- `high_C/`, demais braços, logs e checkpoints: evolução dos treinamentos.
- `retained_phase16/`, `retained_phase24/`: selecionados preservados por fase.
- `full_phase*/`: avaliações completas, por configuração e janela.
- `extension_decisions.json`, `guards.json`, `REPORT.md`: decisões e síntese ao fim.
- `COMPLETE`: fila finita concluída; `FAILED`/status de falha requer investigação.

```bash
systemctl --user status febus-spatial-abc-20261024-v2.service --no-pager
cat results/experiments/spatial_abc_20261024_v2/STATUS.json
```

Para retomar após investigar uma interrupção, com código/corpus inalterados:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -u \
  finetuning/run_spatial_abc_study.py \
  --output-dir results/experiments/spatial_abc_20261024_v2 \
  --samples-dir data/finetune/spatial_band_v1_fold0_20261024 \
  --audit-dir results/experiments/spatial_corpus_audit_20261024 \
  --smoke-dir results/experiments/anisotropic_rate_smoke_checkpointed_20261024 \
  --resume
```

Não executar uma retomada enquanto a unidade original estiver ativa. O lock
impede dois orquestradores simultâneos. Resultados de execução devem ser lidos
nos artefatos acima, não inferidos de uma notificação de espera do chat.
