# Integração DAS do Google Drive com VQ-VAE

Projeto experimental independente do repositório `smartgrids`. Ele avalia a fidelidade da reconstrução produzida pelo checkpoint convolucional VQ-VAE treinado no outro repositório:

```text
/home/murilo-barroso/source/smartgrids/runs/vqvae_512ch/best.pt
```

O projeto não converte `Strain` em `Strain Rate`, nem o contrário. Cada arquivo é processado usando a variável de saída que já existir no HDF5.

## Escopo

O piloto usa seis arquivos da pasta compartilhada `DAS`:

- dois arquivos SMF da `Configuration 1`, em 200 Hz e 10000 Hz;
- um arquivo de SMF dopada, em 1000 Hz;
- um arquivo de fibra i29, em 1000 Hz;
- dois canais do ensaio multimodo `C1 and C2`, em 1000 Hz.

Os arquivos originais do Drive não são modificados. Os dados transferidos ficam em `data/raw/`, que é ignorado pelo Git. Os resultados permanecem locais.

## Estrutura

```text
config/          seleção de arquivos e parâmetros do piloto
acquisition/     aquisição, inventário e extração genérica de HDF5 FEBUS
compression/     preparação, codificação e decodificação VQ-VAE
presentation/    gráficos e relatório tabular
data/raw/        HDF5 copiados do Drive
data/extracted/  matrizes distância-tempo extraídas
data/samples/    blocos [canais, tempo] para o modelo
results/         manifestos, métricas, reconstruções e figuras
```

## Dependência do modelo

O modelo `VQVAE` e seus pesos são reutilizados a partir de `smartgrids/vqvae`. `compression/encode_decode_samples.py` adiciona o caminho definido por `repo_root` a `sys.path` para carregar o checkpoint; este projeto não copia nem duplica o modelo convolucional. A normalização da inferência é implementada no próprio script de compressão.

## Pipeline

```text
config/*.json
  -> IDs do Drive e parâmetros da execução
  -> HDF5 local
  -> inventário de grupos/datasets/atributos
  -> janela FEBUS [bloco, canais, tempo]
  -> normalização por trecho temporal
  -> índices discretos VQ-VAE
  -> decoder
  -> reconstrução desnormalizada
  -> métricas e figuras
```

## Scripts

| Script | Função | Entrada principal | Saídas |
| --- | --- | --- | --- |
| `acquisition/download_selected_files.py` | Baixa os HDF5 definidos em `files` por ID público do Drive, com `gdown`. | Configuração e Drive. | `data/raw/*.h5`. |
| `acquisition/download_selected_files_oauth.py` | Baixa a mesma seleção por OAuth quando o acesso público não é suficiente. | Configuração, credencial OAuth e Drive. | `data/raw/*.h5` e token local. |
| `acquisition/inventory_hdf5.py` | Lista datasets, formatos e atributos de cada HDF5 antes da extração. Não altera os dados. | Configuração e `data/raw/*.h5`. | `results/manifests/hdf5_inventory.csv`. |
| `acquisition/extract_febus_windows.py` | Localiza `Strain Rate` ou `Strain`, converte o arranjo FEBUS para `[bloco, canal, tempo]`, recorta a faixa espacial e concatena a janela temporal de cada bloco. | Configuração e HDF5 locais. | `data/samples/*.npz` e `results/manifests/extracted_samples.json`. |
| `compression/encode_decode_samples.py` | Carrega o checkpoint do repositório `smartgrids`, normaliza cada trecho temporal, codifica, quantiza, decodifica e desfaz a normalização. Valores `NaN` são preservados. | Configuração e `data/samples/*.npz`. | `results/reconstructions/*.npz`, `results/metrics/per_sample_metrics.csv`, `results/reconstruction_summary.csv` e `results/metrics/summary.json`. |
| `compression/compute_fidelity_metrics.py` | Calcula correlação, RMSE, correlação do RMS espacial e erro de potência nas bandas de 0-1, 1-5 e 5-25 Hz. | `results/reconstructions/*.npz`. | `results/metrics/fidelity_metrics.csv`. |
| `presentation/plot_results.py` | Gera diagnósticos comparando sinal original e reconstruído: waterfall, evento, estatísticas espaciais, FFT frequência-distância, espectro de ruído e quadro-resumo. | Configuração, reconstruções e resumo de métricas. | `results/figures/<amostra>/*.png` e `results/figures/reconstruction_metrics_summary.png`. |
| `acquisition/update_drive_das_documents.py` | Atualiza o README e corrige o notebook `SR_processing.ipynb` no Drive; não participa da reconstrução VQ-VAE. | Arquivos locais preparados e OAuth com escrita. | Documentos remotos atualizados. |

## Configuração das execuções

As execuções usam arquivos JSON em `config/`. `pilot.json` descreve o piloto de seis arquivos, `comparison_pilot.json` a comparação de três arquivos e `spatial_range_pilot.json` a execução atual de 3 s, 20-130 m, do arquivo SMF de 200 Hz.

| Campo | Uso |
| --- | --- |
| `repo_root` | Caminho absoluto para o repositório `smartgrids`, de onde são importados `VQVAE` e os utilitários de carregamento. |
| `checkpoint` | Caminho absoluto do arquivo de pesos VQ-VAE (`best.pt`) usado na reconstrução. |
| `data_type` | Dataset a extrair: `auto` prioriza `Strain Rate`, depois `Strain`; pode ser `strain-rate` ou `strain`. Não converte uma variável na outra. |
| `time_samples` | Número total de amostras temporais mantidas após a concatenação dos blocos FEBUS. Em `spatial_range_pilot.json`, `60000` corresponde a 3 s em 20 kHz. |
| `model_time_samples` | Tamanho máximo de cada trecho temporal passado ao VQ-VAE. A execução espacial usa `1024` para limitar memória; o script processa todos os trechos e os concatena. |
| `source_time_offset` | Índice inicial, dentro de cada bloco FEBUS, do intervalo que será retido. |
| `source_block_samples` | Quantidade de amostras retidas de cada bloco FEBUS antes da concatenação. Com offset `500` e tamanho `20000`, seleciona `[500:20500]`. |
| `spatial_start_m`, `spatial_end_m` | Distâncias inicial e final solicitadas para a amostra extraída. |
| `source_distance_start_m`, `source_distance_end_m` | Limites físicos do eixo espacial presente no HDF5; permitem converter os limites solicitados em índices de canal. |
| `default_dt_s` | Intervalo temporal de reserva para os gráficos quando o HDF5 não fornece `dt_s`; o extrator prefere o atributo FEBUS `Spacing`. |
| `normalization` | Registra a estratégia esperada na configuração e nos metadados. A implementação atual de reconstrução usa normalização por trecho temporal, com média e desvio padrão calculados somente nos valores finitos. |
| `device` | Dispositivo PyTorch para a inferência, por exemplo `cpu` ou `cuda`. |
| `display_vmin`, `display_vmax` | Limites fixos da escala de cor nos waterfalls. São usados para comparar visualmente original, reconstrução e erro na mesma escala. |
| `fft_fmin_hz`, `fft_fmax_hz` | Limites da faixa de frequência mostrada nos gráficos de FFT frequência-distância. |
| `channels`, `windows_per_file` | Campos presentes no `pilot.json` legado; a extração atual usa o recorte espacial e `time_samples`, portanto esses valores não controlam o processamento atual. |

Cada objeto do vetor `files` identifica uma entrada do Drive e acrescenta metadados à amostra e às métricas:

| Campo de `files` | Uso |
| --- | --- |
| `id` | ID do arquivo no Google Drive, usado pelos dois baixadores. |
| `title` | Nome informativo exibido no log de download. |
| `local_filename` | Nome estável do HDF5 em `data/raw/` e dos artefatos derivados. Se ausente, o ID recebe a extensão `.h5`. |
| `logical_path` | Caminho lógico de origem no Drive, preservado como metadado para rastreabilidade. |
| `group` | Grupo experimental/fibra, preservado nos manifestos e métricas para comparação. |
| `channel` | Identificador do canal físico, quando aplicável ao ensaio multimodo. |
| `frequency_hz` | Rótulo/frequência nominal do arquivo ou experimento, preservado como metadado. As métricas espectrais usam `dt_s` extraído do HDF5; este campo só é fallback para reconstruções sem `dt_s`. |

## Execução

Instale as dependências em um ambiente virtual. O `Makefile` carrega `.env` para definir a configuração e os diretórios locais; `.env.example` é a referência versionada. O arquivo `.env` não deve conter credenciais e é ignorado pelo Git.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
make help
make all
```

Os alvos podem ser executados individualmente ou agrupados:

```bash
make download      # apenas os HDF5 do Drive
make acquire       # download e inventário HDF5
make process       # extração, reconstrução e métricas
make plot          # figuras a partir das reconstruções existentes
make analyze       # métricas e figuras
make all CONFIG=config/pilot.json
```

`CONFIG=...` na linha de comando substitui somente para aquela execução o valor definido no `.env`. Também podem ser sobrescritos `RAW_DIR`, `SAMPLES_DIR`, `RECON_DIR`, `FIGURES_DIR` e `FIDELITY_OUTPUT`.

Para adquirir os arquivos sem Make, use `acquisition/download_selected_files.py`; ele usa os IDs definidos na configuração e `gdown`. Se a autenticação do Drive impedir o download automático, baixe os arquivos selecionados pela configuração para `data/raw/` e execute as etapas seguintes.

```bash
.venv/bin/python acquisition/download_selected_files.py --config config/spatial_range_pilot.json
.venv/bin/python acquisition/inventory_hdf5.py --config config/spatial_range_pilot.json
.venv/bin/python acquisition/extract_febus_windows.py --config config/spatial_range_pilot.json
.venv/bin/python compression/encode_decode_samples.py --config config/spatial_range_pilot.json
.venv/bin/python compression/compute_fidelity_metrics.py
.venv/bin/python presentation/plot_results.py --config config/spatial_range_pilot.json
```

### Comparação offshore original × FEBUS 4× nos dois HDF5 locais

```bash
.venv/bin/python compression/compare_raw_inference.py \
  --output-dir results/inference_results/offshore_vs_4x_new_run
```

O diretório de saída deve ser novo. O runner usa
`config/offshore_vs_4x_raw.json`, extrai os recortes históricos de 3 s/2 s e compara
o offshore original com o checkpoint 4× de melhor RMSE macro do fold 0 (época 40).
Não há treinamento ou seleção por arquivo. Canais inteiramente não finitos são
removidos igualmente, com mapa original registrado; ausência parcial ou mapa
remanescente descontínuo interrompem essa execução.

A inferência restaura a geometria e a normalização registradas no checkpoint,
com chunks comuns de 1.000 amostras. Gera reconstruções, métricas, diagnósticos
individuais, figuras pareadas e hashes. `comparison.csv` distingue o RMSE relativo
ao RMS bruto do RMSE centrado e das caudas por canal. A compressão nominal cobra
índices e side, não o tamanho dos NPZ de diagnóstico nem todos os custos de um
codec implantado. A comparação não é à mesma taxa nem validação externa independente.

Esse runner preserva a seleção histórica (`none`) por padrão. Para repetir a
comparação com o filtro espacial novo, use um diretório novo e a opção explícita:

```bash
.venv/bin/python compression/compare_raw_inference.py \
  --spatial-band-policy persistent-amplitude-v1 \
  --output-dir results/inference_results/offshore_vs_4x_filtered_new_run
```

## Fine-Tuning FEBUS

### Seleção espacial antes da normalização

Novas execuções de `acquisition/extract_febus_windows.py` e
`finetuning/prepare_finetune_samples.py` usam por padrão
`persistent-amplitude-v1`: amplitude robusta por canal, excesso de 8× a mediana
espacial em pelo menos 80% das janelas observadas, mínimo de quatro janelas e
dois canais adjacentes. Há proteção contra perda de mais de 25% dos canais
observados. Picos isolados não determinam a exclusão. A máscara é uma heurística
de cobertura, não um indicador físico validado.

O filtro atua antes da normalização, não modifica os HDF5 e não altera valores
dos canais retidos. Regiões separadas são salvas/processadas separadamente, com
mapas físicos. A preparação de treino divide regiões longas em tiles sem
descartar o restante por crop central. Use diretórios novos; arrays preparados
antigos não são filtrados automaticamente. Para reproduzir a preparação legada,
adicione `--spatial-band-policy none`.

```bash
.venv/bin/python acquisition/extract_febus_windows.py \
  --config config/spatial_range_pilot.json \
  --output-dir data/new_spatial_samples \
  --manifest-output results/manifests/new_spatial_samples.json

.venv/bin/python finetuning/prepare_finetune_samples.py \
  --output-dir data/finetune/spatial_band_v1_fold_0
```

Checkpoints novos registram a política; mudar cobertura exige um ramo novo de
adaptação, não resume silencioso do experimento antigo. A regra funciona offline,
com estatísticas do próprio arquivo; não garante preservação de eventos fortes
que persistam por quase toda a gravação. Ver
[relatório 61](progress_reports/61-PERSISTENT_SPATIAL_BAND_FILTER.md).

O corpus espacial auditado e a comparação A/B/C em dois orçamentos de índices
estão no [relatório 62](progress_reports/62-SPATIAL_CORPUS_AND_CONTROLLED_NEURAL_COMPARISON.md).
O runner `finetuning/run_spatial_abc_study.py` registra protocolo, status, decisões
de extensão e avaliações completas; não acessa o teste para inferência nem promove
um modelo final automaticamente.

O fine-tuning fica inteiramente neste repositório e reutiliza somente a arquitetura de `smartgrids`. A cópia local imutável do checkpoint-base fica em `models/febus_finetune_base/best.pt`; checkpoints ajustados são gravados em `results/fine_tune_results/`. O manifesto do Drive inclui apenas HDF5 brutos `Copy of ...h5`; saídas `*_output.h5` e `*_fft_output.h5` são excluídas porque duplicam ou derivam dos brutos.

Os folds são agrupados por família e configuração. Todas as frequências de uma configuração, assim como os núcleos de uma mesma pasta multicore, ficam na mesma partição. O conteúdo concreto está em `data/manifests/febus_folds.csv`, gerado a partir do inventário Drive; `FINETUNE_FOLDS=4` controla apenas a quantidade de folds. O modelo aceita a geometria nativa de cada fibra; a preparação não preenche artificialmente fibras com menos de 512 canais.

```bash
make finetune-inventory
make finetune-folds
make finetune-download
make finetune-prepare FINETUNE_TEST_FOLD=0 FINETUNE_VAL_FOLD=1
make finetune-train FINETUNE_TEST_FOLD=0 FINETUNE_VAL_FOLD=1
make finetune-evaluate FINETUNE_TEST_FOLD=0
# Equivalent to the two commands above, in order:
make finetune-run FINETUNE_TEST_FOLD=0 FINETUNE_VAL_FOLD=1
```

Repita para os quatro valores de `FINETUNE_TEST_FOLD`, sempre escolhendo um `FINETUNE_VAL_FOLD` diferente. `finetune-download` pode transferir aproximadamente 12 GB de HDF5 brutos e o treino não é iniciado automaticamente pelos alvos anteriores.

Para continuar um experimento sem reiniciar pesos ou AdamW, use o último checkpoint salvo. `FINETUNE_EPOCHS` representa épocas adicionais quando `FINETUNE_RESUME` está definido:

```bash
make finetune-train FINETUNE_TEST_FOLD=0 FINETUNE_VAL_FOLD=1 FINETUNE_EPOCHS=10 \
  FINETUNE_RESUME=results/fine_tune_results/fold_0/last.pt
```

## Experimentos para reconstrução generalista FEBUS

O protocolo, resultados preliminares e sequência condicional até o treinamento final
estão em [FEBUS_GENERALIST_ROADMAP.md](progress_reports/FEBUS_GENERALIST_ROADMAP.md).
Os resultados completos de balanceamento e o diagnóstico espectral/quantização estão
em [relatório 3](progress_reports/3-SAMPLING_FULL_AND_RECONSTRUCTION_DIAGNOSIS.md).
O piloto de objetivo espectral e seleção macro está no
[relatório 4](progress_reports/4-SPECTRAL_OBJECTIVE_AND_MACRO_SELECTION_PILOT.md).
O teste de capacidade/memorização, restrito a janelas de treino, está no
[relatório 5](progress_reports/5-TRAIN_ONLY_CAPACITY_PROBE.md); seus pesos não são
modelos de produção nem especialistas por família.
O teste com capacidade compartilhada e controles/validação está no
[relatório 6](progress_reports/6-SHARED_CAPACITY_AND_TRANSFER_PROBE.md).
Cobertura, repetição entre folds e valores extremos que comprometem a normalização
estão no [relatório 7](progress_reports/7-CORPUS_COVERAGE_AND_NUMERIC_INTEGRITY_AUDIT.md).
Proteção reversível testada e comando do próximo piloto manual estão no
[relatório 8](progress_reports/8-REVERSIBLE_NUMERIC_PROTECTION_AND_MANUAL_PILOT.md).
Resultados do piloto e protocolo do treino longo em background estão no
[relatório 9](progress_reports/9-NUMERIC_PILOT_RESULTS_AND_LONG_PRESERVATION_EXPERIMENT.md).
Resultados do ajuste completo/decoder-only estão no
[relatório 10](progress_reports/10-LONG_PRESERVATION_RESULTS.md); auditoria e piloto de
representação por canal, no [relatório 11](progress_reports/11-CHANNEL_REPRESENTATION_AUDIT_AND_PILOT.md).
Rastreabilidade e verificação dos canais selecionados estão no
[relatório 12](progress_reports/12-SPATIAL_CHANNEL_MAPS_AND_GEOMETRY_INTEGRITY.md).
Resultados da normalização treinada e validação ampliada dos candidatos estão no
[relatório 13](progress_reports/13-CHANNEL_PILOT_RESULTS_AND_EXPANDED_VALIDATION.md).
Validação completa da normalização e controle clássico de taxa:
[relatório 14](progress_reports/14-FULL_NORMALIZATION_VALIDATION_AND_RATE_CONTROL.md).
Resultados do controle DCT estão no [relatório 15](progress_reports/15-SPARSE_DCT_RATE_DISTORTION_RESULTS.md);
ablação neural de densidade temporal, no [relatório 16](progress_reports/16-TEMPORAL_LATENT_DENSITY_PILOT.md).
Confirmação na validação completa e continuação longa 2x/4x:
[relatório 17](progress_reports/17-TEMPORAL_DENSITY_FULL_RESULTS_AND_LONG_CONTINUATION.md).
Resultados da continuação e próximos critérios de decisão:
[relatório 18](progress_reports/18-TEMPORAL_DENSITY_LONG_RESULTS_AND_DECISION_GATES.md).
Auditoria completa e comparação DCT com teto pareado:
[relatório 19](progress_reports/19-TEMPORAL_AUDIT_AND_MATCHED_DCT_RESULTS.md).
Interpretação da fase com pesos fixados pelo alvo:
[relatório 20](progress_reports/20-TARGET_WEIGHTED_PHASE_DIAGNOSIS.md).
Comparação controlada de arquitetura e treino do zero:
[relatório 21](progress_reports/21-ARCHITECTURE_AND_SCRATCH_TRAINING_PROTOCOL.md).
Resultados de arquitetura/inicialização e avaliação completa:
[relatório 22](progress_reports/22-ARCHITECTURE_INITIALIZATION_RESULTS.md).
Continuação controlada de orçamento dos braços treinados do zero:
[relatório 23](progress_reports/23-SCRATCH_BUDGET_CONTINUATION_PROTOCOL.md).
Resultados da continuação e diagnóstico da quantização:
[relatório 24](progress_reports/24-SCRATCH_CONTINUATION_AND_QUANTIZATION_DIAGNOSIS.md).
Ablação pareada de reativação de códigos:
[relatório 25](progress_reports/25-PAIRED_CODEBOOK_REFRESH_PROTOCOL.md).
Resultado da reativação e limites generalistas:
[relatório 26](progress_reports/26-CODEBOOK_REFRESH_RESULTS.md).
Especialistas por família com controle compartilhado de exposição:
[relatório 27](progress_reports/27-DOMAIN_SPECIALISTS_AND_BALANCED_CONTROL_PROTOCOL.md).
Resultados da especialização:
[relatório 28](progress_reports/28-DOMAIN_SPECIALIZATION_RESULTS.md).
Comparação de alocação espacial/temporal de índices:
[relatório 29](progress_reports/29-SPATIOTEMPORAL_RATE_ALLOCATION_PROTOCOL.md).
Resultados da alocação:
[relatório 30](progress_reports/30-RATE_ALLOCATION_RESULTS.md).
Confirmação em fold/configurações e seed adicionais:
[relatório 31](progress_reports/31-RATE_ALLOCATION_FOLD_CONFIRMATION_PROTOCOL.md).
Resultados entre folds e diagnóstico de faixa dinâmica residual:
[relatório 32](progress_reports/32-FOLD_CONFIRMATION_AND_RESIDUAL_DYNAMIC_RANGE.md).
Auditoria de proteção adaptativa e comparação congelada:
[relatório 33](progress_reports/33-DYNAMIC_RANGE_AUDIT_AND_FROZEN_COMPARISON.md).
Resultado do modelo congelado sob proteção adaptativa:
[relatório 34](progress_reports/34-FROZEN_DYNAMIC_RANGE_RESULTS.md).
Conclusão da cobertura numérica:
[relatório 35](progress_reports/35-DYNAMIC_RANGE_COVERAGE_CONCLUSION.md).
Curva neural de taxas maiores:
[relatório 36](progress_reports/36-NEURAL_RATE_DISTORTION_SWEEP_PROTOCOL.md).
Resultados da curva neural e controle DCT pareado:
[relatório 37](progress_reports/37-NEURAL_RATE_DISTORTION_RESULTS.md).
Continuação limitada do candidato 4x:
[relatório 38](progress_reports/38-HIGH_RATE_CONTINUATION_PROTOCOL.md).
Resultados da continuação:
[relatório 39](progress_reports/39-HIGH_RATE_CONTINUATION_RESULTS.md).
Confirmação da curva de taxas no fold 0, com referência robusta:
[relatório 40](progress_reports/40-HIGH_RATE_FOLD0_ROBUST_PROTOCOL.md).
Resultados completos e continuação limitada 2x/4x:
[relatório 41](progress_reports/41-ROBUST_RATE_FOLD0_RESULTS.md) e
[relatório 42](progress_reports/42-ROBUST_RATE_CONTINUATION_PROTOCOL.md).
Auditoria pareada da continuação e piloto congelado de estabilidade:
[relatório 43](progress_reports/43-ROBUST_CONTINUATION_RESULTS_AND_PAIRED_AUDIT.md) e
[relatório 44](progress_reports/44-FROZEN_RATE_STABILITY_PROTOCOL.md).
Resultados do piloto congelado:
[relatório 45](progress_reports/45-FROZEN_RATE_STABILITY_RESULTS.md).
Diagnóstico de representatividade e seleção por configuração:
[relatório 46](progress_reports/46-CHECKPOINT_SELECTION_AUDIT.md).
Protocolo de calibração global train-only com rede congelada:
[relatório 47](progress_reports/47-TRAIN_ONLY_GLOBAL_GAIN_PROTOCOL.md).
Resultado da calibração e diagnóstico congelado de quantização:
[relatório 48](progress_reports/48-TRAIN_ONLY_GAIN_RESULTS.md) e
[relatório 49](progress_reports/49-FROZEN_QUANTIZATION_PROBE_PROTOCOL.md).
Resultado do bypass e controle de energia latente:
[relatório 50](progress_reports/50-FROZEN_QUANTIZATION_RESULTS.md) e
[relatório 51](progress_reports/51-NORM_MATCHED_LATENT_CONTROL_PROTOCOL.md).
Resultado da energia latente e piloto full/decoder com guarda:
[relatório 52](progress_reports/52-NORM_MATCHED_LATENT_RESULTS.md) e
[relatório 53](progress_reports/53-GUARDED_DECODER_PILOT_PROTOCOL.md).
Resultados do piloto e objetivo por canal com pesos limitados:
[relatório 54](progress_reports/54-GUARDED_DECODER_PILOT_RESULTS.md) e
[relatório 55](progress_reports/55-BOUNDED_CHANNEL_OBJECTIVE_PROTOCOL.md).
Resultado do objetivo por canal e confirmação entre configurações:
[relatório 56](progress_reports/56-BOUNDED_CHANNEL_OBJECTIVE_RESULTS.md) e
[relatório 57](progress_reports/57-CHANNEL_OBJECTIVE_FOLD_CONFIRMATION_PROTOCOL.md).
Confirmação e registro consolidado das referências de pesquisa:
[relatório 58](progress_reports/58-CHANNEL_OBJECTIVE_FOLD_CONFIRMATION_RESULTS.md) e
[relatório 59](progress_reports/59-GENERALIST_RESEARCH_REFERENCE_REGISTRY.md).
Limpeza conservadora de checkpoints redundantes, sem perda de caminhos/dados:


```bash
# Comparação pareada nos quatro folds, com saídas isoladas (diretório deve ser novo):
.venv/bin/python finetuning/run_sampling_experiment.py --output-dir results/experiments/sampling_new
# Teste rápido, apenas na validação:
.venv/bin/python finetuning/run_sampling_experiment.py --pilot --output-dir results/experiments/sampling_pilot_new
# Retomar uma sequência interrompida (preserva checkpoints e logs):
.venv/bin/python finetuning/run_sampling_experiment.py --output-dir results/experiments/sampling_new --resume
# Adicione --dry-run ao comando acima para apenas verificar o plano.
# Treino individual balanceado sem sobrescrever o baseline:
make finetune-train FINETUNE_TEST_FOLD=3 FINETUNE_VAL_FOLD=0 \
  FINETUNE_SAMPLING=family-group FINETUNE_OUTPUT_DIR=results/experiments/balanced/fold_3
```

`FINETUNE_STEPS_PER_EPOCH=0` mantém uma atualização por janela de treino como
orçamento; balanceamento sorteia com reposição. O runner compara `window` e
`family-group` a partir do mesmo checkpoint-base e gera `REPORT.md` ao concluir.
O runner exibe a etapa da sequência, a barra de treino (época, janelas, tempo/ETA,
loss) e métricas por época diretamente no terminal, salvando a mesma saída nos logs.
Execute sem redirecionar stdout/stderr para ver o progresso; `--quiet` omite os
detalhes no terminal, mas não nos logs. Na avaliação, o progresso aparece a cada
500 janelas. As 13 etapas são contagem de tarefas, não percentual de tempo total.

## Artefatos gráficos


Para cada arquivo processado, `presentation/plot_results.py` gera:

- `*_waterfall_comparison.png`: waterfall original, reconstruído e erro;
- `*_event_waveform_comparison.png`: traço temporal no canal e instante de maior amplitude finita;
- `*_spatial_diagnostics.png`: RMS, ruído robusto e SNR versus distância;
- `*_fft_frequency_distance_comparison.png`: FFT 2D original, reconstruído e do erro;
- `*_noise_spectrum_comparison.png`: espectro de ruído do canal central nos 20% instantes de menor energia.

A tabela `reconstruction_metrics_summary.png` é gerada para consolidar os arquivos processados. A FFT 2D usa distância no eixo horizontal e frequência temporal no vertical; `dt_s` vem dos atributos FEBUS quando disponível. O SNR por distância é uma métrica diagnóstica calculada como RMS dividido pelo desvio padrão dos 20% instantes de menor amplitude, não uma reprodução exata do SNR de eventos do processamento original.

## Interpretação

As métricas devem ser reportadas separadamente para `Strain` e `Strain Rate`, para tipo de fibra, frequência, canal e orientação. A razão de compressão teórica dos códigos não equivale automaticamente à razão prática de arquivos: o relatório deve distinguir índices, metadados e custo compartilhado do codebook/checkpoint.

Este é um teste de transferência de domínio e fidelidade de reconstrução. Ele não valida desempenho sismológico, não testa detecção de eventos e não demonstra que o VQ-VAE é adequado para operação geral em dados FEBUS.
