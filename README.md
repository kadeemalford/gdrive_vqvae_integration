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

## Fine-Tuning FEBUS

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
estão em [FEBUS_GENERALIST_ROADMAP.md](FEBUS_GENERALIST_ROADMAP.md).
O balanceamento ainda é experimental; o padrão continua sendo `window`.

```bash
# Comparação pareada nos quatro folds, com saídas isoladas (diretório deve ser novo):
.venv/bin/python finetuning/run_sampling_experiment.py --output-dir results/experiments/sampling_new
# Teste rápido, apenas na validação:
.venv/bin/python finetuning/run_sampling_experiment.py --pilot --output-dir results/experiments/sampling_pilot_new
# Treino individual balanceado sem sobrescrever o baseline:
make finetune-train FINETUNE_TEST_FOLD=3 FINETUNE_VAL_FOLD=0 \
  FINETUNE_SAMPLING=family-group FINETUNE_OUTPUT_DIR=results/experiments/balanced/fold_3
```

`FINETUNE_STEPS_PER_EPOCH=0` mantém uma atualização por janela de treino como
orçamento; balanceamento sorteia com reposição. O runner compara `window` e
`family-group` a partir do mesmo checkpoint-base e gera `REPORT.md` ao concluir.

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
