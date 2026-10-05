# Diagnóstico da Baixa Fidelidade do VQ-VAE em Dados DAS do Drive

## Escopo e restrição

Este diagnóstico avalia exclusivamente o checkpoint `best.pt` já existente em `/home/murilo-barroso/source/smartgrids/runs/vqvae_512ch/best.pt`. Nenhum modelo foi treinado, ajustado ou teve pesos modificados. Todas as evidências novas foram obtidas por inferência local em cópias de HDF5 ou arrays locais. Nenhum arquivo do Google Drive foi alterado.

O resultado ruim a explicar é a reconstrução da amostra `smf_configuration_1__200Hz` extraída do Drive: correlação `0,2089`, RMSE relativo `0,9829` e MAE `321470` no resultado já produzido pela pipeline. As métricas estão em `results/metrics/per_sample_metrics.csv` e `results/metrics/fidelity_metrics.csv`.

## O que o checkpoint aprendeu

O checkpoint foi treinado no repositório `smartgrids` sobre dados DAS OOI RCA 2024, produto `dphi`, não sobre os HDF5 FEBUS do Drive.

| Item | Treinamento / holdout OOI | DAS FEBUS avaliado |
| --- | --- | --- |
| Variável | `dphi`, unidade `rad/(s*m)` | `Strain Rate [nStrain/s]` |
| Geometria de treino | blocos contíguos `512×1000` | até `551×60000`, processado em chunks `551×1024` |
| Escala temporal | `dt=0,005 s`, 200 Hz; crop de 5 s | `dt=0,00005 s`, 20 kHz; 3 s totais |
| Escala espacial | `dx≈1,0213 m`; cerca de 523 m por bloco | `dx=0,2 m`; 512 canais cobrem 102,4 m |
| Normalização | z-score por amostra | z-score por chunk temporal finito |
| Arquitetura | CNN 2D, quatro downsamplings, codebook de 2048 vetores de dimensão 128 | mesma arquitetura, sem ajuste |

As fontes são `smartgrids/paper/methodology.tex`, `smartgrids/data/prepare_training_data.py`, `smartgrids/vqvae/train_vqvae.py` e `smartgrids/runs/vqvae_512ch/config.json`. O modelo aceita dimensões arbitrárias por padding até múltiplos de 16, mas isso não torna a distribuição física e visual compatível.

No holdout OOI de 2.442 amostras, registrado em `smartgrids/runs/vqvae_eval/results/per_sample_metrics.csv`, as médias foram correlação `0,9017`, RMSE relativo `0,3861`, SNR `9,09 dB`, perplexidade `778,3` e 950 códigos únicos por amostra. Portanto, o checkpoint consegue reconstruir razoavelmente bem dados do seu domínio de origem, embora já perca relativamente mais potência em 10-25 Hz (`0,6199` de erro relativo de potência na média do holdout).

## Testes de inferência realizados

Os testes abaixo usaram sempre o mesmo checkpoint, CPU, z-score por amostra salvo quando indicado e nenhuma atualização de pesos. A amostra FEBUS principal foi limitada aos primeiros 512 canais totalmente finitos. O resultado completo do experimento foi mantido apenas temporariamente durante a análise.

| Condição | Forma | Correlação | RMSE relativo | SNR (dB) | Perplexidade | Leitura |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| FEBUS, evento, forma canônica | `512×1000` | 0,169 | 0,988 | 0,11 | 46,9 | A forma de treino, isoladamente, não recupera a fidelidade. |
| FEBUS, evento, chunk da pipeline | `512×1024` | 0,164 | 0,989 | 0,09 | 44,1 | Diferença de 24 colunas/padding não explica o problema. |
| FEBUS, trecho de menor energia | `512×1000` | 0,161 | 0,989 | 0,09 | 43,5 | A falha não é exclusiva do evento de maior amplitude. |
| FEBUS, normalização por canal | `512×1000` | 0,174 | 0,985 | 0,13 | 125,1 | Mudança de normalização tem ganho desprezível e não resolve a transferência. |
| FEBUS, decimado de 20 kHz para 200 Hz | `512×600` | 0,160 | 0,991 | 0,08 | 36,7 | Decimar sem restaurar a largura esperada não melhora. |
| FEBUS, decimado e reescalado para 1000 colunas | `512×1000` | 0,228 | 0,974 | 0,23 | 39,2 | Há melhora modesta da escala temporal em pixels, mas continua muito distante do holdout. |
| FEBUS, um canal a cada cinco | `103×1000` | 0,150 | 0,991 | 0,08 | 101,2 | Aproximar `dx≈1 m` reduz altura; o teste é inconclusivo para isolar espaço. |
| FEBUS, mesmos 180 canais do controle externo | `180×1000` | 0,100 | 1,003 | -0,02 | 64,7 | Menor altura não é uma explicação suficiente nem uma correção. |
| FEBUS i29 do Drive, arquivo rotulado 1000 Hz | `180×1000` | 0,295 | 0,955 | 0,40 | 217,1 | Outra origem Drive melhora, mas continua muito fora do domínio OOI. |
| Evento OOI Table 1 existente no `smartgrids` | `15×3001` | 0,250 | 1,216 | -1,70 | 142,8 | Confirma que eventos e geometria não treinada já degradavam fortemente no próprio ecossistema OOI. |

O arquivo FEBUS i29 tem dataset `Strain Rate [nStrain|s]`, quatro blocos `21000×51`, `dx=0,2 m` e `dt=0,00005 s`. Ele foi baixado somente para leitura. O controle OOI é consistente com o experimento consolidado de 32 eventos em `smartgrids/runs/false_positive_filter/vqvae_reconstruction/reconstruction_summary.json`: correlação média `0,3585`, RMSE relativo `1,1256` e SNR médio `-1,01 dB`.

## Hipóteses avaliadas

### Com forte evidência

1. **Mudança de domínio entre OOI `dphi` e FEBUS `Strain Rate`.** O modelo foi treinado em uma variável, unidade, instrumentação e distribuição diferentes. Mesmo impondo a forma de treino ao FEBUS (`512×1000`), a correlação é `0,169`, contra `0,902` no holdout OOI. O arquivo i29 independente melhora para `0,295`, mas permanece ruim. Isso mostra que não é apenas um erro de extração de uma única janela.

2. **Escala temporal incompatível.** O treino vê 1000 pixels em 5 s a 200 Hz; o FEBUS tem 1000 pixels em 0,05 s a 20 kHz. Decimar e reescalar o FEBUS para 1000 colunas aumentou a correlação de `0,169` para `0,228`. A melhora é pequena, mas demonstra que a escala temporal é um fator causal parcial.

3. **Uso empobrecido do codebook fora do domínio.** A perplexidade do holdout é `778,3` em média; no FEBUS ela ficou entre `36,7` e `125,1`, e no i29 em `217,1`. Isso indica que a entrada aciona uma região muito menor do codebook. É uma evidência de deslocamento da representação, não prova isolada de que o codebook seja a única causa.

4. **O problema não é específico de eventos.** O trecho FEBUS de menor energia (`corr=0,161`) teve resultado praticamente igual ao trecho centrado no máximo (`0,169`). Eventos centrados são um domínio adicionalmente difícil, mas o sinal FEBUS geral já está fora da distribuição aprendida.

### Com evidência parcial ou inconclusiva

1. **Geometria espacial e abertura física.** O modelo aprendeu correlações locais em 512 canais contíguos com `dx≈1,02 m`, enquanto 512 canais FEBUS cobrem somente 102,4 m a `dx=0,2 m`. Subamostrar um canal a cada cinco aproximou o espaçamento, mas reduziu a altura para 103, e piorou a correlação para `0,150`. Como o número de canais e o contexto espacial mudaram simultaneamente, o teste não isola o efeito de `dx`. A hipótese permanece tecnicamente forte, mas não foi comprovada de forma causal.

2. **Duração e contexto temporal.** O treino usou crops de 5 s; a execução FEBUS concatena 3 s e infere em chunks de 1024 amostras. O teste `512×1000` versus `512×1024` praticamente empatou, refutando padding como causa principal. Ainda não foi possível criar, a partir do FEBUS disponível, uma janela fisicamente equivalente de 5 s a 200 Hz sem reamostragem e sem alterar ao mesmo tempo conteúdo/escalas.

3. **Normalização.** A normalização por chunk da pipeline é próxima da normalização por amostra do treino. A normalização por canal mudou `corr` de `0,169` para `0,174`, insuficiente para explicar ou corrigir a perda. Diferenças estatísticas de normalização podem contribuir, mas não são dominantes nos dados testados.

4. **Capacidade e objetivo de compressão.** Quatro downsamplings e quantização vetorial priorizam uma representação compacta, não preservação perfeita. Mesmo no holdout há perda elevada em 10-25 Hz. Isso provavelmente amplifica a perda em sinais FEBUS de alta taxa temporal, mas não pode ser separado de mudança de domínio com inferência apenas.

## O que não foi testado e por quê

| Questão | Motivo para não testar | O que seria necessário |
| --- | --- | --- |
| Adaptar pesos ao FEBUS | A restrição desta análise proíbe retreinar ou fazer fine-tuning. | Treino/fine-tuning em blocos FEBUS representativos, com holdout por arquivo. |
| Mudar arquitetura, níveis de downsampling ou codebook | Requer novos pesos; inferir com arquitetura diferente é inválido. | Treinar ablações com menor downsampling, representação multi-escala ou maior capacidade. |
| Isolar somente `dx` mantendo 512 canais e a mesma abertura física | O FEBUS disponível possui cerca de 110 m úteis, não os cerca de 523 m de um bloco OOI treinado. | Cabo/dados FEBUS com abertura maior, ou dados OOI reamostrados com controle de geometria. |
| Comparar diretamente com blocos brutos de treino | Os arrays/HDF5 de treino não estão presentes, apenas manifests e resultados. | Restaurar `training_data/raw_data` e `processed_samples_test`, depois repetir a inferência com a mesma rotina diagnóstica. |
| Avaliar todas as frequências, fibras e configurações do Drive | A análise baixou somente o i29 adicional para limitar transferência e tempo. | Executar a mesma matriz de testes sobre 10000 Hz, SMF dopada e multimodo, reportando métricas por origem. |
| Medir impacto sismológico/detecção no FEBUS | Não há rótulos/eventos equivalentes nem pipeline downstream FEBUS validada. | Dados FEBUS rotulados e métricas de detecção/phase picking além de MSE e correlação. |

## Conclusão

A explicação mais sustentada é **transferência de domínio insuficiente**, composta por mudança de variável/instrumentação, escala temporal de 20 kHz versus 200 Hz e geometria espacial muito diferente. Formato, padding e normalização foram testados e não explicam a queda. Reamostrar o eixo temporal melhora marginalmente, mas não restaura a reconstrução, e o forte colapso de perplexidade indica que o codebook não representa bem os padrões FEBUS.

Logo, não é defensável interpretar a taxa teórica de compressão de aproximadamente 504x do FEBUS como compressão com fidelidade útil. Ela mede apenas o tamanho dos índices produzidos; as métricas de reconstrução mostram que esses índices não preservam adequadamente o sinal para esse domínio. A próxima verificação tecnicamente válida, fora da restrição atual, seria um fine-tuning ou novo treino com blocos FEBUS representativos, com `dt`, `dx`, duração e objetivos espectrais compatíveis com a aplicação desejada.
