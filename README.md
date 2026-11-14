# Compressão neural de dados DAS FEBUS

Projeto de pesquisa para desenvolver um sistema **neural de compressão com alta
fidelidade de reconstrução** de dados de *Distributed Acoustic Sensing* (DAS)
adquiridos com interrogadores FEBUS.

O trabalho começou avaliando a transferência de um VQ-VAE treinado em dados
offshore OOI, do repositório [`smartgrids`](../smartgrids). Evoluiu para uma
infraestrutura de aquisição, preparação, treinamento, avaliação e codificação
experimental, buscando melhorar a **fidelidade por bit transmitido** em diferentes
fibras e configurações.

**Estado atual:** há modelos e codecs experimentais, com ganhos de reconstrução
e limitações documentadas. Ainda não há um modelo generalista final validado para
operação. O [roadmap](progress_reports/FEBUS_GENERALIST_ROADMAP.md) organiza as
prioridades; a revisão de direção está no [relatório 77](progress_reports/77-RESEARCH_DIRECTION_REVIEW.md)
e os resultados de representação no [relatório 79](progress_reports/79-ENCODER_RATE_AND_TILE_WEIGHTING.md).
A revisão de LR/fontes externas está no
[relatório 80](progress_reports/80-LEARNING_RATE_AND_EXTERNAL_PRETRAINING_REVIEW.md);
o contraste pareado de scheduler foi concluído no
[relatório 81](progress_reports/81-LR_SCHEDULE_PAIRED_RESULTS.md), sem promoção.
A ponte de pré-treinamento em fase QuantX e seus controles estão no
[relatório 82](progress_reports/82-QUANTX_PRETRAINING_BRIDGE_RESULTS.md), também sem promoção.
O [relatório 83](progress_reports/83-POROTOMO_PRETRAINING_BRIDGE_RESULTS.md) registra
a transferência condicional de PoroTomo, sem vantagem sobre investir no alvo;
o [relatório 84](progress_reports/84-FIXED_ID_SYNTHESIS_AND_CONTEXT_PROBES.md) encerra
os probes de síntese/contexto por ganhos insuficientes. A ablação de objetivo
MSE versus mistura MSE/L1 concluiu com ganho pequeno e mais bytes no
[relatório 85](progress_reports/85-OBJECTIVE_ALIGNMENT_ABLATION.md).
O diagnóstico de probabilidades condicionais dos coeficientes escalares concluiu
sem economia suficiente, mantendo reconstrução bit-exata
([relatório 86](progress_reports/86-SCALAR_CONDITIONAL_RATE_DIAGNOSTIC.md)).
O probe de contexto **antes da quantização** também encerrou abaixo do limiar de
avanço ([relatório 87](progress_reports/87-CONTEXT_BEFORE_QUANTIZATION_PROBE.md)).
O contraste de encoder/codebook/decoder e contexto conjuntamente também concluiu
sem avanço: atenção axial não superou pointwise em fidelidade
([relatório 88](progress_reports/88-JOINT_ANALYSIS_SYNTHESIS_CONTEXT_PROTOCOL.md)).
O piloto de produto VQ treinado conjuntamente encerrou com gate reprovado e uma
limitação auditada do seletor em alvo constante; **o treino longo não começou**.
Os resultados e a auditoria estão no
[relatório 89](progress_reports/89-JOINT_PRODUCT_VQ_LONG_TRAINING_PROTOCOL.md).

## Dados e objetivo de fidelidade

O corpus auditado reúne **207 HDF5 e 23 grupos/configurações** da pasta
compartilhada `DAS` no Google Drive, abrangendo quatro famílias:

- single-mode;
- doped;
- multicore;
- multimode.

O alvo é a matriz física **canal × tempo**, incluindo o ruído presente no sinal
original. `Strain` e `Strain Rate` são tratados conforme a variável armazenada,
sem conversão entre elas. Frequências nos nomes dos arquivos identificam os
ensaios; a taxa de amostragem vem dos metadados HDF5. As janelas FEBUS do corpus
atual têm 1.000 amostras a 20 kHz, equivalentes a 50 ms.

A preparação espacial auditada contém 28.780 janelas distribuídas entre as
partições. Configurações e núcleos relacionados permanecem na mesma partição;
janelas da mesma aquisição não equivalem a observações independentes.

Os brutos são preservados. A seleção espacial `persistent-amplitude-v1` identifica
faixas de amplitude persistentemente excessiva antes da normalização, com mapas
físicos e cobertura registrados. É uma heurística, não uma classificação física
definitiva de canais. Regiões separadas são processadas em tiles próprios, e a
proteção numérica reversível preserva valores extremos em informação auxiliar.

## Fluxo experimental

```text
Drive / HDF5 FEBUS
  → inventário e partições por configuração
  → seleção espacial, tiles e mapas físicos
  → normalização e proteção numérica
  → encoder neural e quantização
  → símbolos codificados + informação auxiliar
  → decoder e restauração da escala física
  → avaliação de fidelidade, bytes e custo computacional
```

A avaliação acompanha RMSE, correlação, amplitude, fase, espectro e erros por
canal, família e configuração. A taxa efetiva considera símbolos, normalização,
escapes, mapas e container; pesos compartilhados são contabilizados separadamente.
Razão nominal de índices, tamanho de NPZ e compressão efetiva são medidas distintas.

## Principais aprendizados

- Adaptação e otimização melhoraram a reconstrução, inclusive à mesma taxa.
- Aumentar a densidade de índices trouxe ganhos fortes, pagando mais bits.
- A CNN ampliada de aproximadamente 28M para 96,5M parâmetros reduziu o RMSE macro
  em cerca de 1,5% no estudo compacto, com regressões localizadas.
- Especialização, outras losses, ROI e refinamento residual apresentaram ganhos
  limitados ou compromissos entre configurações e canais.
- Codificação lossless dos índices economizou bytes sem alterar a reconstrução;
  os ajustes mais recentes tiveram retorno adicional pequeno.
- Busca de índices com pesos congelados recuperou fidelidade nos pilotos;
  com as guardas históricas, a redução de RMSE ficou em cerca de **4–5%**,
  sem aumentar os frames. Em uma comparação adicional, proteger também amplitude,
  magnitude FFT e fase reduziu o ganho para **1,30%**: não há preservação física
  universal demonstrada pelo ganho de RMSE.
- Restauração afim transmitida, com decoder congelado, elevou o ganho fisicamente
  guardado para **3,70%** no desenvolvimento, mas a transferência ao grupo SMF
  externo consumido foi só **~0,19%**. Continua experimental, sem promoção.
- Seis braços escalares e quatro de adaptação do encoder VQ não atingiram o
  critério de investimento frente aos controles. Temperatura mais baixa melhorou
  o alinhamento entre proxy e bytes, mas o ganho adicional de RMSE com guarda foi só **0,115%**
  contra o controle do encoder. Nenhum modelo foi promovido.
- A continuação local também testou correção espacial da saída, orientação/padding
  e VQ com encoder/codebook/decoder treináveis. Os ganhos contra os controles
  permaneceram marginais; alinhar taxa estimada e bytes não bastou para mudar
  o patamar de fidelidade.
- O stress SMF preservou todos os canais e os bits dos NaNs. Preparação de entrada
  afetou bastante os resultados; melhorar apenas o alvo filtrado não demonstrou
  ganho de cobertura integral sob os mesmos bytes e requisitos físicos.
- A triagem de grade × cardinalidade também não gerou avanço admissível:
  denso K64 teve frames quase iguais ao compacto K2048, mas RMSE **5,54% pior**.
  Alfabetos menores economizaram bytes com perda de fidelidade; sem promoção.
  Quantização por produto/PCA e ajuste curto de centroides pelo decoder também
  não preservaram a fidelidade nos probes de treino (relatório 79, seção 11).
- O controle preditivo sem rede também não superou o VQ no mesmo teto de bytes.
  Seus pontos de maior precisão comprimiram 2,5–4,5×, em taxas diferentes; são
  referências experimentais, não avanço simultâneo de fidelidade/compressão.
- Fatores espaciais/temporais transmitidos melhoraram um recorte com menos bytes,
  mas o ganho guardado agregado foi só **0,174% de distorção**. A busca de postos
  não ampliou o ganho; segue como controle experimental, sem promoção.
- Cosine decay melhorou o RMSE em **0,16–0,32% sobre controles de LR constante**
  com exposição pareada, mas aumentou bytes e apresentou regressões por configuração.
  O resultado não justificou ampliar o ramo ou mudar o LR padrão (relatório 81).
- A ponte de fase QuantX teve **11,47% mais RMSE** que investir os mesmos updates
  diretamente em FEBUS, embora economize 4,16% dos bytes. Com dose FEBUS final
  igual, também não houve ganho macro; isso não descarta todo uso de QuantX ou
  de outras fontes, mas encerra essa continuidade bruta (relatório 82).

Nos dois conjuntos de validação dos estudos recentes, os streams compactos ficaram
em torno de **166–173×**, com correlação macro **0,73–0,75**; os densos, em torno de
**44–46×**, com correlação **0,89–0,90**. São modelos/coortes diferentes, com pesos
de reconstrução compartilhados e resultados de desenvolvimento. Esses agregados
não demonstram fidelidade suficiente de todos os canais nem validação externa.

### Próximas hipóteses

1. **Requisitos físicos:** definir o compromisso aceitável entre erro, amplitude,
   fase, espectro e piores canais, sem retunar guardas para favorecer um resultado.
2. **Generalização:** concluir a proveniência dos candidatos externos e reservar
   aquisições independentes; hashes e pastas diferentes não bastam.
3. **Representação/taxa:** escolher nova hipótese por mecanismo e evidência;
   não ampliar os corretores nem repetir as continuações inalteradas. Contexto
   temporal longo continua condicional aos diagnósticos.

Diagnósticos e pilotos de busca de índices/taxa–distorção já foram executados;
resultados e limitações estão nos relatórios [78](progress_reports/78-REMOTE_REPRESENTATION_PILOTS.md)
e [79](progress_reports/79-ENCODER_RATE_AND_TILE_WEIGHTING.md).
Novas execuções dependem de autorização e protocolo próprios. A meta de aproximadamente 700×
permanece exploratória.

## Estrutura do repositório

| Diretório | Conteúdo |
|---|---|
| `acquisition/` | Download, inventário e extração de HDF5 FEBUS. |
| `compression/` | Inferência, streams de símbolos e empacotamento. |
| `finetuning/` | Preparação, arquiteturas, treinamento, diagnósticos e estudos controlados. |
| `presentation/` | Gráficos e organização de resultados de inferência. |
| `config/` | Configurações de dados e execuções. |
| `data/` | Brutos locais, amostras preparadas, mapas e manifestos. |
| `models/` | Referências locais, incluindo o checkpoint-base offshore. |
| `results/` | Checkpoints, protocolos, métricas, streams e figuras dos experimentos. |
| `progress_reports/` | Histórico experimental, resultados e roadmap. |
| `tests/` | Verificações de preparação, modelos, métricas e codecs. |

A arquitetura VQ-VAE nativa é importada de `smartgrids/vqvae`; variantes e
componentes experimentais são implementados neste repositório. O checkpoint-base
local fica em `models/febus_finetune_base/best.pt`. Dados e pesos necessários às
execuções são artefatos locais e não acompanham necessariamente o clone do Git.

## Uso básico

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
make help
```

Consulte [`.env.example`](.env.example) para configurar caminhos, checkpoint e
dispositivo. O `Makefile` carrega `.env`; parâmetros na linha de comando têm
precedência. Ajuste também `repo_root` nas configurações e `FINETUNE_REPO_ROOT`
para a localização do repositório `smartgrids`. Estudos de codificação entrópica
podem exigir as dependências adicionais de `requirements-entropy.txt`.

Os alvos básicos atendem à pipeline de inferência sobre arquivos selecionados:

| Comando | Função |
|---|---|
| `make acquire` | Download e inventário. |
| `make process` | Extração e reconstrução com o checkpoint escolhido. |
| `make analyze` | Métricas, figuras e organização da execução em `results/inference_results/`. |
| `make all CONFIG=config/spatial_range_pilot.json` | Pipeline completa de aquisição e inferência, sem treinamento. |

Os alvos `finetune-*` e os runners em `finetuning/` atendem à preparação e aos
experimentos. `finetune-train` inicia treinamento; `finetune-run` também avalia o
fold de teste. Para reproduzir um estudo, siga seu relatório e protocolo, com
corpus, checkpoint, partições e diretório de saída próprios. Os defaults do piloto
não representam automaticamente o protocolo de pesquisa mais recente.

## Documentação de referência

- [Roadmap generalista FEBUS](progress_reports/FEBUS_GENERALIST_ROADMAP.md): objetivos, prioridades e critérios de decisão.
- [Revisão de direção — relatório 77](progress_reports/77-RESEARCH_DIRECTION_REVIEW.md): síntese crítica, literatura e próximos experimentos.
- [Pilotos remotos — relatório 78](progress_reports/78-REMOTE_REPRESENTATION_PILOTS.md): implementação, busca de índices e primeiros treinos taxa–distorção.
- [Encoder, taxa e tiling — relatório 79](progress_reports/79-ENCODER_RATE_AND_TILE_WEIGHTING.md): quatro braços VQ, guarda durante a busca e diagnóstico temporal contínuo.
- [Corpus espacial e comparação A/B/C — relatório 62](progress_reports/62-SPATIAL_CORPUS_AND_CONTROLLED_NEURAL_COMPARISON.md): preparação e protocolo controlado.
- [Confirmação A/B/C — relatório 64](progress_reports/64-CROSS_FOLD_CONFIRMATION_RESULTS.md): resultados entre configurações.
- [Expansão de capacidade — relatório 66](progress_reports/66-CAPACITY_COMPACT_RESULTS.md): comparação de redes a taxa fixa.
- [Histórico completo](progress_reports/): hipóteses, resultados positivos/negativos, comandos e artefatos de cada estudo.
