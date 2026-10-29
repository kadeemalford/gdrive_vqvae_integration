# Seleção espacial persistente antes de treino e inferência

## Escopo e premissa

A pedido do responsável pelo projeto, adotou-se como premissa operacional que
as faixas de amplitude exorbitante representam ponta/reflexão ou canais além
da fibra útil. Como não existe mapa físico validado, foi implementada uma
**heurística conservadora de seleção espacial**, não uma calibração física.
Ela não prova a origem de cada faixa nem garante ausência de falsos positivos.

Código compartilhado: `finetuning/spatial_band_filter.py`, política versionada
`persistent-amplitude-v1`. Nenhum HDF5 foi alterado. A implementação e a auditoria
operacional iniciais não executaram treino ou forward neural; posteriormente,
a validação por inferência abaixo utilizou checkpoints congelados, sem treino.

## Regra inicial fixa

1. Para cada arquivo, amostrar até **oito janelas não sobrepostas de 1.000
   amostras por bloco**, distribuídas por sua duração. Não usa métricas da rede,
   rótulos ou estatísticas estimadas de outros arquivos.
2. Em cada janela/canal finito, amplitude robusta = máximo entre
   `1,4826 × mediana(|x − mediana(x)|)` e `|mediana(x)|`.
   Isso inclui tanto variação persistente como deslocamento elevado de nível,
   sem deixar que poucos picos dominem a estatística.
3. Referência espacial = mediana das amplitudes dos canais finitos da janela.
   Excesso = amplitude **maior que oito vezes** essa referência.
4. Exigir excesso em **pelo menos 80% das observações elegíveis** do canal,
   com **mínimo de quatro janelas**, e **dois canais adjacentes** candidatos.
5. Não dilatar a máscara nem remover os canais vizinhos por associação.
   Sem referência utilizável, sem evidência temporal suficiente ou com proposta
   de remover mais de **25% dos canais observados**, abster-se da rejeição por
   amplitude e registrar o motivo. Canais não finitos são tratados separadamente.

A decisão é por arquivo, compartilhada entre seus blocos e janelas, e registrada
com parâmetros, votos, coordenadas e hash de identificação. A implementação é
offline: usa amostras do próprio arquivo antes da codificação. Uma versão
streaming exigiria buffer/calibração e política de atualização próprios.

**Garantia estrutural:** os valores retidos não sofrem clipping, filtragem,
interpolação ou alteração numérica por esta etapa. **Não há garantia física** de
que toda faixa selecionada seja inválida: um evento válido, localizado e forte
durante grande parte da gravação pode ser indistinguível sem informação externa.
Faixas moderadas, isoladas em um canal ou com persistência insuficiente permanecem.

## Integração

- `finetuning/prepare_finetune_samples.py`: política ativa por padrão em **novas
  preparações**, antes de normalização e recorte temporal. Cada região contígua
  válida é processada separadamente e, se necessário, dividida em tiles de até
  `--max-channels`, sem descartar o restante por crop central. Salva
  `preprocessing.json` e mapas de canais. O tiling também muda a exposição de
  treino, portanto deve ser controlado entre os braços experimentais.
- `acquisition/extract_febus_windows.py`: mesma política por padrão em novas
  extrações para inferência; emite um NPZ por região contígua. Não une artificialmente
  lados de uma faixa excluída. Mantém posições físicas e listas de canais fonte.
- `finetuning/finetune_vqvae.py`: grava política/parâmetros no checkpoint e rejeita
  resume com mudança de política. Para adaptar pesos históricos ao novo alvo,
  iniciar um ramo novo, não continuar silenciosamente o experimento antigo.
- `compression/encode_decode_samples.py`: verifica compatibilidade da política
  espacial de checkpoints novos com as amostras; avisa quando pesos históricos
  são aplicados à nova distribuição espacial.
- `--spatial-band-policy none` reproduz a seleção espacial legada. O runner
  `compression/compare_raw_inference.py` mantém essa opção como padrão histórico;
  a seleção nova é ativada explicitamente com
  `--spatial-band-policy persistent-amplitude-v1`.

Os datasets preparados anteriormente **não foram sobrescritos ou migrados**.
Extrair/preparar em diretórios novos. Apenas carregar um NPZ antigo não executa
a seleção nova: o filtro pertence à etapa de preparação/extração.

## Verificação operacional

Runner: `finetuning/audit_spatial_filter.py`. Evidência:
[relatório por arquivo](../results/experiments/spatial_filter_validation_20261022/REPORT.md),
`summary.csv`, decisões JSON, perfis espaciais com faixas selecionadas e `COMPLETE`.
Foram analisados os **24 arquivos sorteados** do relatório 60 e os dois exemplos
de `data/raw`, com hashes dos brutos conferidos antes/depois:

| Conjunto | Resultado com a regra inicial |
|---|---|
| Doped: 6 arquivos | Nenhuma faixa ultrapassou conjuntamente os critérios; preservados. |
| Multicore: 6 arquivos | Nenhuma faixa selecionada; preservados. |
| Multimode: 6 arquivos | Exclusão por amplitude de 5–27 canais por arquivo, nas faixas registradas. |
| Single-mode sorteados: 6 arquivos | Nenhuma faixa selecionada; preservados. |
| Configuration 1 / 200 Hz local | 40 canais excluídos por amplitude, índices 505–544: **121,0–128,8 m**. |
| Configuration 5 / 3.000 Hz local | Nenhuma faixa selecionada por amplitude; 111 canais preservados. |

No 200 Hz também são excluídos separadamente seis canais inteiramente não finitos,
resultando em **505 canais retidos, 20–120,8 m**. As novas extrações, sem inferência,
ficam em `data/spatial_filtered_raw_20261022/`; manifestos em
`results/manifests/spatial_filtered_{200hz,3000hz}_20261022.json`.

**Não confundir a concentração de energia do relatório 60 com este critério de
rejeição.** Top 10% concentrar muita energia não implica amplitude robusta oito
vezes maior por uma faixa adjacente persistente. Por isso os doped e alguns
single-mode com concentração elevada não foram automaticamente excluídos.

Testes cobrem picos pontuais de até `1e30`, eventos fortes em uma janela/bloco,
ganho global, offsets persistentes, limites de cobertura, insuficiência de dados,
regiões internas, igualdade exata dos valores retidos e paridade treino/inferência.
A suíte completa passou **208 testes**. Essa é validação funcional/operacional,
não demonstração de ganho de reconstrução nem de acurácia da máscara física.

## Validação da melhoria por inferência com pesos congelados

A comparação offshore original × FEBUS 4× foi repetida nos dois arquivos de
`data/raw`, agora com `persistent-amplitude-v1` **antes da normalização e da
codificação de ambos os modelos**. Foram mantidos os checkpoints: offshore sem
fine-tuning, época 30, e melhor RMSE macro 4× do fold 0, época 40. Não houve
treinamento, ajuste do filtro ou seleção de modelo por arquivo nesta validação.

Evidência: [relatório da execução](../results/inference_results/offshore_vs_4x_spatial_filtered_20261022/REPORT.md),
`protocol.json`, `coverage.json`, `comparison.csv`,
`preprocessing_effect_same_coverage.csv`/JSON, `execution.log` e `COMPLETE` no
mesmo diretório. Hashes dos HDF5 e checkpoints permaneceram inalterados.

### Controle sobre o mesmo alvo espacial

Para verificar ganho real no sinal restante, e não apenas melhora decorrente de
excluir canais da avaliação, a reconstrução da execução anterior foi recortada
**nos mesmos 505 canais fonte agora retidos** no arquivo de 200 Hz. A igualdade
exata dos alvos e posições foi conferida. Comparam-se, portanto:

- **Antes:** inferência com a faixa excessiva presente, mas avaliação somente
  nos canais que a política nova retém.
- **Depois:** exclusão da faixa antes da inferência e avaliação nesses mesmos
  canais, com os mesmos pesos e chunks de 1.000 amostras.

| Modelo / arquivo 200 Hz | Correlação antes → depois | RMSE relativo centrado antes → depois | p90 relativo por canal antes → depois |
|---|---:|---:|---:|
| Offshore original | 0,043326 → **0,125462** | 1,057001 → **0,993875** | 1,077207 → **1,032229** |
| FEBUS 4×, época 40 | 0,728732 → **0,805477** | 0,685150 → **0,596336** | 0,808237 → **0,715363** |

**O teste valida um benefício localizado do pré-processamento implementado:**
a reconstrução dos mesmos canais do arquivo de 200 Hz melhorou em ambos os
modelos, sem retreinar. No 4×, o RMSE centrado caiu aproximadamente **13,0%**.
O contraste mede conjuntamente a mudança de normalização e de geometria/contexto
espacial de borda; não isola qual mecanismo causou cada parcela do ganho.

No arquivo de **3.000 Hz**, nenhum canal foi excluído por amplitude e as
reconstruções foram **numericamente idênticas** às anteriores, nos dois modelos.
O 4× manteve correlação **0,892990**, RMSE centrado **0,456897** e p90 por canal
**1,373540**, ainda indicando erro elevado em parte dos canais.

As compressões nominais índices+side do 4× foram **45,867×** em 200 Hz e
**45,939×** em 3.000 Hz. A base é o sinal retido; custo do mapa espacial e demais
overheads de implantação não estão integralmente incluídos. Offshore e 4× não
operam à mesma taxa. A exclusão de cobertura não é ganho de compressão neural.

### Limites do detector e da validação

O sinal retido ainda apresenta heterogeneidade de amplitude. O filtro seleciona
canais por **excesso persistente**, não equaliza o sinal nem limita amplitudes
de amostras individuais.

- Em 200 Hz, o canal em **20 m** ultrapassou o limiar em somente **3/24** janelas;
  os canais em **120,4–120,8 m** também não cumpriram o critério de persistência.
  Permaneceram, apesar de amplitudes/RMS elevados.
- Em 3.000 Hz, os canais próximos de **106–110 m** não atingiram o critério de
  amplitude relativa 8×; a região mais energética permaneceu.

Logo, houve ganho demonstrado no exemplo de 200 Hz, **não eliminação de toda
faixa de alta amplitude nem validação física da máscara**. Tornar a seleção mais
abrangente exigiria uma nova versão/controlada avaliação dos critérios, não uma
mudança silenciosa após observar o erro. Os dois arquivos são exemplos de desenvolvimento
já inspecionados, não confirmação independente de todas as famílias ou do novo
regime de treinamento.

## Próxima decisão

### Uso confirmado nos treinamentos A/B/C

As novas comparações utilizam os corpora versionados
`data/finetune/spatial_band_v1_fold0_20261024/` e
`data/finetune/spatial_band_v1_fold1_20261026/`, preparados com esta política,
não os datasets legados sem seleção por amplitude. Na confirmação do fold 1,
foram cruzadas todas as **14.620 janelas do manifesto de treino** com seus mapas
e decisões: nenhum canal marcado para remoção permaneceu nas entradas retidas.
O checkpoint também registra o diretório e `persistent-amplitude-v1` com os
parâmetros descritos acima. A auditoria integral anterior verificou os valores
dos NPZ contra os HDF5 originais.

O filtro roda uma vez na preparação; a cada época o modelo recebe as janelas
já selecionadas, antes da normalização. Detalhes e retomada estão no
[relatório 63](63-SPATIAL_ABC_RESULTS_AND_FOLD_CONFIRMATION.md).

O corpus corrigido e a auditoria de cobertura já foram concluídos
([relatório 62](62-SPATIAL_CORPUS_AND_CONTROLLED_NEURAL_COMPARISON.md)); a comparação
e sua confirmação seguem no relatório 63. Continuar reportando a fração
excluída; calcular taxa/fidelidade em referência ao sinal útil retido, não contar
remoção de canais como ganho de compressão neural. A/B/C devem usar a mesma regra,
tiling, splits e contabilização de bits. Nenhum modelo histórico é promovido por
esta alteração de alvo.
