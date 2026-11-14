# Learning rate e pré-treinamento externo: análise e auditoria QuantX

**Data: 09/11/2026.** Análise solicitada pelo usuário, posterior ao
[checkpoint 2](PROGRESS_CHECKPOINT_2.md). Inclui inventário novo do Drive,
download verificado e diagnóstico CPU de sinais QuantX, revisão do histórico de
otimização e busca de fontes públicas. **Não houve treinamento nem avaliação de
novos modelos.** Os protocolos abaixo são propostas, não resultados de fidelidade.

**Atualização após autorização:** a seção 7 registra a implementação e o lançamento
da continuidade pareada; a seção 8 registra a auditoria nova de arquivos públicos.
As conclusões da revisão inicial acima não são resultados desses treinamentos.

**LR concluído e auditado em 10/11:** a repetição válida terminou sem promoção.
Cosine ganhou apenas 0,324% / 0,159% contra os controles constantes, com mais
bytes e regressões. Resultado e auditoria no
[relatório 81](81-LR_SCHEDULE_PAIRED_RESULTS.md). As menções a resultados pendentes
nas seções de lançamento abaixo são registros históricos dessas etapas.

**Ponte QuantX concluída em 10/11:** o
[relatório 82](82-QUANTX_PRETRAINING_BRIDGE_RESULTS.md) registra o negativo frente
ao investimento direto em FEBUS e o controle posterior de dose alvo igual.
A comparação pública PoroTomo foi lançada depois desses controles, sem outro
treino concorrente. As indicações de lançamento pendente abaixo são históricas.

**PoroTomo concluído:** o
[relatório 83](83-POROTOMO_PRETRAINING_BRIDGE_RESULTS.md) distingue a melhora de
4,537% com dose FEBUS igual da perda contra investir os mesmos updates em FEBUS.
A fonte pública ajudou a adaptação curta, mas o ponto foi dominado pelo controle
direto em erro e bytes; não houve promoção ou extensão automática desse estudo.

## 1. Decisão resumida

- **Vale testar scheduler**, em continuação pareada e curta da CNN transferida.
  Não foi localizada uma ablação controlada de scheduler nos runners e relatórios
  anteriores. Houve mudanças de LR entre campanhas, com confundidores importantes.
  A expectativa é refinar otimização, não resolver sozinho o gargalo de bits.
- **Vale testar outra fonte de pré-treino**, mas como hipótese de transferência,
  não por presumir que mais dados ou um fabricante específico garantam melhora.
  A evidência a favor é a liderança repetida da inicialização offshore sobre
  scratch; ela não identifica qual corpus externo seria melhor que OOI.
- **QuantX merece um piloto condicionado, não pré-treino integral imediato.** A
  amostra apresenta fase óptica, geometria mais estreita, variações lentas dominantes
  e metadados de comprimento inconsistentes. Isso não invalida os dados, mas exige
  evitar um pré-treino dominado por drift/canais sem região útil confirmada.
- **Principal candidato público para triagem:** o corpus bruto de eventos de
  Tomasov et al. (2025), por diversidade de atividades, HDF5 multicanal e aquisição
  declarada de 20 kHz. A taxa efetivamente armazenada e o produto físico ainda
  precisam ser confirmados nos arquivos, não apenas no artigo.

Não se alteraram guardas físicas nem modelos/codec de referência. O stress SMF
anterior continua desenvolvimento externo; os grupos EDAS reservados anteriormente
não foram abertos. Os grupos EDAS **da nova pasta QuantX** foram auditados aqui e
não constituem holdout final deste estudo.

## 2. LR: o que já foi feito e o que ainda não está isolado

| Evidência | Regime | O que permite concluir |
|---|---|---|
| [Relatório 2](2-FINETUNE_EVALUATION_REPORT.md) | LR `1e-5`; propôs reduzir para `3e-6`/`1e-6` após estagnação | A recomendação existia, não é resultado de scheduler. |
| [Relatórios 17–18](18-TEMPORAL_DENSITY_LONG_RESULTS_AND_DECISION_GATES.md) | Piloto `1e-5`, continuação com AdamW **novo** a `3e-6`, mais oito épocas completas | Temporal-4× ganhou **3,63% de RMSE**; redução de LR, optimizer novo e exposição adicional não foram isolados. |
| [Relatório 22](22-ARCHITECTURE_INITIALIZATION_RESULTS.md) | Continuação transferida a `1e-4`, AdamW novo, 65.536 updates | RMSE **0,8098 → 0,7199** no contraste histórico; não prova que só aumentar LR explica o ganho. |
| [Relatório 53](53-GUARDED_DECODER_PILOT_PROTOCOL.md) e seguintes | Pilotos a LR baixo, como `3e-6` | Baixo LR já foi explorado, mas não equivale a testar sua trajetória ao longo do treino. |
| [A/B/C](62-SPATIAL_CORPUS_AND_CONTROLLED_NEURAL_COMPARISON.md), capacidade e continuações recentes | Predomínio de LRs fixos por braço, incluindo `1e-4` | Ausência de superioridade arquitetural não demonstra otimização ótima de cada braço. |

A inspeção dos runners em `finetuning/`, do loop de `finetune_vqvae.py` e do
treinador original `../smartgrids/vqvae/train_vqvae.py` não localizou scheduler
ativo. Isso caracteriza o código/histórico disponível, não toda execução que
possa ter ocorrido fora dele. Busca de índices E1 usa outro optimizer sobre
propostas latentes: suas LRs não são experimentos de LR da rede.

### 2.1 Mecanismo e expectativa

Reduzir LR ajuda a refinar uma trajetória que oscila perto de uma solução;
**não é, por si só, mecanismo de escapar de qualquer mínimo**. Reinícios/ciclos
podem explorar outras regiões, mas também desestabilizar o codebook e apagar
transferência. A evidência existente não demonstra que os modelos estejam todos
estagnados: vários encerraram budgets ainda em melhora.

Começar por **cosine decay sem restart** oferece um contraste simples e previamente
definido, sem ajustar a política repetidamente à validação já consumida. Não
combinar nesse primeiro teste com nova loss, quantizador, corpus ou arquitetura.

### 2.2 Teste recomendado — proposto, não executado

1. Usar o **último estado completo** de um braço A transferido compacto do estudo
   espacial, com manifesto e proveniência exclusivos do fold. Preservar separadamente
   seu selecionado histórico como referência congelada. Verificar SHA-256 e optimizer.
2. Ramificar o mesmo modelo/AdamW/RNG, sem reset em apenas um braço; manter exatamente
   os mesmos batches, normalização, loss `0,5 MSE + 0,5 L1 + VQ`, geometria e codebook.
3. Executar sequencialmente **três braços × 16.384 updates**, organizados como
   oito épocas de 2.048 draws. Essa definição não equivale a oito passagens completas
   pelo corpus. Checkpoints/triagem a cada 2.048 updates; nenhum early stop assimétrico,
   salvo não finitos ou falha de integridade. Candidato inicial permanece elegível.

| Braço | LR | Papel |
|---|---|---|
| C-alto | Constante `1e-4` | Controla simplesmente continuar treinando. |
| Cosine | `1e-4 → 1e-5`, suave por update | Testa a trajetória de redução. |
| C-médio | Constante `5,5e-5` | Controla a média aritmética do LR do cosine. |

Para `t = 0, ..., T-1`: `LR(t) = 1e-5 + 0,5*(1e-4-1e-5)*(1+cos(pi*t/(T-1)))`.
A média de LR é igual à de C-médio, **não a dinâmica do AdamW**. O estado do
scheduler, contador global, LR efetivamente usado e RNG devem ser salvos/restaurados;
teste de resume deve reproduzir a sequência sem saltar/repetir o decaimento.
Não é necessário warm-up para continuar o mesmo estado funcional. Se o optimizer
fonte não estiver disponível, usar AdamW novo em **todos** e declarar a mudança.

4. Selecionar com a mesma regra predefinida por braço; registrar também o endpoint
   de exposição igual. Validar os selecionados no mesmo conjunto completo, com replay
   discreto, RMSE macro, famílias/configurações, p90, RMS, magnitude e fase. Medir
   **streams reais**: igualdade de grade nominal não garante entropia/bytes iguais.
5. **Triagem de investimento para este teste barato:** redução de RMSE ≥1% sobre
   ambos os controles constantes, sem aumento do stream e sem regressão das guardas
   históricas por configuração. Abaixo disso, registrar ganho marginal/negativo,
   sem extensão automática. Esse 1% não é tolerância física nem substitui o alvo
   sugerido de ≥5% para uma nova linha estrutural no relatório 77.
6. Se passar, confirmar o contraste em outro fold/seed, com fonte que não viu seus
   grupos reservados. Não promover com base na validação reutilizada. Não modificar
   simultaneamente o pré-treino: primeiro isolar o efeito de otimização.

`ReduceLROnPlateau` e warm restarts ficam como alternativas **não prioritárias**,
não uma grade automática de tentativas após ver o resultado. Um negativo nesse
cosine não refuta todos os schedules; um positivo não demonstra convergência ótima.

## 3. O que foi efetivamente auditado no QuantX

Pasta fornecida: [Drive QuantX](https://drive.google.com/drive/folders/16HRc9KiVRAMSDXERcRybeX1ek2k30Cvr).
Artefatos: `results/experiments/quantx_pretraining_audit_20261109/`.

- Inventário: **878 arquivos**, **302 HDF5**, **192 grupos de cenário** e
  **15.289.333.817 bytes** no total. Os quatro grupos têm 48 cenários cada,
  cruzando quatro segmentos, comprimentos rotulados 3/4/5 m e 200/500/800/1000 Hz.
  Os nomes são rótulos experimentais, não prova da geometria ou taxa de saída.
- Seleção **anterior à inspeção dos sinais**: maior HDF5 de cada um de três cenários
  fixos (`segmento1_3m_200hz`, `segmento3_4m_500hz`, `segmento4_5m_1000hz`), nos
  quatro grupos. Isso reduz o viés de escolher apenas o fragmento final curto;
  ainda assim, não é amostra aleatória/representativa de todos os 192 cenários.
- Download: **12 HDF5**, configs/logs associados, **886.809.072 bytes**; MD5 do
  Drive e tamanho verificados, SHA-256 local registrado. Os HDF5 somam **458,16 s**
  armazenados, não necessariamente independentes. Nenhum duplicado MD5+tamanho
  entre os 302 HDF5; nenhum Drive ID compartilhado com o inventário FEBUS.
  Isso **não prova independência física** entre os ensaios ou cabos.
- Sinais: cinco janelas distribuídas de 2.048 amostras por HDF5, **60 janelas**,
  todos os loci de cada uma: ~12,288 s amostrados. Sem seleção por erro neural,
  remoção de canais, filtro de denoising ou alteração dos brutos. Flags `GpBits`
  foram lidas por inteiro nos 12 HDF5: nenhuma amostra marcada como modificada
  ou comprometida. Não houve não finitos nas janelas de sinal lidas.

### 3.1 Compatibilidade física observada

| Item | Amostra QuantX | Referência FEBUS |
|---|---|---|
| Produto | `Diversity Processed Phase. Single Pulse. OCP 10` | `Strain`/`Strain Rate`, conforme arquivo |
| Unidade armazenada | `rad * 2PI/2^16`, int32 | Não assumir equivalência com fase |
| Taxa de saída | **10.000 amostras/s** nos 12 arquivos | 20.000 amostras/s no corpus atual |
| Espaçamento | **1,020952 m** | ~0,2 m |
| Gauge length | **2,041905 m** | Ler por arquivo/configuração |
| Matriz | Tempo × **70 ou 100 loci** | Tiles de até 512 canais |
| Duração dos arquivos selecionados | **32,2–40 s** | Recortes atuais de 50 ms para a rede |

A configuração confirma `LRQDAS_QUANTX`. Os 200/500/1000 Hz nas pastas **não
são sample rates**. Apenas trocar a escala numérica não transforma fase em taxa
de deformação; calibrar/diferenciar requer confirmar o produto e o tratamento de
unwrap, saltos e resposta instrumental. Pré-treino pode deliberadamente usar
outra variável, mas ela precisa ser declarada, não falsamente chamada de FEBUS.

`fibres.lengthMetres` varia entre **0 e 72 m** na amostra, inclusive SMF com 9 m
e outro cenário com 0 m apesar de 100 loci registrados. Esses campos não oferecem
um mapa espacial validado; não concluir que todos os loci restantes sejam inválidos,
nem alimentar a rede indiscriminadamente com essa interpretação de extensão útil.
As famílias QuantX também não comprovam diversidade multicore equivalente ao FEBUS.

### 3.2 Conteúdo dos sinais e implicação para o pré-treino

Na fase bruta, após **apenas centrar por canal para o diagnóstico**, a mediana de
60 janelas mostrou ~**90,0% da energia espectral nos cinco maiores bins** (Hann,
DC excluído). Esses bins não representam necessariamente cinco tons: na maioria
dos arquivos predomina a primeira frequência resolvida, ~4,88 Hz, sugerindo forte
contribuição de variação lenta/drift. O diagnóstico de projeção linear explica
**62,4%** da energia centrada na mediana das medianas por arquivo; há heterogeneidade.

O rank efetivo espacial da fase centrada tem mediana ~**4,0**, contra ~**42,3**
na primeira diferença temporal. A concentração nos cinco bins cai para ~**0,84%**
na diferença. São diagnósticos de matrizes diferentes: **não demonstram rank baixo
do DAS em geral nem autorizam descartar o residual de alta frequência**. Diferenciar
também amplifica altas frequências; o contraste não é uma medição independente
de ruído branco ou uma conversão física validada.

Na banda ±20 Hz em torno da excitação rotulada, a fração pooled da fase bruta
é só **0,0107–0,3553%**, intervalo das medianas por arquivo. Em canais específicos,
após detrend **somente diagnóstico**, a concentração pode ser bem maior. Portanto,
não declarar ausência de excitação pela energia global, nem presumir que o
pré-treino por MSE vá dar atenção suficiente aos trechos/canais excitados.

**Interpretação:** esses dados podem ensinar resposta de fibra/instrumento e
variações locais, mas também podem ensinar principalmente drift e ruído de regiões
não caracterizadas. Baixa loss na fase bruta não é o endpoint da proposta:
o teste decisivo é a reconstrução FEBUS **depois** do fine-tuning controlado.
Não remover drift ou ruído do alvo FEBUS para favorecer a nova inicialização.

## 4. Fontes públicas encontradas e prioridade

| Fonte / acesso | Utilidade para este projeto | Decisão |
|---|---|---|
| [Tomasov et al., Scientific Data 2025](https://doi.org/10.1038/s41597-025-05088-4); [Figshare](https://doi.org/10.6084/m9.figshare.27004732), CC BY 4.0 | Brutos OptaSense ODH-F, atividades urbanas, fundo, manipulação, obras, 1.663 loci; artigo declara pulsos de 20 kHz e banda máxima de 10 kHz. | **Primeiro candidato público a auditar**. Usar waveform, não PSD/bitmaps ou apenas exemplos de classificação. Confirmar `OutputDataRate`, variável, gauge length e partições por aquisição. |
| [EGS Collab, experimento 2](https://gdr.openei.org/submissions/1606), CC BY 4.0, S3 público | Três sistemas: OptaSense ODH3, Silixa iDAS e Terra15 Treble, ambiente de poços e aquisições contínuas. | Bom candidato multi-instrumento, mas selecionar subconjunto compatível, não baixar centenas de TB. A fibra monitorada por DAS é single-mode; haver fibras multimode no pacote não significa corpus DAS multimode. |
| [PubDAS / Spica et al. 2023](https://doi.org/10.1785/0220220279) | Oito campanhas e ~90 TB na publicação, urbano/minas/fundo marinho, contínuos e fontes ativas. | Fonte para diversidade de campanhas, não um dataset homogêneo pronto; priorizar registros não filtrados e sem sobreposição OOI conhecida. Acesso/licença precisam ser checados por componente. |
| [Stanford/Huot, Zenodo](https://zenodo.org/records/6344570), CC BY 4.0, ~1,3 GB | Terremotos e fundo, mas TFRecords já processados para detecção. | Menor prioridade para compressão física; primeiro verificar se preservam waveform e geometria, sem perda prévia de fase/bandas. |
| [Dejdar et al. 2025, sons audíveis](https://doi.org/10.1038/s41597-025-05119-0); [dados](https://doi.org/10.6084/m9.figshare.27620397) | Fundo, sweeps e STIPA em infraestrutura real, modos P9 e SP1. | Complemento acústico, não fonte generalista principal. SP1 é intensidade/amplitude, P9 é fase: não misturar como a mesma variável; evitar selecionar apenas sinais fáceis/coerentes. |
| [Forbriger et al., FEBUS A1-R / BFO](https://doi.org/10.35097/fx5ztp5htp7y9704), CC BY 4.0, ~41,5 GB | Mesmo fabricante, estudo de strain/coupling e comparação de interrogadores. Dados disponibilizados já decimados/formato genérico, com foco de baixas frequências. | Interessante para instrumentação/calibração, baixa prioridade para inicialização visando 20 kHz. Metadados têm discrepância de ano entre título e descrição; resolver antes de usar. DOI do head record informado no registro respondeu 404 nesta consulta. |

### Verificação de acesso que foi além da busca bibliográfica

- **Figshare:** a API confirma ZIP público de **47.120.418.495 bytes** (~43,88 GiB).
  Uma leitura HTTP Range de **18.111 bytes** do diretório ZIP identificou **16 HDF5**,
  distribuídos nas atividades; o menor tem ~1,01 GB. Não baixamos o ZIP nem
  inspecionamos os sinais/HDF5 internos. Milhões de janelas derivadas desses arquivos
  não serão milhões de aquisições independentes. O ZIP comprimido limita acesso
  aleatório aos HDF5 internos: planejar extração de poucos arquivos antes do treino.
- **EGS:** listagem S3 pública e download de **um HDF5 de 15.879.772 bytes**, SHA-256
  registrado, **somente metadados abertos**. Nesse arquivo, `PulseRate=20000`, mas
  **`OutputDataRate=200`**, 776 loci, gauge ~7,15 m e fase filtrada. Não é a amostra
  de alta taxa procurada. Não generalizar essa condição a todo o arquivo EGS,
  mas tampouco vender a fonte como 20 kHz com base nos pulsos.

Os demais candidatos foram analisados pelas páginas/artigos, sem download de
waveforms. A evidência bibliográfica de classificação/denoising não é evidência
de ganho de compressão/reconstrução FEBUS.

## 5. Como testar a hipótese de outra inicialização sem confundir mais compute

**Primeiro decidir o produto e as aquisições úteis**, não fabricar uma equivalência
via upsampling. A 10 kHz, 1.000 amostras representam 100 ms, contra 50 ms FEBUS.
Pode-se pré-treinar com esses contextos diferentes, explicitando-os; não repetir
os 70–100 canais para imitar 512 nem concatenar fibras/ensaios descontínuos.
Usar shapes admitidos pela arquitetura e registrar padding/grade, não mudar a
arquitetura junto com o dataset. FFTs/log-PSDs não substituem alvos de waveform.

### Piloto prioritário de utilidade prática — proposto

Partir do **mesmo checkpoint offshore original** e arquitetura compacta usada
nos controles; não de um modelo FEBUS que já tenha visto a futura validação.
O scheduler escolhido, se confirmado, deve ser comum a todos os braços.

| Braço | Etapa 1: 16.384 updates | Etapa 2: 8.192 updates FEBUS |
|---|---|---|
| F — exposição controlada | FEBUS treino | Mesmo FEBUS treino |
| Q — ponte QuantX | QuantX auditado, sem dados reservados | Mesmo FEBUS treino |
| P — ponte pública, após preflight | Subconjunto público compatível | Mesmo FEBUS treino |

Todos têm **24.576 updates adicionais** e orçamento final FEBUS pareado na etapa 2;
F tem mais exposição FEBUS na etapa 1, deliberadamente: responde se investir esse
compute na nova fonte é melhor do que gastá-lo diretamente no alvo. A ordem dos
batches FEBUS da etapa 2 é idêntica. Restaurar o mesmo checkpoint de partida e
mesma seed antes de cada braço; descartar/resetar optimizer na fronteira de etapa
da mesma forma em todos, inclusive F. Balancear a etapa externa por campanha/fibra
antes de janelas, sem multiplicar o peso de uma única gravação longa.

Primeiro rodar **F e apenas uma ponte**, selecionada pela auditoria, não as duas
automaticamente. QuantX é operacionalmente acessível, mas depende da resolução
do mapa/produto; o corpus público é mais promissor em diversidade, ainda não
auditado em waveform. Não há evidência para escolher um vencedor por antecipação.

Esse contraste testa **OOI → nova fonte → FEBUS**, não substituir integralmente
OOI. Se funcionar, executar a proposta estrita **aleatório → nova fonte → FEBUS**
e um controle aleatório → FEBUS de exposição compatível, além da referência OOI.
Sem retreinar OOI sob budget equivalente, não declarar comparação causal pura
de qual dataset de pré-treino é superior: seu compute histórico é diferente.

### Avaliação e critérios

- Particionar por aquisições/dias/fibra fisicamente relacionada, agrupando variantes
  simultâneas e arquivos de uma mesma sessão. Conferir a relação dos ensaios QuantX
  com os grupos FEBUS antes de qualquer alegação de generalização externa.
- Nenhuma escolha do corpus/transformação usando o teste FEBUS. Os dados auditados
  aqui são desenvolvimento; preservar aquisições novas de ambos os domínios para
  confirmação posterior, sem chamar janelas vizinhas de holdout independente.
- Seletores e budget fixos, candidato inicial elegível, replay discreto e streams
  completos; RMSE físico relativo/absoluto, correlação, p90, RMS, espectro/fase e
  custo por configuração. Não assumir que maior perplexidade do codebook é ganho.
- Usar o critério estrutural sugerido no relatório 77: **buscar ≥5% menos RMSE a
  bytes não maiores**, preservando guardas e confirmando em outra coorte. Ganhos
  menores podem ser informativos, mas não acionam campanha longa automaticamente.
  Reportar contraste por arquivo/configuração; incerteza agrupada por aquisição,
  não erro-padrão artificial de milhares de janelas correlacionadas.
- Preservar originais/ruído, nenhuma conversão strain/strain-rate não declarada,
  sem misturar a fonte externa com validação/teste FEBUS. Pré-treino para denoising
  ou classificação não é a mesma tarefa de reconstruir também o ruído.

## 6. Prioridade e rastreabilidade

Ordem recomendada: **(1) scheduler pareado; (2) preflight físico da fonte pública
e resolução de mapa/produto QuantX; (3) um piloto F versus ponte; (4) confirmação
antes de ampliar o corpus ou testar substituição completa da inicialização OOI.**
Isso se encaixa no roadmap em convergência/otimização e ampliação de dados
compatíveis, sem reabrir automaticamente largura, anisotrópica ou refinamentos
encerrados. Mais dados podem melhorar o prior aprendido; não aumentam magicamente
a informação transmitida pelo gargalo nem prometem alta fidelidade a 700×.

Runners novos: `acquisition/audit_quantx_pretraining.py`,
`acquisition/analyze_quantx_pretraining.py`, `acquisition/diagnose_quantx_drift.py`.
Inventário/downloads/protocolo/hashes, metadados físicos, estatísticas por janela e
contraste de drift estão no diretório de resultados acima, com snapshots dos
scripts. Quatro testes de diagnósticos sintéticos passaram; não foi reexecutada
a suíte científica completa por não haver mudança nos codecs/modelos.

**Execução:** apenas aquisição e CPU nesta análise; nenhuma GPU local utilizada.
Treinos futuros destes protocolos serão sequenciais em **uma única RTX 4090
remota**, com dispositivo explicitamente fixado. Não há treino pendente/ativo
lançado por esta revisão. Nada foi commitado ou enviado ao Git/Drive.

## 7. Continuidade autorizada: execução do contraste de LR

Após autorização do usuário, foi implementado um runner separado, preservando
os treinadores históricos e suas políticas: `finetuning/run_lr_schedule_study.py`
e `finetuning/update_schedule.py`. A avaliação em
`finetuning/evaluate_lr_schedule_study.py` transmite índices/side em FAK1 e os
reproduz antes de medir fidelidade; FSB1 cobra tabelas, mapas, manifesto, framing
e registro, com pesos de implantação contabilizados separadamente do stream.
Nenhum estado do AdamW é cobrado como peso necessário ao decoder.

**Fonte fixada:** `spatial_abc_20261024_v2/compact_A/best_macro.pt`, época 24,
que também é o endpoint completo desse braço. O artefato preservado contém os
117 estados de parâmetros do AdamW no update **98.304**, mais RNG de uma GPU.
Não se usará o treino nativo do zero nem um encoder-only recente como substituto
da transferência A. A validação é a do fold 1 da primeira coorte, com treino dos
folds 2/3 e teste fold 0 intacto; a confirmação, se justificada, exigirá fonte
compatível com a outra rotação.

O plano continua **3 × 16.384 updates**, oito blocos de 2.048, sem extensão ou
early stop por desempenho. A sequência de entradas é compartilhada e registrada
antes do primeiro update. A seleção usa a mesma triagem estratificada histórica,
com inicialização elegível; endpoint e selecionado serão avaliados separadamente
em toda a validação. Nada disso constitui teste externo ou promoção de modelo.

**Verificações antes do lançamento:** a identidade checkpoint/protocolo fonte e
os passos do optimizer foram conferidos. A retomada sintética de AdamW + cosine
reproduziu parâmetros e sequência de LRs bit a bit. A suíte local passou em
**340 testes**. Teste de memória/integração na única RTX 4090 e resultados da
campanha serão registrados após sua execução; não há ganho de reconstrução novo
estabelecido por esses testes de implementação.

O smoke remoto verificou os três braços, replay dos índices, containers e retomada
sem atualizações adicionais. O primeiro smoke revelou um erro de nome de campo na
impressão do avaliador (`macro_relative_rmse`), corrigido antes da campanha; os
artefatos interrompidos foram preservados. O runner e avaliador finais passaram
a ter seus hashes congelados no protocolo antes de abrir a avaliação completa.
Foi lançado o pipeline remoto `lr_schedule_fold0_20261109`: somente GPU física 0,
três braços sequenciais, seguidos da avaliação de fonte, selecionados e endpoints.
**Resultados científicos ainda pendentes nesta atualização.** O smoke de duas
atualizações por braço é teste de integração, não evidência de ganho de LR.

**Auditoria posterior do primeiro run completo — comparação rejeitada:** os
três braços terminaram com contadores AdamW 114.688 / 131.072 / 147.456, quando
todos deveriam terminar em **114.688** a partir dos mesmos 98.304. Embora os
pesos iniciais fossem recarregados, `optimizer.load_state_dict` preservou alias
dos tensores CPU de `step` do estado fonte em memória. O arquivo fonte não foi
modificado. Isso viola o isolamento do protocolo; os resultados não serão usados
para confirmar ou rejeitar a hipótese do scheduler. Os artefatos originais foram
preservados com `AUDIT_REJECTED.json`, sem promoção.

A correção usa cópia profunda do estado em cada ramificação, teste sintético de
imutabilidade de passos/momentos e assertivas por bloco de que todos os parâmetros
estão no passo fonte + updates do próprio braço. **341 testes locais passaram.**
A repetição controlada terá diretório distinto, `lr_schedule_fold0_v2_20261109`,
com o mesmo budget de 16.384 updates por braço e as mesmas guardas, corpus e
critérios de seleção. A primeira execução não será reclassificada silenciosamente
como comparação pareada válida mesmo se o efeito numérico do contador for pequeno.
Os contadores cumulativos não significam que os pesos receberam mais batches:
cada braço executou de fato 16.384 updates. O defeito foi no estado inicial do
optimizer dos braços posteriores; a repetição distingue esse problema de um
eventual efeito numérico nulo sobre as trajetórias já observadas.

## 8. Preflight público em waveform: compatibilidade revista

Artefatos: `results/experiments/public_das_pretraining_preflight_20261109/`.
Foi extraído apenas o menor HDF5 de abril do ZIP de Tomasov et al., sem baixar
o ZIP inteiro: ~1,009 GB comprimidos, tamanho e CRC do membro verificados,
SHA-256 `8a0e6d7c387a07a17b9b175c4d4e755948ff910e731371fe8f998cb408f7ac76`.

**O arquivo confirma saída a 20 kHz**, não apenas pulsos a essa taxa. Porém,
o conteúdo é `Single Pulse, SR: 1.5, OCP: 1`, dados de 16 bits, sem unidade
`RawDataUnit` ou gauge length físico declarados no grupo Acquisition. Essa
descrição é consistente com o readout single-pulse de intensidade, não uma
medida de strain calibrada; confirmar a cadeia instrumental antes de qualquer
conversão física. A correspondência de sample rate não resolve essa diferença.

O HDF5 tem **425.600 × 1.700** valores, ~21,28 s. Cinco janelas de 2.048 amostras,
todos os loci, mostraram cosseno temporal lag1 ~0,058–0,143 e espacial adjacente
~0,066–0,101, rank efetivo espacial ~623–836 e 3,52–9,99% da energia espectral
nos cinco maiores bins. São estatísticas da medição bruta centrada, não do
latente FEBUS; não provam independência, impossibilidade de compressão ou ausência
de eventos localizados. Também não caracterizam toda a campanha pública.

**Decisão revista:** a fonte permanece candidata de diversidade, mas não deve
ser priorizada como strain/strain-rate compatível apenas pelo artigo e pelos
20 kHz. Foi selecionado para auditoria complementar o menor HDF5 da campanha
de novembro, antes de olhar seus sinais, para verificar se há outro readout.
Não iniciar o pré-treino público integral a partir de uma equivalência física
que a primeira inspeção não confirmou.

**Auditoria complementar concluída:** o menor HDF5 de novembro foi baixado por
range do mesmo ZIP, com CRC/tamanho íntegros e SHA-256
`3b5e8f9cf13d91a624cbebd80f86cf32d9c40060e88b0cb0c24fe88c249308d6`.
Resultados em `results/experiments/public_das_november_preflight_20261109/`.
O arquivo contém **494.400 × 1.700** valores a **20 kHz**, ~24,72 s, com locus
inicial 100. Seu readout também é `Single Pulse, SR: 1.5, OCP: 1`, 16 bits,
`Phase LSB=0` e sem unidade RawDataUnit/gauge físico declarado. Não foi encontrada
uma segunda modalidade de fase calibrada nessas duas aquisições inspecionadas.

Nas cinco janelas do segundo arquivo, os cossenos temporal lag1 e espacial
adjacente ficaram em ~0,027–0,041 e ~0,051–0,058; rank espacial efetivo ~882–904;
2,75–3,93% da energia nos cinco maiores bins. Novamente, não são limites
informacionais nem medições do rank de latentes; o arquivo pode conter sinais
localizados apesar da baixa dependência global diagnosticada.

**Prioridade após os dados, não só o artigo:** não iniciar por enquanto a ponte
pública com esse corpus. Uma transferência a partir de intensidade ainda é
hipótese válida, mas seria um contraste mais distante do produto FEBUS e merece
prioridade menor que fontes de fase/strain-rate interpretadas. Esses dois arquivos
não permitem afirmar que todos os 16 HDF5 do ZIP usem o mesmo readout. Também não
promovem automaticamente QuantX: sua fase é mais diretamente relacionada à
deformação, porém os avisos de drift e mapa útil permanecem. O teste de scheduler
não foi alterado por esses resultados de aquisição.

## 9. Próximo contraste delimitado: ponte QuantX como fase bruta

Após a inspeção pública, a fonte acessível mais diretamente relacionada à
deformação é a fase QuantX, embora sua utilidade para FEBUS siga desconhecida.
Para um piloto **não é necessário fingir que os produtos sejam iguais**: a
hipótese será explicitamente a utilidade dos dados QuantX como foram registrados,
com todos os loci, antes de adaptar ao alvo FEBUS inalterado.

Isso resolve a política de entrada sem inventar um mapa útil: **não será aplicada
nenhuma exclusão espacial** baseada nos comprimentos inconsistentes do config.
O piloto não demonstrará o valor de um QuantX hipoteticamente recalibrado/filtrado,
nem a validade física de todos os seus loci. Pode resultar em transferência
negativa precisamente por drift/ruído/partes da entrada inadequadas; isso será
registrado, não corrigido retrospectivamente para salvar a comparação.

Preparação CPU: os mesmos 12 HDF5 auditados, todas as janelas completas e não
sobrepostas de **1.000 amostras × todos os loci**; contagens de fase int32 são
centralizadas/escaladas em float64 **antes** de converter para float32, evitando
perder variações pequenas em torno de offsets grandes. Sem diferenciação, unwrap,
remoção de drift, denoising, reamostragem ou concatenação de aquisições. Preservar
os HDF5 originais. O sufixo incompleto fica fora do treino apenas; isso não é
economia de um codec. Cada janela QuantX representa 100 ms, contra 50 ms FEBUS;
é uma diferença física declarada, não eliminada por igualar 1.000 pixels.

O piloto previsto continua **F versus Q**, a partir dos mesmos pesos offshore
originais com geometria compacta comum, sem usar o modelo FEBUS da campanha de LR
como inicialização:

- **F:** 16.384 updates em FEBUS treino + 8.192 updates no mesmo FEBUS treino.
- **Q:** 16.384 updates em fase QuantX + os mesmos 8.192 updates FEBUS finais.

AdamW novo em ambos no início e na transição de etapa, LR constante `1e-4`,
mesma loss temporal+VQ, pesos/quantizador treináveis, sem outra mudança de rede.
O LR constante é o controle pré-definido desta hipótese separada; não depende de
selecionar retrospectivamente o vencedor da campanha de scheduler. Na etapa
QuantX, balancear os quatro grupos de fibra e depois aquisições dentro do grupo;
não chamá-los de cobertura equivalente às quatro famílias FEBUS. Os batches da
etapa FEBUS final serão exatamente os mesmos entre F/Q. Seeds, sequências e
hashes serão congelados antes dos updates. A dose é pareada em updates adicionais,
**não FLOPs nem exposição FEBUS**: a diferença de exposição é parte da pergunta
prática de onde investir o compute.

Avaliar modelo inicial, selecionado e endpoint em toda a validação FEBUS, com
mesmo replay discreto, framing e guardas. O corpus QuantX não será chamado de
holdout externo nem haverá seleção de pré-treino por teste FEBUS. O candidato
offshore inicial permanece elegível; selecionados não apagam endpoints. Procurar
≥5% menos RMSE que F, sem aumento de bytes ou regressão das guardas, antes de
qualquer extensão/segunda coorte. Não usar a primeira comparação de LR rejeitada
para escolher coeficientes ou budgets. Esta preparação não lança treino adicional
concorrente: a única GPU remota continua reservada à repetição controlada de LR.

**Preparação concluída:** `quantx_bridge_preparation_20261109/` registra
**4.579 janelas**, 1.579.636.683 bytes preparados e nenhuma janela constante.
São ~457,9 s nominais de fase fonte, sem alegação de independência; o sufixo
incompleto somado corresponde a 2.600 amostras (~0,26 s). Um exame dos clocks
completos nos 12 HDF5 encontrou incremento constante de timestamp e contadores
de amostra sempre `+1`, sem regressão do relógio nem saltos de contador dentro
dos arquivos. Isso não estabelece calibração absoluta do clock ou independência
entre aquisições. O manifesto SHA-256 é
`acef5a630b26dc6d0d48151d7d474b11ea1735a73a116f1bf5732989949c7644`.

O runner `finetuning/run_pretraining_bridge.py` e o avaliador
`finetuning/evaluate_pretraining_bridge.py` foram implementados separadamente da
campanha de LR congelada. A transição de etapa reinicia AdamW em ambos, com
assertivas do contador e batches FEBUS finais compartilhados; o avaliador usa
o mesmo FAK1/FSB1 e replay de índices, não uma nova fronteira de codec.
**344 testes locais passaram.** A integração GPU e o piloto ainda não foram
lançados: não sobrepor seus treinos à repetição controlada do scheduler.

**Continuidade de 10/11:** o scheduler válido concluiu (relatório 81) e foi
lançado o pipeline `pretraining_bridge_fold0_20261110`, precedido por smoke e
testes de integração, na GPU física 0. Os braços seguem sequenciais; não se
interpretam métricas parciais de treino como confirmação de transferência.

## 10. Alternativa pública com readout verificado: PoroTomo/iDAS

Na continuidade autorizada, foi inspecionada uma aquisição DASV de
[PoroTomo/GDR](https://gdr.openei.org/submissions/980), DOI
[10.15121/1778858](https://doi.org/10.15121/1778858), licença CC BY 4.0.
Trata-se de um poço com Silixa iDAS, não do QuantX privado nem de uma validação
externa das quatro famílias FEBUS.

**A evidência do produto não vem apenas do nome do dataset:** o cabeçalho textual
do SEG-Y nativo declara `Trace amplitude is proportional to fibre strain-rate`
e recomenda integrar no tempo somente para comparação com geofones. O gauge
length de 10 m e espaçamento ~1,021 m são descritos na documentação da fonte.
Não se afirma que o fator de calibração absoluta seja igual ao FEBUS.

Foi baixado um HDF5 de **46.590.272 bytes**, 30.000 amostras × 384 canais,
seguido de seu SEG-Y nativo de **46.175.760 bytes**. A auditoria encontrou:

- **1.000 amostras/s**: `sampling_interval_seconds=0.001` no HDF5 e intervalo
  nativo SEG-Y de **1.000 µs**. O `t` deste HDF5 está em **segundos desde epoch**,
  apesar de o README genérico descrever microsegundos. Inferência anterior baseada
  apenas no README foi rejeitada antes de qualquer treino/avaliação de modelo.
- **Todos os valores do waveform são bit-exatos** entre HDF5 e SEG-Y após trocar
  a ordem dos bytes. A conversão não filtrou/integrou o sinal deste arquivo.
- O SHA3-256 do SEG-Y coincide com a proveniência gravada no HDF5:
  `f51fd2b69a5e0d42677da5c71995c6852fe7e2dea5cd0bcecaf264a9cd4783ca`.
- Horário absoluto inconsistente: o nativo indica 16:03 UTC, enquanto os campos
  convertidos do HDF5 indicam 10:03; o diretório também não é prova de data de
  aquisição. Não concatenar arquivos ou alegar independência por esses campos.
  A cadência interna foi verificada separadamente.
- Em cinco janelas de 2.048 amostras, todos os 384 canais: cosseno lag1 temporal
  ~0,877–0,902, adjacente espacial ~0,964–0,978 e rank espacial efetivo ~10,7–16,2.
  Isso caracteriza estrutura aproveitável nessa amostra, **não** limite de taxa
  nem ganho de transferência. O conteúdo observado é de baixa frequência; não
  representa automaticamente as bandas até 10 kHz do alvo FEBUS.

Artefatos válidos: `porotomo_pretraining_preflight_v3_20261110/` e
`porotomo_native_identity_20261110/`, em `results/experiments/`. Os diagnósticos
inválidos anteriores foram preservados e marcados, sem alterar os dados brutos.

**Decisão:** PoroTomo é um candidato público mais bem fundamentado que os dois
arquivos single-pulse de intensidade inspecionados anteriormente. Foi iniciada
somente a preparação CPU de um subconjunto fixado antes de inspecionar seus
waveforms: **15 arquivos por estrato de diretório em quatro dias**, 60 arquivos,
teto de 3 GiB de brutos. A seleção usa quantis lexicográficos das listas completas,
sem procurar exemplos com boa reconstrução neural. Preservar todos os canais e
janelas completas de 1.000 amostras: aqui cada janela é **1 s**, não 50 ms.
Os quatro estratos temporais **não são quatro famílias de fibra**.

Ainda não foi lançado treino com PoroTomo. Sua prioridade posterior dependerá do
piloto QuantX e da integridade do subconjunto; se usado, comparar investimento de
updates contra FEBUS direto, sem tomar a coerência observada como prova de ganho.
Sempre que uma referência FEBUS anterior for reutilizada, verificar pesos fonte,
sequência de dados, optimizer, seeds e bytes, em vez de repetir compute sem necessidade.

Outras fontes consultadas, sem treino ou download de sinais: Harper Adams/Dryad
(produto disponibilizado a 100 Hz e ACFs processadas, menor prioridade) e
[MAEPD, 2026](https://doi.org/10.3390/s26072057). MAEPD faz reconhecimento sobre
imagens redimensionadas/transformadas e denoising prévio: suas acurácias não são
evidência de reconstrução do ruído original ou codec competitivo, nem seus pesos
ViT são inicialização direta desta CNN VQ. Não abrir nova arquitetura apenas
por existir um modelo chamado foundation model.
