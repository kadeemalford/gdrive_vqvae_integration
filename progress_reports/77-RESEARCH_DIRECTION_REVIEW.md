# Revisão de direção: eficiência de representação e próximos experimentos FEBUS

**Data da análise: 31/10/2026.** Revisão dos documentos, código, artefatos existentes
e literatura. Nenhum treinamento, otimização de latentes ou nova inferência neural
foi executado. As intervenções abaixo são propostas, não experimentos iniciados.

## 1. Recomendação principal

**Priorizar aprendizado conjunto de taxa–distorção e diagnóstico da escolha dos
índices, antes de outra ampliação da CNN, continuação de ROI ou prior pós-treino.**

O histórico caracteriza bem várias limitações do VQ-VAE atual. Ainda não caracteriza
a fronteira dos codecs neurais FEBUS em geral. Grande parte dos estudos preservou
o mesmo mecanismo: um vetor de 128 dimensões é substituído por um dos 2.048 códigos,
em uma grade fixa, com reconstrução de janelas independentes de 50 ms. Mais largura,
outras convoluções e refinamentos mudaram partes desse sistema, mas não testaram
integralmente uma representação aprendida em função do custo real de seus símbolos.

Ordem recomendada:

1. **Diagnóstico com decoder/codebook congelados:** buscar índices melhores para o
   mesmo dado, mantendo o orçamento. Investigar se a codificação atual deixa ganho
   de fidelidade acessível ao mesmo decoder.
2. **Primeiro novo estudo de treinamento, quando autorizado:** codec neural com
   quantização escalar e prior simples, treinados conjuntamente por taxa–distorção;
   depois, hyperprior como ablação. Usar o VQ atual como referência prática.
3. **Contexto físico maior/condicional:** avançar se diagnósticos de dependência
   temporal/espacial sustentarem essa hipótese, sem descartar bandas do alvo.

VQ com objetivo entrópico diferenciável é um controle especialmente pertinente para
separar o efeito da otimização conjunta do efeito de trocar a família de quantização.

## 2. O que o histórico efetivamente mostrou

### Houve ganhos à mesma taxa

A percepção de que o ganho mais forte veio de mais índices é sustentada pela curva
de taxas, mas não descreve todos os resultados. No [relatório 22](22-ARCHITECTURE_INITIALIZATION_RESULTS.md),
a referência transferida passou de aproximadamente **0,810 para 0,720 de RMSE**
e **0,560 para 0,682 de correlação**, mantendo aproximadamente **165,9× nominais**.
O regime de otimização e a exposição adicional importaram; o contraste não isola
qual deles foi responsável.

Na confirmação A/B/C, o transferido chegou a **0,416764 / 0,904839** em
RMSE/correlação, contra **0,457670 / 0,889594** do histórico compatível, ambos a
**40,367× nominais** ([relatório 64](64-CROSS_FOLD_CONFIRMATION_RESULTS.md)).
Isso é avanço de fidelidade à mesma taxa, embora com regressões localizadas.

### Aumentar largura não mudou o patamar

O [relatório 66](66-CAPACITY_COMPACT_RESULTS.md), conferido em
`results/experiments/native_capacity_compact_fold1_20261029/results.json`, mostra:

| Controle | Parâmetros | RMSE macro | Correlação | Taxa nominal |
|---|---:|---:|---:|---:|
| CNN menor | 27.958.081 | 0,684932 | 0,716521 | 157,357× |
| CNN maior | 96.517.033 | 0,674562 | 0,724291 | 157,357× |

Ganho relativo de RMSE de **1,51%**, com regressão importante da cauda de
single-mode 11. Treino e validação ainda melhoravam: não há evidência para explicar
o resultado apenas por overfitting, mas tampouco para esperar salto de qualidade
ao ampliar novamente a largura. Foi uma comparação de largura, uma seed e um fold;
não um teste de toda arquitetura grande ou de todo mecanismo de contexto.

### Resultados posteriores ao relatório 66 também limitam as próximas apostas

| Linha | Resultado relevante | Consequência |
|---|---|---|
| Refresh de VQ | Mais códigos, ganho agregado pequeno e regressões; relatório 26. | Ocupação do codebook não é objetivo de fidelidade. |
| Especialistas | Ganho pequeno frente ao controle de exposição, mais modelos e regressões; relatório 28. | Baixa prioridade para outro MoE com a mesma representação. |
| Loss espectral/por canal | Magnitude ou caudas podem melhorar sem preservação conjunta; relatórios 4 e 58. | Outra loss isolada não é a principal aposta estrutural. |
| CNN anisotrópica | Não superou a transferência nas duas coortes; relatórios 63–64. | Já houve teste estrutural; não recomendar simplesmente “usar CNN multiescala”. |
| Residual VQ | RMSE melhorou apenas 0,045% sem guarda; três códigos ao final; relatório 70. | Não prolongar esse refiner inalterado. |
| Residual FSQ | Ganho de RMSE de 3,65% contra a base, com mais bytes; relatório 71. | FSQ funcionou melhor nesse refiner, não demonstrou superioridade geral à mesma taxa. |
| Híbrido a bytes pareados | Ganho relativo de RMSE de 0,023%–0,356%, com trade-off de cauda; relatório 73. | Complementaridade pequena para o custo de vários modelos. |
| Prior neural congelado | Economizou 2,41% nos frames, mas perdeu 0,75% no total incluindo pesos; relatório 75. | Resultado negativo daquele contrato/coorte; não refuta treinamento conjunto. |

O residual dos relatórios 69–73 codifica **erro no sinal com a base congelada**.
Não é uma avaliação de RVQ nos embeddings treinada ponta a ponta, nem de um
codec hierárquico em que a base se reorganiza para colaborar com o refinamento.
Essas hipóteses permanecem distintas, mas não são a prioridade recomendada aqui.

### Atualização encontrada nos artefatos do relatório 76

O estudo `transition_index_coding_20261030` já está **concluído**, com timestamp
de término de 30/10/2026 20:48 UTC. `REPORT.md` e `results.json` registram:

| Coorte/modo | Razão real com transições | Economia contra marginal/range | RMSE macro preservado | Correlação preservada |
|---|---:|---:|---:|---:|
| Fold 0 compacto | 173,217× | 0,186% | 0,667586 | 0,727658 |
| Fold 0 denso | 45,906× | 0,125% | 0,445187 | 0,888093 |
| Fold 1 compacto | 165,839× | 0,308% | 0,648189 | 0,749319 |
| Fold 1 denso | 44,382× | 0,086% | 0,416764 | 0,904839 |

São streams com mapas, tabelas e fronteiras; pesos de waveform compartilhados,
contabilizados separadamente nos artefatos. As linhas usam configurações/modelos
diferentes entre folds. Não combiná-las como uma trajetória de um único modelo.

Essas economias são úteis, mas não justificam esperar uma transformação da
fidelidade a partir de outra pequena mudança de empacotamento. O negativo do prior
neural também precisa ser contextualizado: seus pesos podem ser amortizados em
operações maiores, mas a economia medida de payload continua pequena.

## 3. Três lacunas que merecem atenção

### 3.1 O reconstrutor não aprende o custo entrópico dos seus códigos

Em `finetuning/finetune_vqvae.py`, linhas 390–404, a perda é essencialmente:

```text
0.5 * MSE + 0.5 * L1 + codebook_loss + 0.25 * commitment_loss
```

Há termos opcionais espectral/por canal, mas não taxa em bits. Em
`../smartgrids/vqvae/train_vqvae.py`, linhas 139–156, a escolha é pelo código mais
próximo em distância euclidiana no embedding. O prior posterior aprende a codificar
os índices que esse mecanismo já escolheu.

**Isso é uma formulação válida de distorção com orçamento nominal fixo**, não um
bug. Quando todos os índices custam exatamente 11 bits e a grade é fixa, a taxa
é constante. A lacuna aparece ao buscar eficiência com entropy coding: encoder,
quantização e prior não foram treinados conjuntamente para negociar fidelidade
versus comprimento do stream.

Um objetivo conjunto seria, esquematicamente:

```text
L = D(x, x_hat) + lambda * R
R = (-log2 p(y_hat | z_hat) - log2 p(z_hat)) / N_cells
```

`z_hat` só existe se houver hyperprior. Side de normalização, escapes, mapas e
container continuam cobrados nos bytes medidos. Aumentar lambda pode piorar
fidelidade e reduzir taxa; **não é um ganho gratuito no mesmo ponto**. A hipótese
se confirma apenas se a curva completa melhorar frente aos controles.

Também não basta somar cross-entropy sobre `argmin` inteiro ao treino VQ: isso
treina o prior, mas não fornece automaticamente o gradiente de taxa ao encoder.
É necessário um estimador/relaxação apropriado e avaliar índices discretos exatos.

### 3.2 Janela em pixels não equivale a contexto físico

O offshore usava 1.000 amostras em **5 s**, a 200 Hz; os manifestos atuais FEBUS
usam 1.000 amostras em **0,05 s**, a 20 kHz. Frequência no título do arquivo não
é sample rate. O README do Drive é orientativo e tem descrições de famílias
ambíguas; metadados HDF5 e inventário auditado são as referências operacionais.

Uma janela de 50 ms não contém um ciclo completo de 1 ou 10 Hz. Aumentar dilatação
dentro dessa janela não oferece o contexto temporal que ficou fora dela. Ao mesmo
tempo, a auditoria histórica encontrou bastante energia em 5–10 kHz e coerência
espacial relevante: “usar mais contexto” não autoriza decimar essas componentes.

Os NPZ atuais não devem ser simplesmente concatenados para criar um sinal longo:
os offsets preparados, por exemplo 0/1052/2105, deixam intervalos entre janelas.
Um estudo temporal precisa recortar HDF5 contínuo e tratar o overlap entre blocos.

### 3.3 A ponderação de treino mudou de significado após o tiling

Contagem nova, somente dos metadados de
`data/finetune/spatial_band_v1_fold0_20261024/samples_manifest.csv`:

- Treino: **11.760 janelas, 87 arquivos e dez grupos/configurações**.
- Tiles com até três canais: **1.200 janelas, 10,20% do treino**, mas somente
  **0,262% das células**.
- Validação: esses tiles são 18,01% das janelas e 0,387% das células.

O treino atribui uma atualização e uma perda média a cada janela; a validação nova
pondera tiles por células dentro da configuração. Logo, há um desalinhamento a
quantificar. A participação de 10,20% nas atualizações **não** mede sua participação
nos gradientes ou explica causalmente o erro. Antes de outro sampler, medir perdas
e normas de gradiente por tamanho/configuração e comparar pesos esperados com o
seletor. Eventual controle deve manter todas as regiões, inclusive as pequenas.

O balanceamento antigo por família já foi testado; este diagnóstico é específico
do tiling posterior. Não é justificativa para repetir a mesma busca de samplers.

## 4. Literatura: o que realmente informa a decisão

Revisão dirigida, não sistemática. Distinguir fonte com método/código auditável de
resumo. Nenhum resultado abaixo demonstra alta fidelidade a 700× neste corpus.

| Fonte | Evidência consultada | Aplicação proposta e limite |
|---|---|---|
| **Ballé et al., ICLR 2018 — Variational image compression with a scale hyperprior** ([texto](https://arxiv.org/html/1802.01436v2)) | Texto integral: transformadas neurais, quantização escalar, taxa–distorção conjunta e side do hyperprior. | Principal referência de desenho de um codec com custo de informação aprendido; resultados de imagens naturais não são promessa FEBUS. |
| **AI4EPS/ZipDAS** ([repositório](https://github.com/AI4EPS/ZipDAS), [modelo](https://github.com/AI4EPS/ZipDAS/blob/main/zipdas/model.py)) | Código: `BLS2017Model`, prior fatorizado, compress/decompress e `bpp + lambda * distortion`. | Precedente diretamente DAS para uma família neural diferente de VQ. Não é benchmark pronto: `mse` no método consultado calcula L1, há caminhos de normalização/log e o trainer usa a mesma fonte para treino/validação. Adaptar ao split/alvo FEBUS, não copiar scores/defaults. |
| **Yang, Bamler e Mandt, NeurIPS 2020 — Improving Inference for Neural Image Compression** ([texto](https://arxiv.org/html/2006.04240v4)) | Texto integral: otimização da representação na codificação, com modelo fixo; lacunas de amortização/discretização. | Fundamenta a busca de índices com decoder congelado. O método publicado usa latentes escalares; busca em codebook VQ é adaptação experimental, não reprodução literal. |
| **Jiang et al., CVPR 2026 — RDVQ** ([texto](https://arxiv.org/html/2604.10546v1), [conferência](https://cvpr.thecvf.com/virtual/2026/oral/40298)) | Texto integral: distribuição suave baseada em distâncias permite gradiente de taxa para encoder; reconstrução/codificação usam VQ duro. | Controle causal interessante: VQ atual versus VQ com custo entrópico aprendido. Aproveitar o mecanismo, não GAN/LPIPS nem completar tokens omitidos. O artigo avalia fidelidade perceptual e tem ~252M parâmetros; não é receita leve comprovada de fidelidade científica. |
| **Li, Li e Lu, NeurIPS 2021 — Deep Contextual Video Compression** ([artigo](https://arxiv.org/abs/2109.15047)) | Resumo e página oficial: codec condicionado em contexto de features para reduzir redundância entre frames. | Inspira codec temporal condicionado em histórico já decodificado. Resultado é de vídeo; não copiar movimento óptico ou sua economia percentual para DAS. |
| **Wang et al., Applied Optics 2024 — RAE para DAS** ([editora](https://doi.org/10.1364/AO.527625)) | Resumo e origem dos dados: RNN/LSTM, 40,60 dB em CR=8; menciona CR máximo 512. Texto integral restrito. | Motiva contexto temporal. O resumo não permite atribuir 40,60 dB ao ponto 512× ou verificar todos os custos e generalização. |
| **Pavelkin et al., ORCAS, preprint 2026** ([texto](https://arxiv.org/html/2609.22140v1)) | Texto integral: representação ortogonal aprendida, quantização e stream; ~13× com SNR ~13 dB em seu protocolo. Há filtragem e reamostragem para 100 Hz antes da compressão. | Contextualiza a dificuldade e a importância do alvo. Não é evidência de reconstrução do bruto FEBUS 20 kHz nem indicação para substituir o produto neural por método clássico. |

Os artigos RAE e ViT-DAS citados no roadmap não sustentam escolher LSTM ou
Transformer somente por anunciarem CR elevado. A pergunta útil é **que redundância
exploram, o que efetivamente transmitem e que sinal avaliam depois do preprocessing**.

## 5. Experimentos propostos e critérios de decisão

### E0 — Diagnóstico do sinal e dos denominadores

Antes de comprometer compute em outra rede, usar uma amostra fixa do treino para:

- Examinar dependência espacial/temporal, espectro de covariância e erro residual
  por bandas físicas; incluir variação do contexto de 50 ms para 0,25–1 s quando
  houver trecho contínuo. São comprimentos experimentais, não ótimos publicados.
- Localizar erro em bordas versus interior e em canais por faixa de amplitude.
- Reconstituir arquivos para métricas de baixa frequência e quantis por canal;
  média de p90 de janelas não é p90 pooled do arquivo.
- Confrontar erro absoluto e relativo. Um canal quase constante pode produzir
  erro relativo enorme; isso exige contexto físico, não retirada da avaliação.

PSD plana, SVD ou fracasso de um preditor simples não provam ruído independente
nem um limite informacional. Servem para escolher entre hipóteses de estrutura.

### E1 — Busca de códigos com o mesmo decoder

**Pergunta:** o codec já contém representações melhores, mas o encoder/nearest-neighbor
não as encontra?

1. Fixar um compacto transferido de cada coorte, seu codebook, normalização e side.
2. Selecionar 32–64 janelas por metadados, cobrindo grupos e tamanhos; configurar
   método/orçamento no treino antes da avaliação na validação.
3. Partir dos índices atuais e explorar alternativas discretas com orçamento
   computacional limitado, por exemplo candidatos próximos no codebook com busca
   orientada pelo erro após o decoder. Relaxações só auxiliam a busca.
4. Medir sempre a reconstrução de embeddings exatos recuperados dos índices.
5. Primeiro, comparar a taxa fixa de 11 bits, com mesmo número de índices. Depois,
   repetir sob teto do stream real: trocar frequências dos índices pode mudar bytes.
6. Reportar tempo de codificação, ganho por grupo e taxa; o decoder não muda.

Formalmente, buscar aproximadamente:

```text
min_k D(x, decoder(codebook[k]))   sujeito a B(k, side) <= B_baseline
```

O encoder tem acesso ao original: essa busca é uma operação de codificação
implementável, não envio gratuito do alvo. O decoder recebe somente os índices.
É diferente dos probes de memorização, que alteraram pesos, e dos bypasses, que
transmitiriam latentes contínuos muito maiores.

**Se ganhar claramente:** há oportunidade comprovada de melhorar a codificação
com esse decoder; considerar encoder melhor, destilação da busca ou quantização
orientada à distorção. **Se não ganhar:** a busca pode ter falhado localmente;
não concluir que a melhor reconstrução possível foi encontrada.

Manter a solução inicial garante não piorar o objetivo escolhido no encoder;
essa propriedade não é evidência independente de preservação das outras métricas.
O decoder usa GroupNorm sobre dimensões espaciais/temporais, portanto não presumir
que trocar um índice afeta exclusivamente uma vizinhança local na avaliação.

### E2 — Codec treinado para fidelidade por bit

**Hipótese principal de novo treinamento:** uma transformada neural com quantização
escalar e prior aprendido utiliza melhor o orçamento do que a VQ fixa atual.

Desenho econômico e separável:

| Braço | Função |
|---|---|
| VQ atual + melhor stream existente | Referência prática congelada nas duas taxas. |
| Quantização escalar + prior fatorizado | Primeiro candidato; backbone de porte moderado, mesmo alvo/corpus e distorção temporal de referência. |
| Mesmo candidato + hyperprior | Ablação posterior: medir se o side adicional realmente se paga. |
| VQ + termo entrópico diferenciável | Controle para separar aprendizado de taxa da família de quantização. |

Não lançar todos automaticamente. Primeiro verificar gradiente de taxa, calibração
de bits estimados versus reais e viabilidade de uma curva pequena. O candidato
escalar não recebe o latente contínuo gratuitamente: quantizar e serializar todos
os coeficientes, hyperlatentes e parâmetros auxiliares necessários.

Usar **bits por célula retida** como eixo principal. Os pontos atuais reais são
aproximadamente **0,185–0,193** no compacto e **0,697–0,721** no denso. Começar
perto deles evita comparar uma rede nova muito mais cara com o compacto. Um ponto
de **0,10 bit/célula (~320× sobre float32)** pode investigar maior compressão;
**0,0457 (~700×)** é uma extensão condicionada ao resultado, não requisito para
autorizar a triagem inicial.

Mesma contagem de tokens não é o controle adequado entre famílias de quantização:
comparar bytes completos e curvas sobrepostas. Manter distorção/split/cobertura
comuns no primeiro contraste; mudanças de backbone e de loss física ficam em
ablações próprias. Para uma comparação causal de treinamento, incluir controles
com orçamento/inicialização compatíveis e declarar o pré-treino histórico de A.

### E3 — Contexto mais longo, se houver estrutura aproveitável

Comparar codec independente com um condicionado em histórico **já decodificado**,
mantendo orçamento por segundo/célula. Pode usar CNN temporal dilatada ou estado
recorrente em resolução reduzida; atenção é uma escolha de implementação posterior.

- Reconstruir todas as amostras/bandas e cobrar quadros de reinicialização, estado
  que precisar ser transmitido e side. Estado derivável do stream não é skip grátis.
- Avaliar propagação de erro, início de arquivo, mudanças abruptas, acesso aleatório
  e latência. Não dar ao decoder o passado original enquanto o encoder usa esse
  passado sem simular sua reconstrução.
- Separar contexto para **probabilidades** de contexto para **síntese**: o primeiro
  reduz bytes de códigos fixos; o segundo pode mudar a informação que precisa ser
  transmitida para reconstruir o bloco seguinte.

Isso difere tanto do prior checkerboard dentro de uma janela quanto do refiner
do erro de uma base congelada. Se o erro mostrar pouca dependência entre janelas,
essa linha perde prioridade; maior contexto não resolve inovação imprevisível.

## 6. O significado de 700× e de fidelidade

Para referência float32, `R = 32 / CR` bits por célula:

| Razão | Bits/célula |
|---|---:|
| 40× | 0,8000 |
| 80× | 0,4000 |
| 160× | 0,2000 |
| 320× | 0,1000 |
| 700× | 0,0457 |

700× significa aproximadamente **um bit para 22 valores de entrada**, antes de
reservar parte do orçamento para side. É possível apenas com muita estrutura
aproveitável e/ou tolerância à distorção; não depende só de capacidade da rede.

Como ilustração, para uma fonte idealizada **gaussiana i.i.d.**, com erro quadrático
e variância sigma², `D(R) = sigma² * 2^(-2R)`. Em 0,0457 bit/amostra, o melhor
RMSE normalizado desse modelo teórico é aproximadamente **0,969**. Essa conta
**não é um limite medido para FEBUS**, que tem dependências espaciais/temporais e
estatísticas heterogêneas; ilustra por que preservar a realização de componentes
imprevisíveis pode ser caro. Não inferir gaussianidade/independência de flatness.

Adotar erro absoluto e relativo conjuntamente, com limites físicos definidos
pela aplicação. Preservar ruído significa medir o erro contra sua realização
original; gerar ruído parecido ou acertar apenas a PSD não satisfaz esse objetivo.
Correlação ~0,90 tampouco significa 90% das amostras reconstruídas corretamente.

Como **critério de investimento sugerido**, um próximo ramo estrutural deveria
buscar algo claramente maior que os refinamentos marginais recentes: por exemplo,
≥5% de redução relativa de RMSE a bytes não maiores, ou ≥10% de economia a
fidelidade equivalente segundo tolerâncias predefinidas, com confirmação na outra
coorte. Esses números são propostas de gestão experimental, não limiares físicos
validados nem significância estatística. Examinar configurações/canais junto à
média e não relaxar retrospectivamente as guardas dos estudos encerrados.

## 7. Dados e conclusão

O corpus auditado cobre 207 HDF5 e 23 grupos, mas a auditoria temporal histórica
estimou apenas cerca de 15 minutos na soma das uniões de clock por grupo. Isso não
é duração independente exata, e tiling adicional não cria novas aquisições.
Diversidade limitada é um risco de generalização, não prova de que mais dados
resolveriam a fidelidade a taxa compacta.

A pasta DAS do Drive contém outras condições, como EDAS/orientações, potencialmente
úteis para um holdout externo após verificar origem, compatibilidade e relação
com os dados existentes. Não presumir independência só por estarem em outra pasta.

**A decisão recomendada é mudar a pergunta experimental:** deixar de perguntar
principalmente quanto uma CNN maior melhora com o mesmo gargalo e investigar
quanta fidelidade adicional pode ser extraída de cada bit, com um encoder melhor
e/ou uma representação aprendida junto com seu custo de codificação. O histórico
permite priorizar essa pergunta; ainda não permite prometer 700× com alta fidelidade.

### Fontes locais e remotas principais

- `README.md`, roadmap completo e relatórios de resultados citados acima.
- `../smartgrids/AGENTS.md`, `vqvae/train_vqvae.py`, configuração do checkpoint
  original e preparação OOI.
- `finetuning/model_factory.py`, `reconstruction_objective.py`,
  `finetune_vqvae.py`, `physical_validation.py`.
- `results/experiments/native_capacity_compact_fold1_20261029/results.json`.
- `results/experiments/transition_index_coding_20261030/{REPORT.md,results.json,STATUS.json}`.
- Manifesto espacial citado na seção 3.3; contagens calculadas por leitura CSV,
  sem abertura de sinais de teste ou execução de modelos.
- [README da pasta DAS, lido via MCP do Drive](https://drive.google.com/file/d/1Eh_ovULjX_EbTAVzqoyLY2VnMjLlHMXT/view).

Para reproduzir as frações da seção 3.3: filtrar o CSV por `split`, contar linhas
com `channels <= 3` e dividir pelo total; para a fração de células, usar as somas
de `channels * time_samples` no numerador e no denominador. Os ratios e métricas
da seção 2 vêm dos artefatos existentes, não de uma reavaliação nesta revisão.
