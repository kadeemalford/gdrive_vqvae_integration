# Reconstrução fiel de DAS FEBUS: viabilidade, limites e novo direcionamento

**Revisão iniciada em 16/11/2026 e concluída em 17/11/2026, após o encerramento
dos contrastes da revisão 90.**
Documento de análise e recomendações; não é um novo resultado de codec.

Leitura dirigida: [conclusão executiva](#1-recomendação-executiva),
[reinterpretação dos negativos](#3-o-que-os-negativos-realmente-acrescentam),
[limites informacionais](#5-limitações-informacionais-o-que-podemos-e-o-que-não-podemos-concluir),
[curadoria](#6-curadoria-o-recurso-escasso-é-cobertura-física-identificável),
[literatura](#7-estado-da-arte-confrontar-o-alvo-antes-de-comparar-números),
[próximas avaliações](#9-próximas-perguntas-com-decisões-falsificáveis) e
[factibilidade](#11-avaliação-final-de-factibilidade).

## 1. Recomendação executiva

**Recomendo mudar a pergunta principal de “qual arquitetura testar agora?” para
“qual parcela do erro é recuperável, a que custo e em quais aquisições?”.**
O próximo investimento deve produzir um diagnóstico de viabilidade que discrimine
limitação de dados, de representação/otimização e de taxa. Uma nova rede deve ser
consequência desse diagnóstico.

Minha avaliação atual é:

1. **Ganhos adicionais são plausíveis, mas um salto generalista de fidelidade no
   mesmo orçamento ultracompacto tem sustentação fraca.** O histórico reduz a
   expectativa de que mais largura, contexto genérico, um novo quantizador ou
   outra ponte de pré-treino resolvam o problema. Não fornece, porém, uma prova de
   impossibilidade.
2. **A escassez mais clara é de diversidade identificável e de evidência externa.**
   No treino fold0 há somente uma configuração doped e uma multimode. Milhões de
   células dessas configurações ajudam a estimar padrões locais, mas não ensinam
   automaticamente a distribuição de novas montagens, instrumentos ou condições.
3. **O custo de preservar a realização do ruído pode ser o obstáculo dominante
   quando a exigência de fidelidade aumenta.** Mais dados podem ensinar sua
   distribuição; não revelam ao receptor os valores novos de uma inovação
   independente. É necessário distinguir essa possibilidade de erro criado pelo
   próprio codec.
4. **Ainda existem lacunas experimentais importantes.** H4 avaliou precisões
   extremamente estritas; os estudos recentes de diversidade/FiLM treinaram em
   crops de oito canais e avaliaram tiles completos; vários testes foram
   limitados a operadores ou objetivos estreitos. Essas limitações restringem
   conclusões gerais, sem invalidar seus resultados negativos.
5. **Se houver oportunidade mensurável, a candidata neural mais justificada é
   um modelo probabilístico pequeno que codifique diretamente valores
   quantizados, com precisão explícita e sem depender de uma base VQ completa.**
   Ele aprenderia probabilidades de detalhes observados, em vez de tentar
   sintetizar detalhes ausentes. A prioridade depende de demonstrar economia
   condicional líquida; não recomendo iniciá-lo por suposição.

O produto científico imediato deveria ser uma **matriz de viabilidade por
regime: fidelidade exigida × bits completos × generalização × custo**.
Ela pode justificar uma arquitetura melhor, mostrar que são necessárias novas
aquisições ou concluir que determinada meta de compressão não é defensável com
os dados e requisitos atuais.

**700× deve permanecer hipótese exploratória, não promessa ou critério que
force a interpretação dos resultados.** A meta intermediária de reduzir erro
em ≥5% à mesma taxa, ou bytes em ≥10% a fidelidade equivalente, continua útil
como referência de investimento; não há evidência para prometer nem esses ganhos.

### O que muda em relação à minha revisão 90

A revisão 90 apostou prioritariamente em dependência espacial e learned lifting.
H1/H2 foram implementados e não deslocaram a referência prática. Minha atualização
é reduzir a prioridade de outra transformada baseada apenas em correlação
espacial: **a existência de redundância não demonstra que reste muita redundância
barata além daquela já aproveitada pelo VQ maduro**.

A prioridade passa a ser identificar a parcela efetivamente recuperável e
conseguir dados que permitam distinguir as causas. Isso é uma mudança na ordem
de investimento, mantendo o objetivo de reconstruir o waveform medido.

## 2. Base documental e verificações desta revisão

Foram consultados README, roadmap, os dois checkpoints de progresso, a revisão
[90](90-FIDELITY_PER_BIT_NEXT_RESEARCH_REVIEW.md), as etapas posteriores e fontes
anteriores relevantes, com aprofundamento em capacidade/overfit, cobertura,
normalização, codificação e generalização. Foram examinadas implementações de
métricas, geometria, treinamento, normalização interna e preparação, incluindo
a CNN nativa em `../smartgrids/vqvae/train_vqvae.py`.

Também foram consultadas fontes primárias de literatura, código e catálogos de
dados, descritas nas seções 7, 8 e 12. A busca é dirigida, não uma revisão
sistemática de toda a literatura.

### Verificação local nova, restrita a metadados e resultados existentes

Artefatos:

```text
results/experiments/feasibility_review_20261116/
  metadata_review.py
  metadata_review.json
```

Reprodução, a partir da raiz:

```bash
.venv/bin/python results/experiments/feasibility_review_20261116/metadata_review.py
```

O script:

- confere o manifesto contra o SHA-256 do protocolo P0;
- reconta arquivos, configurações, janelas e células, inclusive o subconjunto
  de treino fora dos arquivos de check;
- confere os hashes dos 13 resultados recentes registrados no fechamento;
- preserva hashes dos relatórios 1–105;
- calcula a escala efetiva das precisões H4, a geometria nominal e exemplos
  matemáticos hipotéticos de taxa–distorção.

As contagens analíticas de índices foram adicionalmente confrontadas com
`finetuning.latent_geometry.latent_positions`, sem instanciar uma rede.
**Zero updates, zero inferência neural, zero uso de GPU e nenhuma nova leitura
de waveforms brutos ou preparados.** Metadados do manifesto, inclusive os nomes
dos splits, não são avaliação dos sinais reservados.

A análise do acervo Drive usa o censo já documentado em
[98](98-PHYSICAL_METADATA_AND_PROVENANCE.md). Ele foi reutilizado, sem nova
listagem do Drive nesta revisão. Não foi incorporado dataset externo.

## 3. O que os negativos realmente acrescentam

### 3.1 Síntese causal e prática

| Evidência | Conclusão sustentada | Inferência que continua aberta |
|---|---|---|
| [92 — H1](92-LEARNED_LIFTING_STUDY.md): operadores pequenos, quatro braços, coder e alocação | A representação testada não supera a referência prática. Recodificar coeficientes poupou ~53–60% nos pontos destacados, sem promover H1. | Uma implementação pequena com prior fatorizado e treino/coder inicialmente desalinhados não representa toda compressão neural multirresolução. |
| [93](93-NEURAL_ANCHOR_INNOVATION_STUDY.md) e [95 — H2](95-FROZEN_DEVELOPMENT_VALIDATION.md) | Dependência espacial foi útil: na validação consumida, H2 teve 8,48% menos erro e 1,50% menos bytes que o controle independente. | O ganho não superou o VQ maduro: 25,12% mais erro e 12,07% mais bytes. “O mecanismo funciona” e “o codec é competitivo” são perguntas diferentes. |
| [94 — observação completa](94-ENCODER_OBSERVATION_STUDY.md) | Acrescentar observações ao encoder recuperou pouco neste desenho. | Não mede o melhor uso possível dessas observações em outra representação. |
| [96 — residual e latentes](96-FULL_TILE_BOTTLENECK_DIAGNOSTIC.md) | Lag1 do erro é muito menor que o do alvo; corretores lineares locais têm oportunidade limitada nessa medição. | Erro descorrelacionado não é necessariamente ruído instrumental independente. |
| [97 — produto/curva](97-DENSE_PRODUCT_AND_RATE_MATCH_STUDY.md) | Economia com mais erro; a grade medida não demonstrou dominância RD. | Taxas sem sobreposição e curva finita não demonstram equivalência nem ótimo universal. |
| [100 — normalização interna](100-LOCAL_INTERNAL_NORMALIZATION_STUDY.md) | A intervenção local residual não melhorou a fronteira. | Não foi uma substituição integral de GroupNorm nem teste de todo modelo invariável ao tamanho de tile. |
| [102 — índices discretos](102-DIRECT_DISCRETE_INDEX_STUDY.md) | Ganho de 0,994025% em NRMSE* veio com pior SSE em oito configurações e erro espectral em 25/30 pares. | Uma pequena melhora de score pode redistribuir erro, sem recuperar mais waveform de maneira física abrangente. |
| [103 — H4](103-ERROR_BOUNDED_RESIDUAL_STUDY.md) | Garantia física funcionou; a base VQ quase não se pagou. O MLP causal poupou 0,37821% com reconstrução idêntica. | Não é uma medida do custo mínimo em toda precisão nem teste de um modelo completo de distribuição condicional. |
| [99 — H5](99-CLOCK_ALIGNED_MULTICORE_OPPORTUNITY.md) | Triagem linear e buckets de magnitude não geraram oportunidade líquida; 24/24 fallbacks por braço no segundo desenho. | Não certifica independência dos núcleos nem mede sua entropia condicional ideal. |
| [104 — composição/diversidade](104-SEMISYNTHETIC_AUGMENTATION_STUDY.md) | A composição piorou o controle real; a curva 4/6/10 foi não monotônica a dose fixa. | Não testa mais sessões independentes nem estima uma lei de escala de dados. |
| [105 — hardware/FiLM](105-PHYSICAL_CONDITIONING_STUDY.md) | Hardware falhou contra constante/categoria nos critérios registrados. | Não isola física de configuração em um corpus com poucos vetores de hardware e sem sessões certificadas. |

Houve, portanto, **ganhos técnicos e mecanísticos reais**, mas nenhum candidato
promovido. A ausência de promoção não torna todos os resultados equivalentes a
zero. O histórico anterior também inclui avanços de adaptação à mesma taxa,
como os ~8,94% de RMSE no contraste do relatório 64, além de economia lossless
de índices; os checkpoints [1](PROGRESS_CHECKPOINT_1.md) e
[2](PROGRESS_CHECKPOINT_2.md) discriminam esses resultados.

Os **356.352 updates** desde 90 incluem passos de quantização H1, CNNs e MLPs.
Não representam uma única rede grande treinada por esse orçamento, nem ensaios
independentes. Muitos contrastes compartilham fonte, dados, checks, coder e
gargalo. Não se pode transformar a quantidade de negativos em uma probabilidade
estatística de impossibilidade.

### 3.2 H4 cobriu alta precisão; não a transição inteira até o VQ

Há um detalhe decisivo na interpretação das taxas de H4.
Em `finetuning/audit_prospective_preparation.py`, a escala de cada grupo é:

```text
s_g = 0,01 × mediana dos desvios positivos das janelas ordinárias de treino
epsilon_H4 = alpha × s_g
```

Consequentemente:

| Alpha histórico | Epsilon / mediana de desvio usada na calibração |
|---|---:|
| 0,1 | 0,001 = **0,1%** |
| 0,03 | 0,0003 = **0,03%** |
| 0,01 | 0,0001 = **0,01%** |

**Alpha 0,1 não significava erro permitido de 10% do desvio típico.**
Essa mediana é por grupo/janelas, não o desvio de cada canal; o bound relativo
de um canal individual pode ser diferente. Os resultados reportados permanecem
corretos: ~9,19–13,45 bpc para base zero, ~9,20–13,58 bpc com base VQ e NRMSE*
da ordem de 0,00165–0,000165.

O VQ maduro, nos mesmos 136 checks, está perto de **0,490336 / 0,679244 bpc**.
Esses extremos mostram um compromisso grande, mas não localizam a região em
que uma aplicação começaria a considerar a reconstrução aceitável. Alguns
controles anteriores oferecem pontos intermediários, em outras coortes; não
podem ser conectados como uma única curva.

Assim, **o histórico não demonstra que toda melhoria expressiva de fidelidade
exija necessariamente 9–14 bpc**. Também não demonstra que seja possível chegar
perto de erro 0,1 a ~0,7 bpc. Essa lacuna merece uma pergunta de viabilidade,
com contrato novo, e não uma extensão adaptativa dos alphas encerrados.

### 3.3 Há uma mudança de contexto entre treino e avaliação em estudos recentes

Diversidade e condicionamento físico usam **8 × 1.000** no ajuste, com
normalização do crop, mas avaliam tiles completos. A arquitetura usa GroupNorm
e padding; seu comportamento depende do conjunto espacial apresentado.

Na geometria densa `spatial4-temporal4`, conferida por cálculo:

| Shape físico | Índices | Bits nominais de índices / célula |
|---|---:|---:|
| 2 × 1.000 | 1.000 | 5,5000 |
| 8 × 1.000 | 1.000 | 1,3750 |
| 16 × 1.000 | 1.000 | 0,6875 |
| 512 × 1.000 | 32.000 | 0,6875 |

São **bits nominais, sem entropia/side**, não taxas medidas de transporte.
O crop de oito canais oferece o dobro de posições por célula que um tile grande
divisível por 16, além de proporção de bordas e estatísticas distintas.
Igualar apresentações de células não iguala o problema aprendido.

Isso é uma **possível explicação parcial a testar**, não um bug demonstrado nem
motivo para apagar o negativo. O desenho era igual entre braços, portanto seus
contrastes continuam válidos naquele regime; o alcance sobre dados/metadados
em um treino representativo do codec final é que fica limitado.

Também é importante delimitar o teste 100. A implementação é:

```text
GN_global + 0,5 × tanh(gate) × (GN_local − GN_global)
```

A componente global nunca é completamente removida: seu coeficiente fica entre
0,5 e 1,5. Isso preserva a inicialização e testa uma intervenção limitada.
O negativo não certifica que a dependência de contexto da GroupNorm seja
irrelevante em qualquer arquitetura.

**Não proponho repetir FiLM, o sampler ou o gate encerrado.** Proponho que
qualquer diagnóstico futuro de dados/capacidade confira primeiro se o controle
aprende e é avaliado sob contextos compatíveis.

### 3.4 Métrica, treinamento e admissibilidade física não são a mesma coisa

P0 corrigiu um problema real dos seletores antigos em constantes/fracos.
O NRMSE* prospectivo faz pooling por canal/arquivo e usa piso congelado:

```text
RMSE_c / max(desvio_c, s_g)
```

Depois há macros entre canais, arquivos, configurações e famílias.
Esse score não é a raiz de uma MSE física única; tampouco coincide com a
loss normalizada por crop usada por vários braços. O relatório 102 demonstra
concretamente que melhorar essa macro pode aumentar SSE físico.

A guarda “nenhum grupo/métrica pior” é também mais exigente que melhorar uma
curva RD sob uma única distorção. Ela pode vetar redistribuições legítimas para
um uso, mas inadequadas para outro. **Isso não autoriza relaxá-la depois de ver
o resultado.** É necessário definir, com base no uso científico, limites
absolutos e margens de equivalência antes de uma nova seleção.

Até isso existir, temos resultados de pesquisa e guardas conservadoras,
não uma definição operacional completa de “fiel”.

## 4. Quatro limitações que precisam ser separadas

### 4.1 Fonte: o que há de fato para comprimir?

Uma descrição conceitual é:

```text
x = resposta instrumental(campo físico) + componente comum + inovação local
```

Essa decomposição **não foi identificada quantitativamente no corpus**.
O campo pode conter tons, transientes e eventos fracos; o instrumento pode
introduzir correlação, deriva, fading, diferenças espaciais/temporais e ruído
comum. A “inovação local” depende do contexto/modelo considerados, não é
automaticamente ruído branco.

Possibilidades que exigem decisões diferentes:

- **Ruído comum/correlacionado:** pode ser representado uma vez e reconstruído
  nos vários canais, pagando referência, parâmetros e correções.
- **Ruído heteroscedástico:** a média pode ser imprevisível, mas a distribuição
  de magnitudes pode ser previsível; isso ajuda a codificar probabilidades.
- **Inovação verdadeiramente independente:** não pode ser prevista fielmente
  por um receptor sem informação da realização; seu custo cresce ao exigir
  erro menor.
- **Erro do modelo:** aliasing aprendido, escolhas de índices, redução de
  amplitude, bordas e falta de capacidade podem gerar resíduos aparentemente
  aleatórios. Não são prova de aleatoriedade da fonte.
- **Problema de aquisição/exportação:** resolução numérica, saturação e extremos
  podem exigir preservação exata e esclarecimento instrumental.

Gauge length maior que dx favorece dependência, mas não determina sozinho a
entropia. Diferenciação temporal pode reforçar altas frequências e produzir
correlações negativas; “lag1 pequeno” não significa “não há estrutura”.
Integrar strain-rate ou suavizar canais muda o alvo se não houver inversa,
condições de contorno e precisão contabilizadas.

### 4.2 Dados: mais exemplos podem ensinar estatística, não cada ruído futuro

É essencial distinguir:

1. **Estimar regularidades locais** com muitas células de poucos arquivos.
2. **Generalizar entre sessões/regimes** com experiências fisicamente diversas.
3. **Reconstruir uma realização nova** a partir de um número limitado de bits.

Mais dados podem melhorar 1 e 2 e a eficiência de 3. Não eliminam a informação
nova que precisa ser transmitida em 3. Uma fonte simples, mas ruidosa, pode
exigir muitos bits mesmo que sua distribuição seja conhecida exatamente.
Uma fonte complexa e muito estruturada pode ser compressível, porém difícil
de aprender com pouca cobertura.

Portanto, “pobreza dos dados” e “limitação informacional” podem coexistir,
mas não são sinônimos. **Novas aquisições têm alto valor para decidir; não são
uma garantia de ganho de compressão.**

### 4.3 Modelo: generalizar reconstrução não é prever o próximo waveform

O encoder observa o dado que deve comprimir. Não precisa adivinhar o ruído:
precisa escolher uma representação que o receptor consiga decodificar.
O decoder só tem os símbolos/side/modelo disponíveis.

Uma CNN com muitas features e um único índice K2048 por posição continua
dispondo de, no máximo, 11 bits nominais naquela posição. Aumentar features
antes do quantizador ou parâmetros compartilhados não transmite informação
nova da aquisição automaticamente.

Os testes [5](5-TRAIN_ONLY_CAPACITY_PROBE.md) e
[6](6-SHARED_CAPACITY_AND_TRANSFER_PROBE.md) mostraram ajuste individual e
overfit de um pool pequeno. Pesos ajustados ao próprio alvo podem memorizar
informação que não foi cobrada nos índices. Isso não estabelece a capacidade
de um codec compartilhado à mesma taxa.

A expansão de ~28M para ~96,5M teve retorno modesto; não demonstrou que a rede
seja universalmente pequena demais. A hipótese “grande demais para a diversidade”
é plausível, mas também não foi isolada: houve curvas ainda em melhora e
diferenças de otimização. Não existe uma regra útil de “células por parâmetro”
que resolva isso para estes dados correlacionados.

O rank efetivo ~4,34 do latente em 96 é uma propriedade da representação
pooled existente. **Não significa que o waveform possua apenas quatro graus
de liberdade**, nem que quatro números substituam seus índices.

### 4.4 Experimento: capacidade de distinguir causas é um recurso limitado

A repetição do mesmo desenvolvimento torna possível selecionar pequenas
vantagens por adaptação da pesquisa ao corpus. Hashes, CRCs e 498 testes
passados sustentam integridade de implementação; não tornam a evidência
independente nem demonstram validade do modelo físico.

Os checks em arquivos do treino não são um conjunto de generalização da fonte
madura. A validação completa já consumida continua desenvolvimento. A pequena
diferença entre ~0,490 nos checks e ~0,519 na validação não é uma estimativa
causal de gap de generalização: mudam grupos e população.

Ainda assim, **a fidelidade limitada já dentro do domínio conhecido enfraquece
a tese de que todo o erro seria resolvido apenas com dados externos**.
Representação, objetivo, taxa e otimização precisam participar do diagnóstico.

## 5. Limitações informacionais: o que podemos e o que não podemos concluir

### 5.1 A taxa solicitada é extremamente baixa em unidades da fonte

Para payload de origem float32 e cobertura fixa, `CR = 32/R`, com R em
bits/célula. Não usar tamanho HDF5 comprimido, canais removidos ou pesos omitidos
para aumentar artificialmente CR.

Como referência matemática, para uma fonte **gaussiana i.i.d. hipotética**
de variância σ² sob MSE, o ótimo assintótico satisfaz:

```text
D(R) / σ² = 2^(−2R)
RMSE(R) / σ = 2^(−R)
```

| R (bpc) | CR sobre float32 | RMSE/σ ideal nesse modelo hipotético |
|---|---:|---:|
| 0,045714 | 700× | 0,96881 |
| 0,2 | 160× | 0,87055 |
| 0,7 | 45,71× | 0,61557 |
| 1 | 32× | 0,50000 |
| 2 | 16× | 0,25000 |
| 4 | 8× | 0,06250 |

**Nenhuma linha é uma previsão ou um limite medido do FEBUS.** O R acima
também concede todos os bits à informação da fonte, sem overhead. A conta
mostra a escala do desafio quando a redundância é pequena.

O ponto OOI que inspirou razões ~700× usa outro domínio: o relatório 1
documenta `dphi` a **200 Hz / 5 s por crop**, contra strain-rate FEBUS a
**20 kHz / 50 ms**. Seu desempenho histórico não cria uma obrigação física de
atingir o mesmo ratio preservando a banda FEBUS e suas realizações ruidosas.

### 5.2 Uma pequena parcela de inovação já pode custar bastante

Considere **apenas como hipótese** `X = S + N`, com N gaussiano i.i.d.,
independente de S, e fração de variância `q = Var(N)/Var(X)`.
Conceda gratuitamente S ao encoder e ao decoder. Mesmo nesse problema
facilitado:

```text
RMSE / σ_X ≥ sqrt(q) × 2^(−R)
R ≥ max(0, 0,5 × log2(q/r²)), para alvo r = RMSE/σ_X
```

Para `r = 0,1`, a hipótese exigiria:

| q hipotético | R mínimo, com S gratuito | CR máximo sobre float32 |
|---|---:|---:|
| 10% | 1,661 bpc | 19,27× |
| 25% | 2,322 bpc | 13,78× |
| 100% | 3,322 bpc | 9,63× |

Inversamente, para atingir r=0,1 em 700×, q teria de ser no máximo **~1,065%**;
em 160×, **~1,320%**; em ~45,7×, **~2,639%**, ainda concedendo o restante
gratuitamente. O custo de descrever S e os metadados tornaria a exigência maior.

Essas contas são para RMSE normalizado por um desvio da fonte especificada,
**não para a macro NRMSE***. O exemplo r=0,1 não é uma tolerância já escolhida
pelo projeto. Tampouco é permitido estimar q elevando ao quadrado o NRMSE*
do VQ: seu residual contém erro de representação, quantização e otimização.

Esse cenário torna a incompatibilidade entre alta precisão e compressão extrema
uma possibilidade séria. **Ainda não sabemos se o FEBUS satisfaz as hipóteses
necessárias para aplicar esse limite.**

### 5.3 O sentido dos limites precisa estar correto

- **Um codec executado fornece um ponto atingível**, sujeito à população
  avaliada. Seu custo é uma referência superior para a taxa mínima necessária
  naquele nível de distorção, não um piso que codecs melhores não possam superar.
- **Entropia marginal de coeficientes** depende da representação/quantização e
  ignora dependências. Não é entropia da fonte nem mínimo de todo codec com perdas.
- **Cross-entropy de um modelo causal** inclui erro de modelagem; falhar em
  reduzi-la não prova independência. Economia efetiva, por outro lado,
  demonstra uma oportunidade concreta.
- **Covariância/PSD e reverse water-filling** dão a curva da fonte gaussiana
  correspondente. Sob covariância e MSE fixadas, a gaussiana é um caso extremal
  difícil; tomar essa curva como limite inferior universal do dado real
  inverte a interpretação.
- **Covariância de dimensão alta com poucas amostras** pode parecer de baixo
  rank por insuficiência amostral. Rank truncado não prova compressibilidade.
- **Dicionário dos exemplos observados:** num conjunto finito, armazenar todas
  as janelas no decoder e transmitir IDs pode parecer excelente. O custo da
  coleção e o desempenho em novas realizações impedem tratá-lo como lei da fonte.

Existem métodos de estimar limites RD por amostras, como Yang–Mandt e
Yang et al. [L6–L7](#literatura-teoria). Eles dependem de hipóteses estatísticas,
capacidade de estimadores, otimização e dimensão. Aplicá-los diretamente a
512.000 células correlacionadas com poucos arquivos não produziria um
“certificado de impossibilidade”. Sua utilidade inicial seria em blocos pequenos,
com controles de distribuição conhecida e análise de estabilidade.

### 5.4 Ruído plausível não preserva a realização observada

Se o decoder gerar outra realização independente com a mesma PSD/variância,
ela pode parecer correta e falhar em waveform, fase e correlação amostra a
amostra. Para duas realizações independentes com a mesma média e variância σ²,
o erro quadrático esperado entre elas é **2σ²**.

Sob MSE, a média condicional é a estimativa ótima para a informação recebida;
reduzir amplitude de componentes incertas pode ser consequência do gargalo,
não apenas um erro de loss. Recuperar simultaneamente amplitude e realização
exige informação adicional ou uma representação mais eficiente.

Modelos generativos são úteis como **modelos de probabilidade para codificação**.
Usá-los para inventar detalhes não transmitidos não atende ao objetivo atual.
Conhecer o nível de ruído do sensor também não autoriza apagá-lo: tolerância
de medição e fidelidade ao arquivo armazenado são contratos distintos.

## 6. Curadoria: o recurso escasso é cobertura física identificável

### 6.1 O que o inventário permite afirmar

A recontagem do manifesto atual confirmou:

| Split | Arquivos | Configurações | Janelas | Células retidas |
|---|---:|---:|---:|---:|
| Treino | 87 | 10 | 11.760 | 915.440.000 |
| Validação consumida | 51 | 6 | 7.220 | 635.820.000 |
| Teste reservado nesta sequência | 69 | 7 | 9.800 | 1.158.820.000 |
| **Total** | **207** | **23** | **28.780** | **2.710.080.000** |

Metadados indicam 20 kHz e datasets rotulados strain-rate, incluindo Ch1/Ch2.
São contagens de valores armazenados, não observações estatisticamente
independentes.

O treino utilizado na sequência recente é especialmente limitado:

| Família | Configurações de treino | Arquivos de treino | Arquivos fora dos checks históricos |
|---|---:|---:|---:|
| Doped | **1** | 9 | **5** |
| Multicore | 2 | 36 | 18 |
| Multimode | **1** | 6 | **3** |
| Single-mode | 6 | 36 | 18 |
| **Total** | **10** | **87** | **44** |

Os 44 arquivos correspondem às 5.600 janelas disponíveis fora dos arquivos
de check; restrições adicionais, como as duas configurações excluídas em H3,
reduzem alguns ajustes. Fonte madura e novos braços tiveram históricos distintos
de acesso a esses dados.

No multimode de treino, a mediana do tamanho dos tiles no manifesto é de
**dois canais**, embora existam tiles de até 247. Essa é a distribuição dos
tiles preparados, não a largura física de toda a fibra nem uma fração de
energia. Reforça a necessidade de publicar resultados por geometria e de
distinguir quantidade de tiles de quantidade de conteúdo.

O relatório [7](7-CORPUS_COVERAGE_AND_NUMERIC_INTEGRITY_AUDIT.md) estimou
~896,1 s na soma das uniões de clock por grupo e cobertura temporal histórica
~95%. É evidência de pouca duração/diversidade, não contagem exata de sessões.
As auditorias de 98 encontraram dez combinações de Hostname/AcqStartTime,
81 clocks distintos e nenhum fingerprint amostrado não constante coincidente:
**nem dez sessões comprovadas, nem 81 aquisições independentes**.

### 6.2 Curadoria que efetivamente mudaria a capacidade de concluir

Prioridades concretas:

1. **Reconstruir o grafo de aquisição:** sessão, montagem, dia, instrumento,
   fibra/núcleo, excitação e exportações derivadas. Usar registros do laboratório
   e responsáveis pela aquisição; timestamps isolados não resolvem isso.
2. **Esclarecer o produto armazenado:** unidade/calibração, quantização nativa,
   derivação temporal, filtros/decimação internos, overlap de blocos, pulso e
   gauge length. Manter campos sem semântica conhecida explicitamente incertos.
3. **Relacionar canais a posições e condições físicas:** região útil, bobina,
   ponta/reflexão, fading, não finitos e escapes. A política
   `persistent-amplitude-v1` continua uma heurística; curadoria não é excluir
   os trechos difíceis até o codec melhorar.
4. **Catalogar regimes por aquisição**, com descritores de banda, amplitude,
   coerência e estacionariedade. Regimes descritivos não se tornam classes
   físicas confirmadas nem rótulos gratuitos para o decoder.
5. **Publicar uma matriz de cobertura**, destacando combinações nunca observadas.
   Testar nova configuração, novo dia da mesma configuração e nova família
   são desafios distintos.

Dados existentes e máscaras históricas permanecem preservados. Uma política de
cobertura cientificamente revisada teria versão própria, denominador comum e
custo explícito; seu benefício não seria atribuído à rede.

### 6.3 Aquisição proposta: cruzar fatores e medir repetibilidade

Mais uma coleção de tons com hardware diferente em cada pasta preservaria a
confusão atual. A nova coleta deve repetir **a mesma condição em sessões distintas**
e variar excitação mantendo hardware fixo; parte da campanha deve variar um
parâmetro de hardware com excitação controlada.

Exemplo inicial de planejamento:

- quatro famílias, **quatro sessões documentadas por família**;
- em cada sessão, quatro regimes de 30 s: fundo medido, tons/chirps controlados,
  excitação larga e transientes/misturas representativos do uso;
- repetir um protocolo comum entre sessões e registrar remontagem/acoplamento,
  potência, firmware, temperatura e parâmetros de aquisição;
- referência de excitação sincronizada; quando disponível, bobina/canal de
  referência e medição co-localizada para investigar componente comum;
- reservar, antes do uso, duas sessões por família para ajuste, uma para
  desenvolvimento e uma para auditoria. Expandir conforme variabilidade observada.

Isso soma **32 min / 16 sessões**, ou ~78,64 GB em 512 canais × 20 kHz × float32,
antes de containers e streams auxiliares. É um exemplo de custo de piloto,
**não cálculo de tamanho amostral suficiente**. Uma sessão de auditoria por
família ainda oferece pouca precisão estatística.

Repetições ajudam a separar resposta estável e variabilidade. Usar metade
da variância da diferença entre repetições como ruído requer mesmo sinal,
alinhamento, ruídos independentes e variâncias comparáveis; drift, jitter ou
mudança de acoplamento quebram essas hipóteses. A análise deve apresentar
incerteza e hipóteses, preservando cada waveform original.

O template `config/new_febus_acquisitions.template.csv` e seu validador podem
ser reutilizados. O preenchimento não certifica independência física.

## 7. Estado da arte: confrontar o alvo antes de comparar números

**Não identifiquei, nas fontes consultadas, uma demonstração diretamente
comparável de alta fidelidade do waveform FEBUS 20 kHz completo, incluindo
sua realização de ruído, nas quatro famílias e a ~160–700×.**
Isso é o resultado desta busca dirigida, não afirmação de inexistência.

### 7.1 STeP-C: resultado recente e relevante, com outro contrato de sinal

[Li et al., preprint de novembro de 2026](https://arxiv.org/html/2611.17126v1)
apresentam encoder espacial 1D, predição temporal LSTM, representação do residual
latente e codificação entrópica, com treinamento conjunto. Reportam **87,30×,
0,367 bps, 31,04 dB de PSNR e 535 MB/s** numa implantação de 81.000 canais
a 1 kHz.

O texto integral, porém, especifica:

- DADAS/PFT filtrados em **1–200 Hz** e normalizados por **tanh** com escala
  de percentil 99;
- NJCFA com unwrapping e passa-altas de 5 Hz;
- PSNR definido no sinal normalizado;
- generalização espacial para canais não usados no treino do mesmo sistema;
- consistência de detector, que não é igualdade do waveform.

“Full-band” nessa avaliação de implantação cobre a extensão de canais; não
permite concluir que preservou toda a banda temporal do readout sem filtragem.
Tanh é invertível idealmente, mas muda a ponderação da distorção e pode amplificar
erro após a inversa nas caudas; PSNR normalizado não é garantia em unidades físicas.

**O que aproveitar:** organização do pipeline, custo do coder, contexto
compartilhado e comparação com/sem preditor. O preditor é descrito como open-loop
a partir do primeiro latente reconstruído; não é justificativa para fornecer
histórico original ao decoder. A desigualdade de entropia condicional, sozinha,
também não garante que um preditor/coder finito reduza bytes.

**Decisão para FEBUS:** pista condicional, não motivo para lançar uma LSTM ou
substituir o alvo por dados filtrados. O ganho temporal precisa existir no
readout FEBUS integral, além dos controles já negativos.

### 7.2 ORCAS: diversidade externa útil, fidelidade de dados pré-processados

[Pavelkin et al., preprint 2026](https://arxiv.org/html/2609.22140v1) usam
transformada ortogonal aprendida, esparsificação, Lloyd–Max e codificação
estrutural/entrópica. Reportam, por exemplo, **13× e SNR 13,32 dB**, e
~6,5×/27 dB em configuração de maior precisão.

Há uma separação interessante por fontes: treino em várias instalações e teste
em Svalbard/Alaska. Porém, o pipeline remove média, aplica filtragem de banda,
reamostra para **100 Hz**, normaliza e extrai patches. O próprio artigo limita
suas conclusões a dados pré-processados. A descrição de filtro até 100 Hz e
saída a 100 Hz ainda exige esclarecer a implementação de antialiasing/Nyquist.

Não transportar a curva publicada para FEBUS 20 kHz. O ganho intelectual é a
curadoria entre instalações e o acoplamento entre transformada/quantização/coder,
não a expectativa de que outra base esparsa resgate automaticamente H1.

### 7.3 DASPack: referência de precisão e de escala dos ratios

[Seguí et al., GJI 2026; versão pública de 2025](https://arxiv.org/html/2507.16390v2)
avaliam 15 datasets e separam quantização de precisão fixa de transformadas e
codificação lossless. Reportam até ~3× lossless, ~6× com alta fidelidade nos
exemplos e ~10× com degradação maior; os fatores variam com dataset/tolerância.

O benchmark mistura strain/strain-rate, origens int16/int32/float32 e 50–10.000 Hz.
O artigo esclarece que converter floats a inteiros por arredondamento é uma
etapa com perdas. Logo, “modo lossless” não deve ser extrapolado para qualquer
float32 sem conferir o contrato.

Esses resultados tornam plausíveis ratios bem menores quando precisão é a
prioridade; não estabelecem um teto FEBUS nem tornam nossos pontos superiores
por terem CR maior. Um único controle desse tipo poderia verificar uma curva
de viabilidade, se útil, sem transformar benchmarks clássicos em pré-requisito
de toda pesquisa neural. O objetivo de produto continua neural.

### 7.4 Outros resultados que precisam de leitura cuidadosa

- **ZipDAS:** o [código consultado](https://raw.githubusercontent.com/AI4EPS/ZipDAS/main/zipdas/data.py)
  aplica clipping de percentil no loader, antes e depois de normalizar; o
  default é 0,99. Um ramo de leitura também diferencia o sinal. É um projeto
  útil para estudar engenharia de compressão, mas seus defaults não preservam
  o alvo integral deste projeto.
- **DLPR / NGLR:** a ideia de precisão explícita e residual condicionado continua
  relevante. [DLPR](https://arxiv.org/abs/2209.04847) trata imagens;
  [NGLR](https://arxiv.org/html/2606.05389v2) trata campos de simulação e
  precisão por bloco. Estrutura de campos simulados e seus denominadores não
  estabelecem previsibilidade do ruído FEBUS. H4 já testou uma instância
  limitada desse mecanismo; citar esses artigos de novo não constitui novidade.
- **OmniDAS:** o [preprint de outubro de 2026](https://arxiv.org/html/2610.39892v1)
  distingue compressão orientada à tarefa de reconstrução do waveform.
  A razão de compressão muito alta da variante de classificação não mede
  preservação do sinal.
  A disponibilidade efetiva do dataset é discutida abaixo.

**Síntese do estado da arte:** avanços existem, mas os contratos diferem
profundamente. Não há uma tabela de ratios/PSNR que permita concluir que falta
apenas copiar uma arquitetura vencedora para FEBUS.

## 8. Fontes de dados: valor experimental antes de volume

| Fonte | Evidência de acesso/compatibilidade consultada | Uso recomendado |
|---|---|---|
| Novas sessões FEBUS do laboratório | Ainda não disponíveis/documentadas. Mesmo produto e famílias podem ser controlados. | **Prioridade máxima** para repetibilidade, cobertura e confirmação. |
| EDAS/SMF do acervo conhecido | Metadados históricos: strain-rate, 20 kHz, dx 0,2 m, GL 1 m. SMF consumido; Polyimide/Acrylate reservados. | Preservar papéis; não usar as reservas para escolher uma nova arquitetura nem alegar cobertura das quatro famílias. |
| BFO/FEBUS A1-R | [Depósito público](https://radar.kit.edu/radar/en/dataset/hww56j6d6shqd89p), 28,6 GB, CC BY 4.0; declara dados derivados/decimados. Artigo usa FEBUS a 200 Hz, pulso 10 kHz e GL 50 m. | **Alta utilidade para desenho de calibração/ruído**, baixa compatibilidade direta com o treino 20 kHz. Conferir README e datas antes de incorporar: título e descrição do catálogo não são inteiramente consistentes. |
| DAS-BIGORRE-2022 | [DOI e metadados DataCite](https://api.datacite.org/dois/10.57932/5b1302d6-57cd-44e4-81ac-5d585a7f8951) confirmam 19 registros de 10 min de strain-rate, com 13 terremotos/seis explosões e mapa de fibra. | Pista FEBUS para regimes distintos; frequência armazenada, processamento, licença e acesso aos arquivos ainda precisam de auditoria. Nenhum waveform foi baixado. |
| Ponte A8, Mercerat et al. | [Artigo](https://arxiv.org/html/2510.24212v1) já levantado em 90: FEBUS a 8 kHz, análise decimada e dados mediante solicitação. | Solicitar bruto/metadados se houver colaboração; não contar como corpus disponível. |
| PubDAS/Global DAS Month | Fontes usadas por ORCAS cobrem instalações e interrogadores diferentes; TU Graz inclui FEBUS A1-R a 100 Hz no produto usado. | Candidatos a generalização/calibração após triagem física, sem presumir equivalência de banda. |
| OmniDAS | [Página pública](https://gaoyuan-mr.github.io/OmniDAS/) descreve seis cenários; alguns arrays/durações implicam 30–50 kHz. O artigo diz “disponível após publicação”, DOI reservado. A API Zenodo do registro 21992120 retornou **404** nesta consulta. | Monitorar/solicitar informação; não planejar treino sobre um download ainda não confirmado. Readout, calibração e processamento são mais importantes que sample rate. |
| QuantX/PoroTomo locais | Pontes já executadas, com transferência negativa ou benefício insuficiente frente a investir no alvo. | Reutilizar como evidência/controles. Nova mistura exigiria compatibilidade identificada, não outro pré-treino indiferenciado. |

Não encontrei uma fonte pronta que elimine simultaneamente as lacunas de
readout, banda, famílias e sessões. A documentação pública do BFO oferece
um exemplo especialmente útil: referência co-localizada, calibração e
componente comum, mas **pulso a 10 kHz não é saída a 10 kHz**.

Forbriger et al. observam ruído comum em parte dos instrumentos e aumento de
coerência com sobreposição de gauge length. Também encontram offsets de tempo
mesmo com sincronização GNSS. Isso orienta a coleta/calibração; não autoriza
subtrair ruído sem transmiti-lo nem procurar lags nos checks históricos H5.

Para qualquer fonte candidata, a triagem deve registrar produto, banda realmente
armazenada, unidade, geometria, sessões, tratamento numérico e licença. Uma
coleção de FFTs/energias, imagens de waterfall ou sinais já limpos não substitui
as realizações brutas pretendidas.

## 9. Próximas perguntas, com decisões falsificáveis

Estas são **recomendações para protocolos futuros**, não jobs autorizados ou
protocolos completos já congelados. A rodada H1–H5 permanece encerrada.
Uma eventual execução usa **no máximo uma GPU remota, com braços sequenciais**.
F0/F1/F2 podem avançar no desenvolvimento existente mesmo sem novas sessões.
Nesse cenário, F3 permanece indisponível e qualquer conclusão de generalização
deve continuar limitada; uma reserva histórica não substitui a coleta que falta.

### F0 — Fixar a pergunta de viabilidade e o contrato físico

**Hipótese a esclarecer:** parte do impasse decorre de perseguir um ratio sem
uma precisão mínima operacional explicitamente definida.

Entregável:

- alvo/cobertura: waveform armazenado, ruído incluído, produto/unidade e canais;
- horizonte de transporte, acesso aleatório, memória e latência toleráveis;
- erro absoluto/NRMSE*, piores canais, viés, amplitude e erro espectral por banda;
- fase com suporte definido pelo alvo, não pela amplitude reconstruída;
- limites físicos escolhidos com o responsável pelo uso e congelados antes
  de resultados novos.

Separar duas perguntas:

1. **Eficiência:** quanto erro conseguimos reduzir a bytes completos não maiores?
2. **Viabilidade:** quantos bits são necessários para atingir a precisão requerida?

Um ponto de maior taxa pode responder à segunda sem ser ganho na primeira.
Se ainda não houver tolerância operacional, usar uma grade declaradamente
exploratória e mostrar todas as curvas; não chamar seu melhor ponto de solução.

### F1 — Localizar o déficit: representação, contexto, generalização ou taxa

**Hipótese:** controles distantes da referência prática podem refletir
aprendizagem insuficiente, mudança de contexto de inferência ou falta de
informação no stream; essas explicações ainda precisam ser discriminadas.

Começar pelos artefatos existentes, sem repetir a campanha:

1. Reunir pontos compatíveis por população/contrato, com dose, fonte e seleção
   explícitas. Não calcular uma média global dos relatórios.
2. Separar resultado em dados usados no ajuste, novos trechos da mesma origem,
   novas configurações e novas sessões. Quando não existe a última categoria,
   marcar a conclusão como indisponível.
3. Conferir perda hard/bytes, curvas e desempenho dos controles. O ganho ou
   fracasso de um candidato contra um controle muito fraco não decide o ótimo.
4. Se houver execução diagnóstica autorizada, usar um painel de treino fixado
   por metadados e medir a mesma região central com contextos de diferentes
   tamanhos, sem atualizar pesos. Medir os custos reais de cada particionamento
   separadamente; uma melhora de contexto não é automaticamente ganho de codec.

Uma sensibilidade material ao contexto sustentaria **um único contraste de
compatibilidade treino–inferência**, com cobertura preservada e exposição medida.
Isso deve preceder outra curva de “mais dados”. Não é uma retomada automática
de normalização/padding/FiLM; a pergunta é a validade do regime de aprendizagem.

Controle contínuo, quando necessário, deve ser realmente treinado e ter função
diagnóstica. Repetir o bypass congelado ou memorizar janelas com pesos gratuitos
não identifica o piso de compressão.

**Decisão:** se treino e generalização falham mesmo em regimes de informação
mais abundante, investigar entrada/otimização/representação. Se o ajuste
compartilhado funciona e novas sessões falham, priorizar dados e robustez.
Se fidelidade cresce sobretudo com R em ambos, a taxa ganha peso na explicação.
Nenhum desses padrões sozinho prova uma causa exclusiva.

### F2 — Medir redundância utilizável e a transição de precisão

**Hipótese:** o VQ já remove grande parte da dependência barata e a inovação
restante pode limitar o ganho; alternativamente, há dependência de distribuição
que os preditores de média e buckets não conseguiram codificar.

Em novo protocolo train-only, com amostra fixada antes das leituras:

- usar trechos realmente contínuos dos HDF5, mantendo overlaps/exportações
  rastreáveis; não concatenar NPZ espaçados;
- medir estrutura espaço–tempo por regime e banda, incluindo dependência
  não restrita a lag1 e estabilidade fora do arquivo de ajuste;
- separar contexto original otimista de contexto **decodificado e pago**;
- medir custos de símbolos quantizados em uma grade de precisão justificada
  por F0, com todas as tabelas, escalas, resets, escapes e modelos;
- publicar pontos atingíveis na faixa intermediária e as referências VQ
  existentes sob o mesmo contrato; orientar o estudo por R e erro físico,
  não pelos alphas históricos de H4;
- incluir controles sintéticos pequenos de redundância conhecida, como
  gaussiano i.i.d. e gaussiano correlacionado, para verificar se o diagnóstico
  reconhece casos fáceis/difíceis. São calibração de método, não novos dados FEBUS.

A parte probabilística deve comparar marginal e contexto causal **na mesma
quantização**, inicialmente em blocos pequenos. Medir economia em bits/célula,
não somente R², PSD ou loss de magnitude. Se o ganho potencial já for menor
que a instalação/side/latência, o ramo termina antes de uma rede grande.

Uma estimativa neural de limites RD é opcional e secundária: só é informativa
se passar controles conhecidos, mostrar estabilidade por tamanho de bloco e
preservar a incerteza entre arquivos/sessões. Não financiar uma rede gigante
para tentar provar que outra rede gigante é ótima.

**Decisão:** oportunidade líquida consistente autoriza F4; redundância pequena
em vários controles independentes reduz a prioridade desse mecanismo.
Resultado inconclusivo não vira um limite informacional.

### F3 — Curva de dados por sessões, com controle de aprendizagem

**Hipótese:** diversidade física real melhora transferência e eficiência.
A curva antiga variou simultaneamente o número de regimes e a repetição por
configuração sob dose fixa, sem isolar o efeito de novas sessões.

Executar somente após curadoria/aquisição que permita uma pergunta identificável:

- mesma arquitetura funcional, geometria de treino/inferência compatível e
  conjuntos de avaliação fixos;
- subconjuntos aninhados de **sessões inteiras**, preservando relações entre
  núcleos, eventos e exportações;
- reportar duas leituras de orçamento: compute total fixo e exposição por
  sessão/convergência controlada. São estimandos diferentes;
- deixar a dose de adaptação FEBUS explícita quando houver fonte externa;
- inspecionar estabilidade das curvas e erro por sessão/configuração, sem
  selecionar a melhor composição depois dos resultados.

Uma fase inicial pode usar duas seeds para medir sensibilidade do treino; elas
não substituem sessões independentes. Não repetir simplesmente 4/6/10 com
outra seed para salvar uma tendência.

**Decisão:** ganho consistente em sessões novas aumenta a prioridade de coleta
e modelos compartilhados. Ausência de ganho com controle bem aprendido reduz
a expectativa de que apenas acrescentar volume resolva o patamar.

### F4 — Uma candidata neural condicionada ao diagnóstico

#### F4-A: distribuição neural dos valores quantizados, sem base VQ obrigatória

**Minha candidata preferida se F2 mostrar oportunidade de codificação.**

```text
waveform original → quantização física declarada + escapes
                 → modelo neural p(q_i | contexto já decodificado, metadados)
                 → coder entrópico → q exato → waveform de precisão declarada
```

A rede estima probabilidades/escala/mistura; o coder transmite a realização
efetivamente observada. O contexto deve ser estritamente reproduzível: valores
de outros grupos já decodificados, bits mais significativos já conhecidos ou
histórico reconstruído. Metadados não conhecidos entram no stream.

**Diferenças verificáveis frente ao histórico:**

- H4 pagava uma base VQ e treinava um MLP para **prever a média** de coeficientes
  por MSE, seguido de coder genérico;
- H5 previa magnitude para **particionar em buckets**, introduzindo tabelas e
  alterando localidade;
- o diagnóstico 86 usava escalas em coeficientes de outra representação;
- aqui o objeto principal é a **distribuição discreta dos valores a preservar**,
  com objetivo de log-probabilidade e precisão independente da síntese neural.

Isso é uma diferença de mecanismo, não uma alegação de que modelos de entropia
jamais foram testados. Se F2 não sustentar dependência adicional, não vale a
pena implementá-lo.

Primeiro contraste recomendado:

- controle marginal e candidato contextual, mesma quantização, cobertura,
  capacidade nominal e exposição;
- rede pequena e contexto local justificado pelos dados; contabilidade de
  modelo global e qualquer adaptação por aquisição;
- duas condições de precisão predefinidas por F0/F2, com bytes realmente
  medidos; não varrer tolerâncias até aparecer vitória;
- treino por log-probabilidade, com auditoria entre bits previstos e coder
  de probabilidades discretizadas;
- grupos paralelizáveis/bitplanes como desenho inicial, evitando inferência
  Python serial por cada célula; resets e acesso aleatório cobrados;
- fallback explícito para o coder de controle, sem apagar o custo do modelo.

Uma proposta de orçamento, a ajustar em preflight de memória/células, seria
**8.192 updates mínimos por braço**, com teto **32.768** somente se curvas
congeladas, saúde do coder e economia fora do ajuste justificarem a extensão.
Registrar também células e compute, pois updates não equivalem a FLOPs.
Ajustes de memória devem preservar o contrato físico treinado e avaliado.
Uma GPU, braços sequenciais.

**Vantagem metodológica:** erro físico fica determinado pela quantização/escapes;
um prior ruim fora do domínio tende a aumentar bytes, em vez de inventar a
realização. **Limite:** pode continuar exigindo vários bits/célula. Uma garantia
de erro não garante simultaneamente um teto ultracompacto para qualquer entrada.
Se o teto não comportar a precisão, o sistema deve declarar essa incompatibilidade.

#### F4-B: representação menos dependente do contexto, somente se F1 a justificar

Se F1 mostrar déficit de contexto/representação maior que a oportunidade de
entropia, a prioridade muda para **um modelo compartilhado pequeno, treinado no
mesmo contrato geométrico usado para decodificar**, com operações locais e
tratamento de bordas explícito.

Uma transformada de análise/síntese com suporte físico controlado ou hierarquia
de informação pode então ser defensável. Todos os streams, inclusive detalhes,
devem competir pelo orçamento comum. O controle deve isolar a propriedade
identificada, em vez de trocar backbone, quantizador, loss, dados e coder juntos.

Não priorizo hoje outro Transformer, Mamba, MoE, RVQ ou lifting maior apenas
pelo nome. Também não proponho executar F4-A e F4-B em sequência obrigatória:
**selecionar no máximo uma candidata estrutural com base nos diagnósticos**.

## 10. Critérios de investimento, confirmação e encerramento

| Evidência futura | Direção racional |
|---|---|
| Grande sensibilidade de contexto, com entrada íntegra | Corrigir/isolar o contrato de representação antes de atribuir fracasso à falta de dados. |
| Bom ajuste compartilhado, queda em novas sessões | Curadoria, diversidade e robustez de domínio ganham prioridade. |
| Dependência condicional reduz bytes a reconstrução idêntica, após todos os custos | Justifica o candidato probabilístico e confirmação. |
| Ganho aparece apenas ao aumentar R | Avanço na curva de viabilidade; não é ganho de eficiência à mesma taxa. |
| Alta precisão só aparece em múltiplos bpc em mecanismos distintos e sessões diversas | Reduzir a plausibilidade prática da meta ultracompacta; negociar requisito/ratio explicitamente. Ainda não é teorema sobre todo codec. |
| Dados/proveniência continuam insuficientes e candidatos não mudam a fronteira | Encerrar a campanha de otimização, manter a referência e documentar a região ainda não identificada. |

Para investir em confirmação de um candidato:

1. Comparação com controle causal **e** referência prática madura, na mesma
   população/cobertura e em pontos de taxa efetivamente comparáveis.
2. Ganho material predefinido, por exemplo ≥5% de erro sem mais bytes ou ≥10%
   de bytes a fidelidade equivalente. Distinguir ganho de payload e total.
3. Guardas físicas e casos fracos/constantes/escapes auditados; nenhum ganho
   por seleção favorável de canais, bandas ou suporte de fase.
4. Confirmação com nova seed e sessões apropriadas, quando disponíveis,
   sem usar o teste histórico/EDAS para escolher o algoritmo.
5. Throughput de encoder **e** decoder, memória e latência. Em 512 canais
   a 20 kHz, o input float32 é **40,96 MB/s**; precisão/custo offline não
   demonstram operação nessa velocidade.

Custos:

```text
B_total(N) = B_modelo_instalado + B_setup + B_símbolos(N)
             + B_side(N) + B_framing(N) + B_resets(N)
```

Publicar transporte com pesos já instalados e objeto autocontido. Pesos
adaptados à aquisição são informação dela; cobrar ou garantir adaptação
sincronizada usando exclusivamente dados previamente decodificados.
Um modelo global grande pode amortizar em streams longos, mas não em todos os
horizontes. Não comparar exports completos de um braço com receptores mínimos
de outro.

Intervalos de incerteza devem respeitar sessões e fatores experimentais.
Bootstrap de milhares de janelas dos mesmos arquivos não resolve a ausência
de sessões. Tolerâncias de equivalência precisam vir do uso, não do resultado.

**Teto de pesquisa recomendado:** primeiro a investigação de viabilidade e dados;
depois, no máximo uma candidata neural justificada, com controle e extensão
limitada. Se ela falhar, o próximo passo é decisão sobre o objetivo ou nova
evidência física, não uma lista automática de arquiteturas.

## 11. Avaliação final de factibilidade

### O que considero plausível

- Ganhos incrementais de eficiência em regimes específicos.
- Melhor robustez com contexto compatível e sessões representativas.
- Um codec neural de precisão explícita que melhore o custo de armazenamento
  sem garantir as taxas ultracompactas.
- Ganhos maiores em configurações com redundância estável ainda mal explorada,
  caso isso seja demonstrado em novas análises/experimentos.

### O que considero pouco sustentado hoje

- Recuperar fielmente a realização inteira a ~700× em todas as famílias.
- Obter grande salto apenas aumentando rede, updates ou volume externo genérico.
- Tratar estatísticas de ruído aprendidas como substitutas dos bits da realização.
- Usar bons resultados de detecção, espectro ou dados filtrados como prova de
  preservação do waveform FEBUS integral.

### O que permanece indeterminado

- Qual fração da fonte é inovação independente relevante na precisão desejada.
- Quanto espaço existe entre a fronteira atual e a melhor representação
  alcançável, sobretudo nas taxas intermediárias.
- Quanto do déficit em novas condições seria removido por diversidade real.
- Qual precisão realmente precisa ser entregue para o uso científico pretendido.

**Minha conclusão não é “o objetivo é impossível”, nem “basta mais dados”.**
A evidência atual torna prudente tratar um salto generalista no regime
ultracompacto como uma hipótese de alto risco científico. O investimento de
maior valor agora é descobrir em que condições ela é plausível e quando deve
ser abandonada.

Se a fidelidade exigida só puder ser atingida a uma razão muito menor, isso
será uma conclusão útil sobre o problema. Não deve ser disfarçado de ganho de
eficiência nem de falha de engenharia. Se nenhuma precisão aceitável couber
no orçamento, será necessária uma decisão explícita sobre taxa, cobertura ou
objetivo; denoising e compressão orientada à tarefa seriam outros contratos.

O VQ maduro permanece a referência prática. Os negativos, protocolos e gates
1–105 permanecem preservados. Esta revisão entrega uma direção de pesquisa;
**não iniciou treino, sweep, P3 ou consumo de reservas**.

## 12. Referências externas e alcance da consulta

Consulta iniciada em **16/11/2026**, com fechamento documental em **17/11/2026**.
Os números publicados abaixo são dos autores e não
foram reproduzidos neste repositório.

1. **Li et al. — STeP-C: Real-Time Neural Compression for Distributed Fiber-Optic
   Sensing at Scale.** Preprint, v1, 12/11/2026.
   [Texto integral](https://arxiv.org/html/2611.17126v1).
   Seções 3.1–4.7 consultadas: arquitetura, contexto do receptor, filtros/tanh,
   PSNR, dados/splits, implantação e detector.
2. **Pavelkin et al. — ORCAS: Orthogonal Representation for Compression of
   distributed Acoustic Sensing.** Preprint, 2026.
   [Texto integral v1](https://arxiv.org/html/2609.22140v1);
   [implementação indicada](https://github.com/RomanPavelkin/orcas).
   Seções 3–4: fontes, pré-processamento, representação e taxas. Código não executado.
3. **Seguí et al. — DASPack: controlled data compression for distributed
   acoustic sensing.** GJI 244(1), 2026; publicação online 2025.
   [DOI](https://doi.org/10.1093/gji/ggaf397);
   [texto integral acessado](https://arxiv.org/html/2507.16390v2).
   O portal editorial retornou 403 nesta sessão; foi usada a versão pública v2.
4. **Forbriger et al. — Cemented fibers as a testbed for DAS.** Preprint 2026.
   [Texto](https://arxiv.org/html/2603.24353v1);
   [dados/software](https://doi.org/10.35097/gj4yg6478gtesxdw);
   [depósito FEBUS consultado](https://radar.kit.edu/radar/en/dataset/hww56j6d6shqd89p).
   Referência de calibração/ruído comum e sincronismo, não benchmark de compressão.
5. **Yang, Mandt e Theis — An Introduction to Neural Data Compression.**
   Foundations and Trends in Computer Graphics and Vision, 2023.
   [Referência e resumo consultados](https://arxiv.org/abs/2202.06533).
   Base conceitual de modelos de probabilidade, transmissão e taxa–distorção.

<a id="literatura-teoria"></a>

6. **Yang e Mandt — Towards Empirical Sandwich Bounds on the Rate-Distortion
   Function.** ICLR 2022.
   [Texto v2](https://arxiv.org/html/2111.12166v2);
   [código](https://github.com/mandt-lab/empirical-RD-sandwich).
   Consultados pressupostos i.i.d., limites superior/inferior e dificuldades
   de otimização/dimensão. Nenhuma estimativa foi executada em FEBUS.
7. **Yang, Eckstein, Nutz e Mandt — Estimating the Rate-Distortion Function
   by Wasserstein Gradient Descent.** NeurIPS 2023.
   [Artigo](https://arxiv.org/abs/2310.18908);
   [texto nos proceedings](https://papers.neurips.cc/paper_files/paper/2023/file/07eea3fb833c905c5edf46f914231f15-Paper-Conference.pdf).
   Atenção ao custo amostral e ao limite artificial log(n) de suportes finitos;
   não foi tratado como estimador certificado para nossa fonte.
8. **AI4EPS — ZipDAS.**
   [Projeto](https://github.com/AI4EPS/ZipDAS);
   [loader consultado](https://raw.githubusercontent.com/AI4EPS/ZipDAS/main/zipdas/data.py).
   Branch `main` na data da consulta; não foi instalada/executada. A constatação
   de clipping refere-se a esse arquivo, não a toda possível versão/configuração.
9. **Bai et al. — Deep Lossy Plus Residual Coding, TPAMI 2024** e
   **Zhu, Ranka e Rangarajan — Residual Modeling for High-Fidelity Learned
   Compression of Scientific Data, preprint 2026.**
   [DLPR](https://arxiv.org/abs/2209.04847);
   [NGLR v2](https://arxiv.org/html/2606.05389v2).
   DLPR retomado da revisão 90; resumo/mecanismo e limites do NGLR reconferidos.
10. **Gao et al. — Edge intelligence for distributed acoustic sensing with
    ultracompact neural networks / OmniDAS.** Preprint, outubro de 2026.
    [Texto](https://arxiv.org/html/2610.39892v1);
    [benchmark](https://gaoyuan-mr.github.io/OmniDAS/);
    [endpoint consultado, 404](https://zenodo.org/api/records/21992120).
    Anúncio/documentação não foram confundidos com acesso aos dados.
11. **Huynh et al. — DAS-BIGORRE-2022.**
    [DOI](https://doi.org/10.57932/5b1302d6-57cd-44e4-81ac-5d585a7f8951);
    [metadados confirmados](https://api.datacite.org/dois/10.57932/5b1302d6-57cd-44e4-81ac-5d585a7f8951);
    [artigo associado](https://doi.org/10.1093/gji/ggae382).
    Apenas registro e descrição; frequência/arquivos/licença ainda não auditados.
12. **Mercerat et al. — Tracking the normal modes of an overpass highway bridge
    using DAS.** 2025.
    [Texto](https://arxiv.org/html/2510.24212v1).
    Pista já registrada em 90; disponibilidade do bruto não confirmada.
