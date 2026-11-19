# Oportunidade multicore nominalmente alinhada

**Etapa consolidada dos registros 113, 114.** Probe linear não passou o gate; contexto VQ decodificado não foi lançado.

Protocolos, resultados e auditorias foram reunidos sem alterar resultados, budgets ou gates. As partes históricas seguem em ordem: estados de lançamento/execução descrevem aquele momento, não tarefas ainda pendentes. Consulte as últimas partes para o encerramento.

## Registros desta etapa

- [Registro 113](#report-113) — H5: oportunidade entre canais de aquisição com relógios comuns
- [Registro 114](#report-114) — Contexto multicore nominalmente alinhado: gate linear não passou

<a id="report-113"></a>

## Registro histórico 113 — H5: oportunidade entre canais de aquisição com relógios comuns

Fonte original preservada no arquivo: `113-CLOCK_ALIGNED_MULTICORE_OPPORTUNITY_PROTOCOL.md`.

<a id="report-113-h5-oportunidade-entre-canais-de-aquisição-com-relógios-comuns"></a>

**15/11/2026. Registrado antes dos novos recortes e ajuste.**

<a id="report-113-pergunta-limitada"></a>

#### Pergunta limitada

Nos dois grupos multicore de **treino**, verificar dependência Ch1→Ch2 a clock
exatamente comum, mantendo tiles espaciais completos. Isso não é intercalar
pares/ímpares espaciais do H2 e não resolve sozinho as quatro famílias.
Não alegar mapeamento dos números de núcleo apenas pelos nomes de pasta.

<a id="report-113-seleção-e-alinhamento"></a>

#### Seleção e alinhamento

- Usar exclusivamente os arquivos train do inventário 111.
- Par candidato: mesmo grupo, datasets previamente auditados terminando em
  `Ch1`/`Ch2`, pelo menos um clock float64 idêntico.
- Exigir Spacing, Origin, Extent espaciais e GaugeLength iguais nos cabeçalhos.
- Usar somente a interseção de canais retidos dos mapas dos dois blocos. Publicar
  contagens fora dessa interseção; isto é diagnóstico parcial, não codec de
  cobertura integral com canais omitidos.
- Não parear por frequência no filename, não interpolar clocks ou amplitudes.
- Por par: primeiro, central e último blocos de clock comum; janela central
  de 1.000 amostras em cada bloco, sem continuidade inventada entre eles.
- Ordenar pares por IDs dentro de cada grupo: primeira metade arredondada para
  cima em fit; restante em check. Os dois lados de um par não podem atravessar
  esses papéis. Rejeitar ambiguidades em que um arquivo participe de vários pares.
  Esses papéis são **novos e específicos deste diagnóstico**, não reescrita de P1.

Gravar pares/coordenadas/hashes antes do ajuste. Os recortes exatos podem ser
exportados em NPZ de pesquisa train-only para GPU remota: os HDF5 originais
estão locais e não estão presentes na cópia remota.

<a id="report-113-diagnóstico-original-e-eventual-contexto-decodificado"></a>

#### Diagnóstico original e eventual contexto decodificado

Normalização FNE1 global separada por núcleo. Ajustar um único coeficiente
Ch1→Ch2 por grupo, sem intercepto, limitado a [-2,2], somente no fit.
Usar alvos normalizados e máscara explícita; a estatística não transmite
gratuitamente a média de Ch2: um codec posterior teria de pagar ambos os sides.

Publicar erro de previsão / energia de Ch2 em fit/check, com contagem de células,
sem interpretar isso como ganho de compressão. Essa razão pooled é **diagnóstico
de oportunidade**, não NRMSE* macro prospectiva ou limite de entropia.

**Gate prévio para a segunda etapa:** redução de energia de erro ≥10% no fit
dos dois grupos usando Ch1 original. Se passar, codificar/decodificar Ch1 com
o VQ maduro congelado e FAK1, ajustar novamente o mesmo coeficiente somente no
fit e repetir os checks com esse contexto efetivamente recuperado. Probabilidades
dos índices ajustadas só nos novos pares fit. Publicar os bytes de contexto,
side de Ch2 e coeficientes, explicitamente **custo parcial sem inovação**, não
taxa de um codec final.

Sem ajuste de lag, nova rede ou extensão neste protocolo. Se original não
passar, documentar o negativo do preditor linear, não refutar toda arquitetura
multicore. Se decodificado mantiver oportunidade, um novo protocolo deverá cobrar
referência + inovação + parâmetros + framing e comparar contra codificação
independente de **ambos** os núcleos, com mesmo alvo e cobertura.

Zero updates. Clocks comuns não garantem sincronismo fino; não chamar os pares
de novas aquisições independentes. Sem sinais val/test/EDAS.
Destino: `results/experiments/clock_aligned_multicore_probe_20261115/`.

<a id="report-113-resultado"></a>

##### Resultado

18 pares e 54 recortes concluídos. O contexto original não passou o gate de
10% nos dois grupos; a etapa de contexto VQ decodificado não foi lançada.
Resultado e hipótese seguinte no [relatório 114](99-CLOCK_ALIGNED_MULTICORE_OPPORTUNITY.md#report-114).

---

<a id="report-114"></a>

## Registro histórico 114 — Contexto multicore nominalmente alinhado: gate linear não passou

Fonte original preservada no arquivo: `114-CLOCK_ALIGNED_MULTICORE_RESULTS.md`.

<a id="report-114-contexto-multicore-nominalmente-alinhado-gate-linear-não-passou"></a>

**15/11/2026.** Resultado do protocolo 113, sem treino/inferência neural.

Foram identificados **18 pares Ch1/Ch2**, nove por grupo multicore de treino,
com clocks float64 exatamente comuns e geometria espacial compatível.
Os novos papéis foram definidos por par: cinco fit/quatro check por grupo,
mantendo os dois lados juntos. Não alteram os papéis históricos dos estudos P1.

54 janelas centrais foram recortadas em até três blocos de cada par. A
interseção de canais retidos foi explícita e não há alegação de codec completo
ou de cobertura dos canais fora dela. Clocks comuns são alinhamento nominal,
não prova de sincronismo fino.

| Grupo | Coeficiente fit | Erro/energia fit | Erro/energia check |
|---|---:|---:|---:|
| C1 and C4 | 0,021272 | 0,999548 | 0,999895 |
| C2 and C3 | 0,009986 | 0,999900 | 0,999772 |

O erro de previsão caiu somente **~0,045% / ~0,010% no fit**, muito abaixo
dos 10% registrados nos dois grupos. **Não lançar a etapa de contexto VQ
decodificado**, nem treinar outro preditor sobre este negativo linear.
Não trocar lag ou coeficiente após observar os checks.

Isso não refuta dependência não linear, em outras bandas ou em outras
aquisições, nem estabelece entropia irredutível. A razão pooled normalizada
não é NRMSE* macro física ou ganho de compressão.
Os sides de ambos os núcleos e a inovação ainda precisariam ser pagos.

Artefatos: `results/experiments/clock_aligned_multicore_probe_20261115/`,
incluindo seleção, recortes train-only, hashes, coeficientes e gate falso.
Teste, validação e EDAS não foram abertos.

<a id="report-114-continuação-normalização-interna-não-condicionamento-por-configuração"></a>

#### Continuação: normalização interna, não condicionamento por configuração

A recomendação 90 também distinguia **normalização local interna** de
condicionamento físico. A primeira não requer inferir unidades de pulso ou
novas sessões e não oferece rótulos/estatísticas de alvo gratuitos ao decoder.

Testar uma via residual de estatística local dentro da normalização, com
contribuição inicialmente zero e controle igualmente adaptado. Manter a
função inicial do VQ maduro, seu quantizador/grade/coder e a normalização
reversível externa. O teste não presume que GroupNorm seja a causa do déficit.
É diferente da normalização por canal de entrada e de FiLM por configuração,
ambos com limitações históricas já declaradas.

O protocolo 115 fixa o contraste e seu orçamento antes do treinamento.

## Segunda pergunta H5: magnitude condicional, não predição da média

**Protocolo registrado em 16/11/2026, antes do novo ajuste.**
O gate linear histórico acima permanece reprovado; não repetir coeficiente,
lag ou preditor de média, nem lançar o codec VQ condicionado que ele vetou.
Zero correlação linear não exclui dependência de variância/magnitude.
Testar essa pergunta distinta com um coder neural de magnitude e **mesma
reconstrução física** em todos os braços, incluindo custo do núcleo de contexto.

### Dados e contrato

Reutilizar exatamente os 54 recortes/18 pares já salvos e seus hashes:
30 recortes de dez pares fit; 24 de oito pares check, dois grupos multicore.
Manter ambos os núcleos e todos os blocos de cada par no mesmo papel.
Somente as linhas comuns retidas, como no estudo original; não alegar
cobertura das linhas fora da interseção. Clocks são alinhamento nominal,
não certificação de sincronismo fino. Nenhum HDF5 novo/val/test/EDAS.

Quantizar ambos os núcleos separadamente pelo caminho H4 de erro absoluto:
`epsilon = 0,03 × escala_fit_do_grupo`, mesmo epsilon float64 transmitido,
escapes e restauração float32. Primeiro núcleo A é sempre decodificado antes
do contexto neural. Todos os braços devem reproduzir exatamente os coeficientes
e bits físicos do baseline independente A+B; não trocar erro por bytes.
Precisão experimental, não tolerância operacional validada.

### Braços, contexto e coder

Três MLPs `8→32→32→1`, ReLU, saída inicial zero, seed106:

- **none:** seis features de contexto zero + one-hot dos dois grupos;
- **aligned:** seis features do núcleo A reconstruído + mesmo grupo;
- **shifted:** mesmo A reconstruído deslocado circularmente em meio recorte
  temporal, fixado antes dos resultados. Não é busca/ajuste de alinhamento.

Features de A: magnitude/sinal logarítmicos, energia local temporal de nove
amostras, energia espacial de três linhas e vizinhos temporais anterior/posterior.
Centralização/escala ajustadas somente nos pares fit e transmitidas.
Ausência de contexto no controle não recebe estatísticas da janela B.
O outro núcleo não é dado gratuitamente em nenhum braço.

Alvo de treino: `log2(1 + abs(q_B)/RMS_fit(q_B))`, MSE; é proxy de magnitude,
não entropia exata. **4.096 updates/braço, batch4.096 células**, Adam LR0,001,
mesma sequência de grupo → par → recorte → células, sem seleção de checkpoint.
Apenas fit ajusta pesos, normalizadores e sete limiares de quantis das saídas
do modelo, por grupo, formando oito buckets. Nenhum ajuste nos checks.

Codificar q_B em até oito arrays, particionados pelos buckets derivados de A
reconstruído no receptor. Usar o mesmo coder inteiro e tabelas cobradas do
baseline; transmitir modelo, limiares, ordem, SHA dos coeficientes, framing,
escapes, geometria, IDs dos dois núcleos e clocks. Arrays vazios não são enviados.
Escolher por **bytes completos** entre B independente e B particionado.
O modelo continua cobrado mesmo se todas as janelas usarem fallback.
Além da correção B, cobrar o frame A inteiro em todos os braços e baseline.

### Decisão e limites

Oportunidade exige aligned ≥5% menos bytes conjuntos que baseline independente,
incluindo instalação única do MLP/normalizadores/limiares, menos bytes que
**ambos** os controles none/shifted e nenhum aumento de frames por grupo.
Verificar coeficientes, bits físicos, bound e CRC no receptor separado em
todos os pares. Sem recompressão do contexto original gratuito, seleção de lag,
novo número de buckets/seed ou extensão para salvar resultado.

Ganho igual ao shifted não demonstra dependência pontual nominalmente alinhada;
pode refletir estatística de janela. Resultado negativo não refuta modelos de
entropia neurais em geral. Mesmo positivo é oportunidade multicore em alta
precisão, não solução para quatro famílias ou promoção sobre VQ ultracompacto.
O histórico linear e seus budgets continuam separados.

Destino: `results/experiments/multicore_conditional_magnitude_20261116/`.

### Preflight de integridade

Os papéis congelados foram conferidos: 30 recortes/dez pares fit e 24/oito
pares check. Um único recorte **fit** passou roundtrip local de ambos os
núcleos, bound e receptor com setup separado; zero updates. Registro em
`results/experiments/multicore_conditional_magnitude_preflight_20261116/PREFLIGHT.json`.
Isso é smoke de engenharia, não resultado de compressão ou calibração nos checks.
A escala de precisão é a prospectiva de treino já congelada, não reajustada
aos novos fit/check deste contraste. Os papéis H5 não criam confirmação independente.
**498 testes locais passaram**, incluindo controle deslocado fixo, custos de
ambos os núcleos, buckets com replay sem fallback, rejeição de modelo/grupo
incorreto e igualdade dos bits físicos. Budget científico: 12.288 updates
de MLP e 50.331.648 apresentações de células-alvo, não novas amostras.

### Resultado da magnitude condicional — 16/11/2026

Os três MLPs completaram os 4.096 updates, com a mesma sequência de
grupo/par/recorte/células e os mesmos normalizadores fit. Pesos e limiares
foram congelados antes dos checks. Os modelos diferem e têm pesos não nulos;
não houve substituição por um modelo inativo ou interrupção de treino.

Avaliação em **24 recortes / oito pares / 16 arquivos / dois grupos /
3.360.000 células conjuntas A+B**. Não misturar esta coorte com os 136 checks
de outras etapas ou tratar os dois núcleos como réplicas independentes.

| Codec | Bytes conjuntos completos ↓ | bpc ↓ | Fallbacks |
|---|---:|---:|---:|
| A+B independentes, erro controlado | **6.054.017** | **14,414326** | — |
| Modelo sem contexto | 6.062.208 | 14,433829 | 24/24 |
| Contexto deslocado fixo | 6.062.208 | 14,433829 | 24/24 |
| Contexto alinhado nominalmente | 6.062.208 | 14,433829 | 24/24 |

Cada setup neural custa **8.191 bytes**, incluindo pesos, normalizadores e
limiares. Nenhum frame candidato foi menor que o independente; todos os braços
pagaram o mesmo aumento de **0,135299%** após instalar o modelo. Não houve
economia contra baseline ou vantagem do contexto alinhado contra os controles.

O receptor separado reproduziu exatamente os coeficientes de B e os bits
físicos de **ambos** os núcleos; nenhuma violação do bound. Portanto não há
troca escondida de erro por bytes. A taxa alta se refere à precisão experimental
prescrita; não é comparável diretamente ao VQ ultracompacto de outra coorte.

### Diagnóstico descritivo dos candidatos rejeitados

Sem alterar pesos, limiares, buckets, lags ou abrir waveforms adquiridos,
instrumentar o mesmo encoder congelado para observar os frames descartados:

- aligned: B ficou **12,287%–20,202% maior** que B independente;
- shifted: B ficou **12,504%–19,924% maior**;
- none: B ficou **0,430%–4,023% maior**.

Foram forçadas 72 reconstruções dos candidatos rejeitados, apenas para
verificar o decoder: todos os coeficientes e bits físicos de B coincidiram
com o baseline. Isso **não** substitui a seleção por bytes nem cria novos
resultados elegíveis. A instrumentação não mudou o retorno do encoder.

Dividir em buckets tem tabelas/metadados próprios e pode perder estrutura de
ordem/localidade do coder original. Assim, a falta de ganho líquido deste
desenho não estima a entropia condicional do sinal nem prova independência
entre núcleos. Uma loss de magnitude menor no fit também não é ganho de codec.

### Decisão e auditoria

**Gate reprovado. Sem extensão, novo número de buckets, lag/seed, promoção ou P3.**
O teste respondeu à pergunta de magnitude condicional neste mecanismo, sem
reclassificar o antigo gate linear ou lançar sua etapa VQ vetada.
Nova coleta/sincronismo/proveniência continuam necessários para conclusões
mais fortes; não declarar esgotada toda modelagem não linear.

189 arquivos/modelos recuperados com SHA-256, total 36.763.108 bytes remotos.
Auditados **144 frames conjuntos selecionados** (baseline/candidato), incluindo
CRCs, custos completos, disjunção de pares/arquivos e snapshots, além das 72
reconstruções forçadas de B acima. Nenhum treino repetido nem reserva aberta.
Evidências: `RECOVERY_AUDIT.json`, `POSTFLIGHT_AUDIT.json`,
`COMPARISON.json` e `REJECTED_CANDIDATE_DIAGNOSTIC.json`.
