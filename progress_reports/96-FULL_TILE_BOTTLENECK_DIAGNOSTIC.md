# Diagnóstico de gargalos full-tile

**Etapa consolidada dos registros 103, 104.** Descritores de erro, índices e covariância; não são limites informacionais.

Protocolos, resultados e auditorias foram reunidos sem alterar resultados, budgets ou gates. As partes históricas seguem em ordem: estados de lançamento/execução descrevem aquele momento, não tarefas ainda pendentes. Consulte as últimas partes para o encerramento.

## Registros desta etapa

- [Registro 103](#report-103) — Diagnóstico full-tile antes de outro contraste neural
- [Registro 104](#report-104) — VQ full-tile: erro pouco correlacionado e latente concentrado

<a id="report-103"></a>

## Registro histórico 103 — Diagnóstico full-tile antes de outro contraste neural

Fonte original preservada no arquivo: `103-FULL_TILE_BOTTLENECK_DIAGNOSTIC_PROTOCOL.md`.

<a id="report-103-diagnóstico-full-tile-antes-de-outro-contraste-neural"></a>

**15/11/2026. Registrado antes da inferência desta etapa.**
Continuação autorizada após o resultado 102. Não prolongar H2/observação completa.

<a id="report-103-pergunta-e-população"></a>

#### Pergunta e população

Separar três descritores do VQ maduro: custo por posição e padding; estrutura
temporal/espacial do erro residual; ocupação e geometria do erro de quantização.
Um erro de quantização latente pequeno não garante erro físico pequeno, e o
bypass contínuo não é teto causal. Não usar nenhum dos dois como prova de que
um novo quantizador funcionará.

Usar somente o treino preparado e a fonte densa congelada época 23:

- 44 recortes centrais de ajuste, já usados para calibrar registros;
- 129 checks históricos de desenvolvimento dentro do treino;
- um tile mínimo por arquivo, escolhido por canais, comprimento e caminho,
  para não restringir o diagnóstico aos maiores tiles;
- deduplicar caminhos; os recortes mínimos herdam o papel fit/check do arquivo.

Gravar lista e hashes antes de abrir os sinais. Calibrar contagens FAK1 apenas
nos 44 recortes de ajuste originais. Inferência em uma GPU, zero updates.
Sem acesso a waveforms de validação/teste ou EDAS reservado nesta etapa.

<a id="report-103-medições-e-limites"></a>

#### Medições e limites

1. Stream completo real, replay de receptor separado, identidade dos índices,
   literal comum e escapes bit-exatos; pesos identificados e cobrados à parte.
2. Pooling físico por canal/arquivo e bandas, com escalas de treino congeladas.
   Separar papéis fit/check; a amostra favorecida por metadados não representa
   toda a população de treino.
3. Autocorrelação de lag 1 temporal/espacial do erro e do alvo, somente em pares
   ordinários válidos. Escapes não podem fabricar erro artificial nem pares
   através de regiões ausentes. Correlação não demonstra economia realizável.
4. Covariância de z e z−q e espectro/effective rank descritivos por papel.
   Não alimentar z contínuo ao receptor, não recalibrar pesos com esses números.
5. Por shape: posições efetivas, padding externo, payload e framing. A área
   externa não corresponde automaticamente a índices removíveis; convoluções
   e GroupNorm misturam posições.

Critério de decisão: investir numa representação do tile completo com a mesma
grade e teto nominal por posição somente se houver um gargalo concreto a isolar;
não aumentar livros/streams sem um controle a bytes comparáveis. A alternativa
de produto do relatório 89 não foi refutada pelo seletor patológico, mas tampouco
está aprovada por essa falha. Qualquer novo ensaio deve ter protocolo próprio,
inicialização declarada, controle full-tile, métricas prospectivas e budget.

Destino: `results/experiments/full_tile_bottleneck_fold0_20261115/`.
Resultados e escolha seguinte serão registrados separadamente.

---

<a id="report-104"></a>

## Registro histórico 104 — VQ full-tile: erro pouco correlacionado e latente concentrado

Fonte original preservada no arquivo: `104-FULL_TILE_BOTTLENECK_RESULTS.md`.

<a id="report-104-vq-full-tile-erro-pouco-correlacionado-e-latente-concentrado"></a>

**15/11/2026.** Resultado do protocolo 103, sem treinamento ou validação nova.

<a id="report-104-execução-e-limites"></a>

#### Execução e limites

224 recortes de treino deduplicados: **88 fit e 136 check**, de 44/43 arquivos.
O registro de probabilidades foi ajustado somente nos 44 recortes centrais
originais de fit. Os demais recortes acrescentam os menores tiles por arquivo.
Todos os frames passaram replay físico bit-exato com receptor independente;
constantes tiveram modo literal comum. Fonte congelada permaneceu inalterada.

| Papel | NRMSE* macro famílias | Stream bpc |
|---|---:|---:|
| Fit, diagnóstico | 0,464825 | 0,694788 |
| Check, diagnóstico | 0,490336 | 0,679245 |

Essas populações não são a coorte idêntica dos relatórios 94/102. Não atribuir
a diferença de score a ganho de codec. Cinco recortes constantes foram
transmitidos literalmente (três fit, dois check).

<a id="report-104-dependência-residual-e-geometria"></a>

#### Dependência residual e geometria

Nos checks, correlações de lag 1, centradas por canal/janela e somente em
pares ordinários válidos:

| Objeto | Espacial | Temporal |
|---|---:|---:|
| Alvo normalizado | 0,749100 | 0,152224 |
| Erro hard do VQ | 0,066431 | 0,036874 |

No fit, o erro teve ~0,090327/~0,058158. Uma predição linear local simples
tem pouca dependência remanescente a explorar nessa medição; não é prova de
independência, limite informacional ou fracasso de qualquer contexto neural.

A covariância descritiva do latente mostrou forte concentração:

| Objeto | Rank efetivo fit | Rank efetivo check | Variância top16, check |
|---|---:|---:|---:|
| z | 4,300 | 4,341 | 99,9974% |
| z−q | 5,654 | 5,508 | 99,9707% |

Foram observadas 1.172/1.173 entradas ativas, fit/check, de 2.048; entropias
marginais empíricas ~9,739/~9,646 bits por índice, **sem tabelas/framing**.
A covariância é pooled por posição, não macro por família. Padding, aquisição e
normalização podem influenciar a concentração. Não demonstra perda removível
nem equivale a um decoder contínuo ideal.

Os menores tiles têm imposto de padding grande: um tile não constante de
2×1.000 transmite 1.000 posições, frente a 32.000 em 512×1.000. Área externa
de padding não identifica índices descartáveis, pois análise/síntese e
GroupNorm misturam regiões. A próxima comparação manterá exatamente o layout.

<a id="report-104-decisão"></a>

#### Decisão

Não priorizar outro corretor preditivo simples do waveform. Testar um contraste
**full-tile de geometria do dicionário**, mantendo K composto, grade e coder:
flat2048 versus produto32×64, com adaptação conjunta e controle igualmente
treinado. A concentração motiva testar alocação de centroides entre direções,
mas não prediz que produto melhorará a síntese.

Diferença para 89: fonte **densa**, contrato prospectivo físico, literal comum,
FAK1 compacto e checks incluindo tiles pequenos. Não corrigir retrospectivamente
o ranking/gate de 89 nem chamar a repetição de confirmação daquele estudo.

O protocolo 105 fixa inicialização, métrica e orçamento antes do ensaio.
Artefatos: `results/experiments/full_tile_bottleneck_fold0_20261115/`.
