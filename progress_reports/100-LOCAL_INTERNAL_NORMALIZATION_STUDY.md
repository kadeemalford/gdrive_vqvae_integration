# Normalização interna local

**Etapa consolidada dos registros 115, 116.** Via residual ativa, sem avanço prático; sem extensão.

Protocolos, resultados e auditorias foram reunidos sem alterar resultados, budgets ou gates. As partes históricas seguem em ordem: estados de lançamento/execução descrevem aquele momento, não tarefas ainda pendentes. Consulte as últimas partes para o encerramento.

## Registros desta etapa

- [Registro 115](#report-115) — Normalização interna local: via residual neutra, sem bits adicionais nominais
- [Registro 116](#report-116) — Normalização interna local: via ativa, sem avanço prático no piloto

<a id="report-115"></a>

## Registro histórico 115 — Normalização interna local: via residual neutra, sem bits adicionais nominais

Fonte original preservada no arquivo: `115-LOCAL_INTERNAL_NORMALIZATION_PROTOCOL.md`.

<a id="report-115-normalização-interna-local-via-residual-neutra-sem-bits-adicionais-nominais"></a>

**15/11/2026. Registrado antes de smoke/treino.**

<a id="report-115-hipótese-isolada"></a>

#### Hipótese isolada

O VQ maduro usa GroupNorm, cujas estatísticas agregam posições espaciais e
temporais do tile. Testar se uma via estatística **local por posição** melhora
a representação full-tile, sem mudar a normalização reversível de entrada,
resolução, grade, dicionário, coder ou alvos. Não presumir causalidade a partir
dos negativos de H1/H2 ou do baixo rank latente.

Isso implementa a segunda hipótese H3 da revisão 90, **não** o adaptador de
metadados adiado em 110. Não recebe nomes de configuração, família, clocks,
pulso, alvos de decoder ou dados reservados.

<a id="report-115-construção-e-função-inicial"></a>

#### Construção e função inicial

Partir do VQ denso maduro época23 nos dois braços. Em cada GroupNorm existente:

```text
global = original_GroupNorm(x)
local = normalize_over_features_in_each_original_group_at_each_position(x)
output = global + 0.5*tanh(gate) * (local - global)
```

A via local usa os mesmos grupos de features, epsilon e affine original,
mas não agrega altura/largura. Um gate escalar por camada, inicializado em
**zero**, mantém a função original antes do treino.

- Controle global: via extra desligada, gates congelados em zero.
- Candidato local-residual: gates ativos, coeficiente limitado a [-0,5;0,5].
  É uma perturbação residual aprendida, não substituição completa por LayerNorm.
- Mesmo encoder/decoder/backbone e mesma fonte; somente os poucos gates têm
  parâmetros treináveis adicionais no candidato, com custo de pesos declarado.
- Exigir igualdade bit a bit da função inicial e índices com a fonte, no maior
  recorte fit, antes de qualquer update; não usar substituição arbitrária congelada
  como refutação da hipótese.

Decoder usa somente pesos compartilhados e índices pagos. A normalização local
não adiciona side por janela. Bytes reais podem mudar pela distribuição de IDs.

<a id="report-115-dados-dose-e-seleção"></a>

#### Dados, dose e seleção

Mesma sequência de 5.600 janelas de 44 arquivos fit; 43 arquivos check excluídos
dos novos batches. Seed **2117**, já registrada no controle full-tile, sem buscar
nova seed favorável. Fonte herdada viu o desenvolvimento historicamente.
Novo controle é treinado junto: não usar um braço histórico de outro código
como se fosse a execução pareada.
A separação por arquivo não certifica sessões independentes; a auditoria 111
encontrou clocks compartilhados entre alguns arquivos multicore fit/check.
Não chamar este piloto de confirmação independente.

**8.192 updates por braço, 16.384 científicos**. AdamW novo em ambos, LR1e-4,
wd1e-5, clip1, FP32 determinístico. Loss comum 0,5 MSE+0,5 L1 mascaradas+VQ.
Quantização hard exata; uma grade de IDs K2048, 11 bits nominais por posição.

Mesmos 45 checks do piloto 105, incluindo tiles pequenos; seleção inicial
elegível por `macro_familias NRMSE*² + 0,1*bpc/0,7`, checks a cada 2.048.
Calibração de contagens apenas nos mesmos 44 recortes fit. Modo literal,
escapes, registros e transporte comuns.

Smoke de dois updates e teste 1+resume versus 2 contínuos em ambos os braços,
com os mesmos eventos de seleção. Igualdade de checkpoint inteiro obrigatória.
Oito updates técnicos. Gates do candidato devem receber gradientes finitos e
não nulos em conjunto; controle não os treina.

Após o piloto, avaliar selecionados/endpoints e fonte madura nos 136 checks
ampliados, sem reseleção. Sem novos waveforms val/test/EDAS.

<a id="report-115-continuação-e-limites"></a>

#### Continuação e limites

Não há extensão automática. Apenas registrar elegibilidade futura se endpoint
local melhora ≥1% contra origem, ≥0,2% no último bloco, fica até 3% acima do
controle selecionado; selecionado custa até 105% dos bytes do controle e mantém
perplexidade ≥16 e cobertura/replay válidos. Não relaxar thresholds.

Isso não promove: para avanço prático, exigir ≥5% menos erro sem mais bytes,
comparação por grupo e guardas físicas, depois confirmação apropriada.
Não usar tolerâncias desconhecidas da aplicação para apagar regressões.
Os nomes legados `initial_product`/`flat_rmse` do helper de gate significam
aqui candidato local/controle global, explicitados por `kind` e nomes dos braços.

Implementação versionada separada; fontes exatos do piloto105 foram arquivados
por SHA antes da extensão dos loaders/runners. Resultados históricos intactos.
GPU0 global, GPU1 local. Destino:
`results/experiments/local_internal_norm_fold0_20261115/`.

<a id="report-115-lançamento"></a>

#### Lançamento

**461 testes locais passaram.** O supervisor foi lançado com `--kind local-norm`,
começando por smokes, identidade da função e retomada. Somente após essas
guardas ele lança os dois pilotos e, ao final, a avaliação ampliada.
Nesta atualização não há resultado científico nem confirmação de promoção.

<a id="report-115-integração-confirmada"></a>

##### Integração confirmada

Os dois braços passaram identidade de bits float32 normalizados e índices
com a fonte, incluindo o maior recorte de ajuste. Foram identificadas **28
GroupNorm**: o candidato acrescenta 28 gates escalares, 112 B em tensores
float32, mais os custos reais de serialização reportados na avaliação.
O controle mantém esses gates em zero, sem treiná-los.

`RESUME_INTEGRATION.json` confirmou igualdade do checkpoint inteiro em ambos
os braços. O supervisor iniciou os pilotos científicos: GPU0 global,
PID inicial **1728203**; GPU1 local, PID inicial **1728204**.
Comandos em `LAUNCH_pilot_*.json`. O teto permanece 8.192 por braço,
seguido da avaliação ampliada, sem extensão automática.

<a id="report-115-encerramento"></a>

##### Encerramento

Pilotos e avaliação ampliada concluídos; gate reprovado e nenhuma extensão.
Resultados, integridade, regressões e limites no
[relatório 116](100-LOCAL_INTERNAL_NORMALIZATION_STUDY.md#report-116).

---

<a id="report-116"></a>

## Registro histórico 116 — Normalização interna local: via ativa, sem avanço prático no piloto

Fonte original preservada no arquivo: `116-LOCAL_INTERNAL_NORMALIZATION_RESULTS.md`.

<a id="report-116-normalização-interna-local-via-ativa-sem-avanço-prático-no-piloto"></a>

**15/11/2026.** Resultado completo do protocolo 115.

<a id="report-116-execução-e-integridade"></a>

#### Execução e integridade

Os dois braços concluíram **8.192 updates cada / 16.384 científicos**, mais
oito updates técnicos de integração. Mesma sequência e **655.595.000
apresentações de células por braço**. Smokes e retomada confirmaram a função
inicial e igualdade do checkpoint inteiro.

O controle manteve os 28 gates em zero. Uma auditoria adicional confirmou
que seus tensores de base no endpoint são **exatamente iguais** aos do controle
flat do piloto105 sob a mesma dose/seed. O wrapper não alterou silenciosamente
a adaptação do controle. Isso não exige igualdade dos IDs ou bytes de
descritores serializados, que identificam exports distintos.

Os seis conjuntos ampliados (selecionados/endpoints/fontes, dois braços)
cobriram **136 recortes / 43 arquivos / dez configurações / quatro famílias /
12.799.000 células**. Contagem e soma dos tamanhos de frames conferidas;
replay físico bit-exato com receptores independentes.

Artefatos: `results/experiments/local_internal_norm_fold0_20261115/`,
com `POSTFLIGHT_AUDIT.json`, `CONTROL_REPRODUCIBILITY.json`, frames,
checkpoints, gate e fontes arquivados/verificados por SHA-256.

<a id="report-116-seleção-e-resultado"></a>

#### Seleção e resultado

Global selecionou a **origem, update0**; local selecionou **4.096**.
Não trocar a seleção pelos scores ampliados.

| Modelo | Update | NRMSE* macro famílias ↓ | Stream bpc ↓ |
|---|---:|---:|---:|
| Fonte madura, export full-tile comum | — | **0,490336** | **0,679244** |
| Global selecionado | 0 | **0,490336** | 0,679246 |
| Local selecionado | 4.096 | 0,490584 | 0,680526 |
| Global endpoint | 8.192 | 0,491848 | 0,678543 |
| Local endpoint | 8.192 | 0,490609 | 0,680016 |

O selecionado local tem **0,0506% mais erro e 0,1884% mais bytes** contra o
controle selecionado. O endpoint melhora ~0,252% contra o endpoint global,
mas custa ~0,217% mais bytes e não supera a origem em erro.

Os três bytes a mais do global selecionado frente à fonte são do registro
comprimido associado a outra identidade de export, **não** perda de waveform.
Exports completos: 111.875.741 B nos braços com gates versus 111.867.797 B
na referência exportada, incluindo encoders. Embora os gates somem só 112 B
de tensores float32, o arquivo também paga metadados/serialização; não chamar
a diferença de arquivo de apenas 112 B ou alegar receptor mínimo.

<a id="report-116-guardas-e-estado-da-via-local"></a>

#### Guardas e estado da via local

O selecionado piora em single-mode **3, 4, 7 e 8**; em configuração4 o erro
sobe ~4,46%. Macro quase igual não significa preservação por configuração.

Auditoria pareada: 18/43 arquivos com NRMSE* maior, 21/43 com p90 maior,
18/30 pares configuração×banda com erro espectral maior e 17/30 com pior
distância RMS a 1. Fase segue descritiva, com suporte/cobertura explícitos.

A via não ficou desligada por falha de implementação: gradientes passaram
os smokes e os coeficientes aprendidos foram não nulos. No selecionado, o
valor absoluto médio de `0,5*tanh(gate)` foi ~0,001903, com intervalo
[-0,001356; 0,008678]. No endpoint, média ~0,003543 e intervalo
[-0,002593; 0,014511]. São perturbações pequenas da normalização global.

<a id="report-116-gate-e-decisão"></a>

#### Gate e decisão

O endpoint local dos checks do piloto ficou em 0,454559 contra 0,450803
inicial: não melhorou ≥1% contra a origem. O último bloco melhorou e o
quantizador permaneceu saudável, mas os critérios são conjuntos.
`eligible_for_further_investment: false` foi preservado.

**Sem extensão, promoção, P3 ou sweep de LR, ganho de gates ou nova seed.**
O negativo limita esta via residual, fonte, dose e loss; não refuta qualquer
normalização local ou treino de outra arquitetura desde a origem.
Não usar os coeficientes pequenos como justificativa retrospectiva para mudar
o budget/gate até obter um positivo.

Nenhum waveform de validação, teste ou EDAS foi aberto neste piloto.
Consolidação da sequência e decisões no relatório 117.
