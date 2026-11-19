# Viabilidade FEBUS: contrato, decisões científicas e execução delegável

**17/11/2026.** Desenvolvimento da seção 9 do
[relatório 106](106-FEBUS_FEASIBILITY_AND_RESEARCH_REDIRECTION.md).
Autorização atual: avançar nas tarefas de definição/discernimento e delegar as
determinísticas, preferencialmente ao GPT‑6.1 Sol. Este documento registra as
decisões novas; o 106 e os resultados anteriores permanecem históricos.

## 1. Resultado desta etapa

**A próxima pergunta executável é dupla:** quanto a mesma reconstrução muda por
contexto, e quanto custa preservar valores com precisão explícita usando apenas
informação disponível no decoder? Nenhuma resposta isolada identifica um limite
informacional ou prova generalização entre sessões.

Decisões tomadas:

1. F0 terá um **contrato exploratório fechado**, com tolerâncias operacionais
   explicitamente pendentes. Isso permite medir a fronteira sem escolher pelo
   usuário o erro aceitável para a aplicação.
2. F1 começa com síntese retrospectiva e um painel pequeno, pareado e congelado
   por metadados. Variar contexto e normalização separadamente; manter os pesos.
3. F2 começa por quantização **direta**, controle marginal e contexto causal na
   **mesma sequência discreta**. Comparar limites absolutos iguais não basta:
   a reconstrução precisa ser bit-exata entre os coders.
4. Um diagnóstico de dependência deve detectar também **variância condicional
   sem previsibilidade da média**. Só repetir regressão/lag1 deixaria a principal
   hipótese nova sem teste.
5. F3 continua dependente de novas sessões documentadas. F4 permanece uma decisão
   posterior, com no máximo uma candidata e uma GPU remota.

Implementações novas:

- [`feasibility_metrics.py`](../finetuning/feasibility_metrics.py): suporte
  espectral definido apenas pelo alvo, perda de energia com fase ausente e
  estatísticas aditivas.
- [`feasibility_context.py`](../finetuning/feasibility_context.py): planejamento
  de contextos sobre as mesmas células, sem criar canais e com origens alinhadas.
- [`feasibility_protocol_v1.json`](../config/feasibility_protocol_v1.json):
  decisões e estados legíveis por máquina. **Não é um launcher.**
- [`freeze_feasibility_metadata.py`](../tools/freeze_feasibility_metadata.py):
  auditoria de manifesto, painel F1 e pool preliminar F2.

O contrato científico de F1 está definido; seu runner de inferência ainda deve
ser implementado. F2 exige primeiro fechar o manifesto temporal e a implementação
do coder. Não chamar essas etapas de experimentos já executados.

## 2. Distribuição de trabalho

| Etapa | Decisão de alto discernimento, feita aqui | Trabalho determinístico delegável |
|---|---|---|
| F0 | Alvo, estimandos, escalas, fase, taxa e distinção exploração/operação | Integrar métricas, validar serialização, gerar tabelas e curvas completas |
| F1 | Comparabilidade, painel pareado, fatores do diagnóstico, interpretação | Recuperar checkpoints identificados, implementar replay e executar até 480 forwards |
| F2 | Quantização comum, causalidade, controles identificáveis, critérios líquidos | Manifesto temporal, coder pequeno, testes de roundtrip, medidas CPU e relatório bruto |
| F3 | Desenho por sessões e fatores de aquisição | Curadoria dos registros e checks de integridade quando as sessões existirem |
| F4 | Escolha do mecanismo, controle e decisão de investir | Treinos sequenciais e medição depois da escolha |

Delegação não transfere ao executor a decisão de relaxar gates, trocar coortes,
buscar checkpoints mais favoráveis, aumentar budgets ou reinterpretar os negativos.
Falha de integridade ou ambiguidade de contrato retorna como pendência técnica.
Um resultado negativo válido encerra o contraste especificado.

## 3. F0 — Contrato físico e pergunta de viabilidade

### 3.1 O que será preservado

Alvo: valores float32 do produto armazenado, ruído incluído, na cobertura dos
mapas físicos existentes. Para este desenvolvimento, 20 kHz e canais retidos;
unidade e produto continuam identificados por arquivo. Não converter strain rate
em strain, filtrar bandas, remover médias gratuitamente ou recalcular máscara.
Não tratar esta cobertura retida como preservação de todos os canais do HDF5.

NaN, infinitos, payloads não finitos e zeros com sinal são auditados por bits.
Erros físicos são calculados sobre células finitas; a contagem das demais e seu
custo exato são publicados. Transformação afim, escapes e seleção de canais
precisam chegar ao receptor ou já integrar um contrato compartilhado identificado.

**Pendências do responsável pelo uso**, não impedimentos à exploração:

- Erro absoluto tolerável na unidade física; tolerâncias de fase, amplitude,
  viés, piores canais e caudas, eventualmente distintas por regime.
- Horizonte do objeto/stream, granularidade de acesso aleatório, memória e
  latência máximas, e se os pesos já estarão instalados no receptor.
- Necessidade real-time ou offline e cobertura física operacional desejada.

Nenhum valor exploratório abaixo será descrito como tolerância aprovada.
700× equivale a aproximadamente **0,045714 bits/célula float32**, uma referência
de taxa, não um critério de fidelidade.

### 3.2 Métricas: suporte fixo, ausência visível

Manter `PhysicalAccumulator` para RMSE, SSE, viés, máximo e NRMSE* prospectivo.
A escala P0 `s_g` vale **0,01 × mediana(std positiva)** das janelas de calibração
do grupo. Ela é um piso convencional de avaliação; não é ruído instrumental.
Reportar por canal físico, arquivo, configuração e família, com denominadores.
Quantis de erro absoluto exigem distribuição/histograma próprio: não derivá-los
de médias, nem confundir P90 entre canais com P90 entre células.

**Correção prospectiva de fase:** `prospective_spectral.py` usa suporte conjunto
do alvo e da reconstrução. Sua estatística histórica é condicional aos bins
restantes, não uma auditoria suficiente de preservação. A versão nova usa:

```text
X, Y = rFFT(Hann × (sinal − média)), em janelas de 1.000 amostras
S = {bins com frequência > 0 e |X|² > s_g² × soma(Hann²)}
w = |X|² × peso unilateral de energia
erro complexo suportado = sqrt(soma_S(|Y−X|² × peso) / soma_S(w))
energia com fase ausente = soma_{S e |Y|²≤piso}(w) / soma_S(w)
```

Fase observada é a média ponderada de `|arg(Y·conj(X))|` apenas onde ela é
observável, acompanhada obrigatoriamente da fração ausente. Também publicar
**penalidade conservadora** que atribui π aos bins ausentes no denominador fixo.
Essa penalidade não é um ângulo medido nem prova de oposição de fase.

Publicar erro de amplitude e erro complexo em todo o espectro, além do suporte.
Um alvo fraco continua nos erros físicos mesmo sem fase definida. Um decoder
que apaga o alvo terá erro complexo suportado 1 e energia ausente 100%, em vez
de desaparecer da métrica. Viés/DC é auditado no domínio físico separadamente.
Bandas AC: abaixo de 1 kHz, 1–5 kHz, 5–10 kHz, com Nyquist incluído uma vez.

Estatísticas são somadas antes de formar razões; não calcular média de ângulos
por batch. A implementação atual cobre janelas finitas; o integrador deve marcar
explicitamente janelas espectrais indisponíveis por valores especiais, preservando
suas células no ledger físico. Não substituir não finitos por zeros silenciosamente.
Não reavaliar históricos para trocar seus gates.

### 3.3 Precisão: retirar a ambiguidade dos alphas H4

Definir `sigma_cal_g = 100 × s_g`. Grade **exploratória**:

| ε / sigma_cal_g | 0,0001 | 0,0003 | 0,001 | 0,003 | 0,01 | 0,03 | 0,1 | 0,3 | 1 |
|---|---|---|---|---|---|---|---|---|---|
| ε / s_g | 0,01 | 0,03 | 0,1 | 0,3 | 1 | 3 | 10 | 30 | 100 |

Os três primeiros pontos correspondem às três precisões históricas H4, em ordem
inversa. Os demais cobrem a lacuna. Não são promessas de atingir taxa específica.
Cada grupo recebe ε físico fixado pela calibração, não pelo desvio do check.

Quantização proposta: origem zero, passo `2ε`, arredondamento nearest-even,
reconstrução float32 e verificação célula a célula de `|x−x_hat|≤ε`.
Falhas numéricas, valores especiais ou índices fora da faixa vão por escape exato.
O mecanismo de `bounded_residual_stream.py` com base zero serve como referência
de garantia, mas seu stream não substitui o novo contraste de distribuições.
As duas precisões **primárias** de oportunidade ficam fixadas em **0,03 e 0,1**;
as outras sete caracterizam a curva, sem seleção retrospectiva de um par vencedor.

### 3.4 Taxa e implantação

`R = 8 × bytes completos / células físicas únicas reconstruídas`.
Mostrar bytes de frames, tabelas, geometria, escalas, escapes, integridade, resets,
modelo e adaptação, evitando contar o mesmo componente duas vezes.
Custos de encoder não transmitidos são compute/armazenamento, não payload;
o tamanho do checkpoint de treino não é automaticamente o decoder mínimo.

Publicar dois contratos separados:

1. **Transporte com modelo instalado:** frames e todo side incremental;
   identificar o hash e o tamanho da instalação compartilhada.
2. **Objeto autocontido:** incluir decoder/modelo, tabelas e dependências
   necessárias à reconstrução. Medir o objeto real.

Horizontes ilustrativos: 50 ms, 1 s, 60 s e 1 h. Para `M` bytes adicionais de
modelo e economia de `d` bits/célula antes desse custo, break-even
`N > 8M/d`, se `d>0`. Isso só vale mantendo a mesma taxa e geometria de resets;
não extrapolar produtividade de um tile como velocidade de implantação.
Em 512 canais a 20 kHz entram 10,24 milhões de células/s ou 40,96 MB/s float32.

**Eficiência** exige menos erro sem mais bytes, ou menos bytes com reconstrução
equivalente. **Viabilidade** pode ser um ponto de maior taxa que atende precisão.
Não interpolar fronteiras históricas não sobrepostas para fabricar dominância.

## 4. F1 — Síntese dos artefatos e diagnóstico de contexto

### 4.1 O que já pode ser concluído

O inventário delegado conferiu os caminhos e coortes; os valores abaixo são
descritores históricos, não uma nova média ou confirmação:

| População | Evidência útil | Leitura permitida |
|---|---|---|
| 129 checks train / 43 arquivos | H2: NRMSE* 0,6290 contra 0,6798 do independente; madura 0,4913 | H2 melhora seu controle, mas continua atrás da referência madura |
| 136 checks train / 43 arquivos | Madura 0,4903 / 0,6792 bpc; produto 0,5081 / 0,6543 bpc | Produto troca bytes por erro; não domina |
| 7.220 janelas de validação consumida / 51 arquivos | Madura 0,5187 / 0,7045 bpc; H2 0,6490 / 0,7895 bpc | O contraste transfere contra o controle; não supera a madura |
| Ajuste de diversidade/FiLM | Crop 8×1.000 antes de `encode_robust`, avaliação full-tile | Mudam contexto espacial e normalização; contribuição causal ainda desconhecida |
| Fit/check do diagnóstico 96 | 88/136 janelas, arquivos distintos no ajuste novo | Exposição herdada e sessões não independentes impedem chamar o check de generalização nova |

Fontes principais, sob `results/experiments/`:

- `joint_anchor_vq_fold0_20261115/expanded/closed-loop/check_016384/results.json`;
  `expanded/independent/check_030720/results.json` no mesmo estudo;
  `expanded/closed-loop/practical_reference/results.json`.
- `dense_full_tile_product_v2_fold0_20261115/arms/product/expanded/selected/check_004096/results.json`.
- `frozen_development_validation_fold0_20261115/{mature_reference,h2_joint}/results.json`.
- `full_tile_bottleneck_fold0_20261115/results.json` e
  `frozen_dense_rate_match_fold0_20261115/MEASURED_FRONTIER.json`.
- `research_decision_registry_20261115/registry.json`, chave `cohorts`.

Não juntar 129 e 136 checks: há diferença de população e custos de framing.
`history.jsonl` contém loss de treino normalizada, não erro físico pareado nos
mesmos batches. A normalização local de 100 preservou uma via GroupNorm global;
seu negativo não elimina toda sensibilidade ao contexto.

### 4.2 Experimento mínimo definido

**Painel:** dois arquivos por configuração, uma janela por arquivo, escolhidos
dos checks train históricos por SHA-256 com prefixo `feasibility-v1|`.
Seleção antes de novas leituras de sinal, sem critério de energia/erro.
Manter arquivos estreitos, inclusive multimode; um tile de dois canais não tem
contraste espacial 8→full e deve aparecer como tal.

**Modelos fixos:** VQ maduro exportado, endpoint diversidade dez configurações
e endpoint FiLM constante. O primeiro separa sensibilidade de um modelo funcional
da sensibilidade dos controles fracos. Os outros dois foram treinados em crops.
Não escolher melhor seed/checkpoint depois do painel.

- Madura: `results/remote_setup/dense_transfer_fold0_weights_p93.pt`,
  origem `high_A/best_macro.pt`, época 23, SHA
  `b943db9f5cf6418e61ffca1814d788ac23fd6111468a425a918ba7d7cb5bc593`.
- Diversidade: `regime_diversity_curve_20261115/scientific/10/last.pt`,
  update 8.192.
- FiLM: `physical_conditioning_20261115/scientific/constant/last.pt`,
  update 16.384, condição constante com replay correto.

Os dois últimos nomes são os outputs prescritos pelos runners; os pesos não
estão presentes na cópia local consultada. Recuperar e verificar origem/update,
configuração e hashes a partir dos artefatos remotos antes de executar. Ausência
de peso verificável é bloqueio daquele braço, não licença para treinar substituto.

**Fatores cruzados:** contextos espaciais 8, 16, 32 e full, limitados à largura
real, × normalização nativa por contexto ou parâmetros afins do tile completo.
Tempo permanece 1.000 amostras. O planner implementado fixa o mesmo núcleo de
até dois canais, próximo ao centro entre as posições viáveis; origens múltiplas
de 16 reduzem mudança de fase da grade. As bordas/padding nativos continuam
parte da intervenção e precisam ser registrados.

No braço afim fixo, cortar o tensor normalizado completo e sua máscara, usar
a mesma inversa afim e os mesmos escapes. É uma intervenção de entrada para
diagnóstico, não um codec com normalização gratuita. Não recalcular quantis
por crop nesse braço nem transformar entrada limitada em observação full-tile.

**Primário:** RMSE físico sobre exatamente o núcleo fixo, antes de qualquer
agregação, por modelo/arquivo/contexto/norma. Alvo, escala, canais e suporte
espectral não podem variar com o braço. Publicar também erros absolutos,
NRMSE*, amplitude/fase e constantes. Zero no denominador gera razão indefinida
e diferença absoluta publicada, não um epsilon escolhido após o resultado.

**Sinal de sensibilidade predefinido:** diferença de RMSE de pelo menos 5%
entre um mesmo par de contextos e uma mesma condição de normalização, nos dois
arquivos de pelo menos duas configurações, para um modelo fixo. Publicar todos
os pares, inclusive regressões. É regra de triagem descritiva, não teste de
significância ou promoção. Outros padrões são exploratórios.

Se a diferença desaparece com afim fixo, a preparação de entrada ganha peso.
Se persiste, contexto interno/padding/quantização permanecem candidatos; não
atribuir automaticamente tudo a GroupNorm. Se os controles fracos continuam
ruins, isso ainda não distingue insuficiência de updates de representação.
Uma ausência neste painel não descarta dependência temporal longa.

**Budget:** zero updates, no máximo 480 forwards, três modelos sequenciais,
uma GPU. A pontuação central não é uma taxa de codec: se depois houver proposta
de particionamento, transmitir toda a cobertura sem lacunas e cobrar halos,
bordas, escalas e frames. Isso exige work order próprio.

## 5. F2 — Dependência utilizável, sem confundir predição com compressão

### 5.1 População e continuidade

O pool inicial usa treino fora dos 43 checks históricos, excluindo também
componentes ligados a eles por timestamps de bloco iguais ou igualdade de
bloco confirmada. Isso reduz reuso adaptativo, mas não desfaz exposição dos
pesos maduros nem prova que os arquivos restantes sejam independentes.

**Manifesto temporal é um pré-requisito ainda não executado.** Usar cabeçalhos,
clocks e mapas para fixar um bloco por arquivo e regiões retidas contíguas,
com os primeiros 5.000 samples realmente contínuos desse bloco. Registrar
índices absolutos, unidade, origem do clock e intervalo de aquisição.
Não concatenar NPZ espaçados nem atravessar blocos de 1,05 s com overlap.

Ligações conhecidas de fonte/evento, clocks sobrepostos e cópias ficam na mesma
componente fit/check; ausência de tais ligações não certifica sessões.
Especificar o significado dos clocks antes de inferir não sobreposição.
Congelar IDs fit/check por componente e listar configurações sem contraste
possível. Um ID novo ou outro núcleo simultâneo não é uma sessão nova.
O executor entrega esse manifesto **antes** de abrir waveforms; se o desenho
temporal não puder ser certificado, F2 fica bloqueado nesse ponto.

Budget de leitura: no máximo 64 milhões de células originais, um bloco por
arquivo, sem redes nem GPU. Processar os nove ε sobre as mesmas células,
sem reler/alargar a amostra para salvar resultado. Este é um painel diagnóstico
de regiões contínuas, não nova medida da população histórica de NPZ.

### 5.2 Contraste probabilístico a implementar

Fixar quantização/escapes uma vez por precisão. Sequência canônica por região:
tempo externo e canal interno; o decoder conhece os símbolos anteriores.
Reset em cada bloco independente de 1.000 samples, todos os canais retidos
daquela região. Não usar outro núcleo ou região como contexto gratuito.

Controle marginal `p(q)` versus pequeno modelo de tabelas `p(q|c)`:

- `c` descreve sinal e ordem de magnitude do valor quantizado anterior no mesmo
  canal; segundo contexto permitido é o canal físico imediatamente anterior no
  mesmo sample, apenas se contíguo e já decodificado.
- Não ajustar um novo preditor de média por MSE como substituto do contraste.
  A tabela deve poder detectar alteração de escala/distribuição com média zero.
- Fixar buckets no fit por log-magnitude dos inteiros (zero separado, sinal
  separado, expoentes limitados e último bucket de saturação), sem thresholds
  aprendidos no check. Avaliar separadamente temporal e espaço-temporal,
  com os mesmos valores reconstruídos.
- Alphabet finito + ESC, suporte e smoothing congelados no fit; tabela
  marginal compartilhada como backoff de contextos raros. Quantizar CDF para
  inteiros e garantir massa positiva de escape. Transmitir CDF/buckets/config
  e todos os inteiros fora do suporte. Nada fica gratuito por existir em JSON.
- Escolher suporte e limite de contextos **por custo total no fit**, com
  algoritmo determinístico abaixo e teto de memória documentado no preflight do
  coder. Não comparar check antes desse congelamento.
- Executar encoder e decoder independentes; o decoder não recebe raw,
  histogramas do check ou estatísticas futuras. Contexto de um escape vem
  somente da representação já decodificada, conforme regra explícita.

Esse é um controle diagnóstico clássico, não mudança do objetivo de produto
neural. Antes do FEBUS, o executor deve fechar a especificação binária, suporte,
fallback, precisão CDF e custos de finalização em um preflight versionado.
O protocolo atual não finge que essas decisões de implementação já geraram um coder.

**Especificação para evitar uma busca implícita pelo executor:**

- Bucket de cada vizinho: zero; sinal × `min(7, floor(log2(abs(q))))`;
  ou indisponível. São 18 estados: zero, oito positivos, oito negativos e
  indisponível. Valores especiais/escapes físicos sem q representável usam
  indisponível. Temporal tem 18 contextos; espaço-temporal, no máximo 324.
- Suporte `[-K,K] + ESC`, `K ∈ {15,63,255,1023,4095}`.
  Escolher um K por precisão no fit, minimizando **bytes autocontidos do marginal**,
  com desempate pelo menor K. Usar o mesmo K em todos os braços no check.
  Não usar desempenho contextual no check para rever esse suporte.
- Fit de tabelas por contagens e suavização de Jeffreys (0,5/símbolo).
  Contexto com menos de 256 observações aponta para a marginal; nos demais,
  misturar contextual e marginal com peso `n/(n+256)`. Transmitir o mapa de backoff.
- Total CDF `2²⁰`. Para A símbolos e probabilidades p, frequências iniciais
  `1 + floor((2²⁰−A)·p)`; distribuir o restante pelas maiores partes
  fracionárias, desempate por ID de símbolo. Reconstruir a CDF exclusivamente
  dos inteiros transmitidos; backend que requantiza deve expor seu custo exato
  ou ser substituído antes de qualquer medição FEBUS.
- Tabelas inteiras little-endian, compressão zlib nível 6, custo real do blob.
  ESC de alfabeto transmite q int32 exato; escape físico transmite posição
  e bits float32. Distinguir os dois e pagar ambos quando necessários.
  Tabelas marginal e contextual usadas pelo fallback pertencem à instalação.
- Por frame, fallback entre coder contextual, marginal e vetor q int32 bruto
  com mesmos escapes físicos; transmitir o identificador. Cobrar a instalação
  inteira mesmo se todos os frames escolherem fallback.

Distribuições são ajustadas **somente no fit**, sem adaptação no check.
Amostras de check não podem determinar a CDF; escolher um modo por frame no
encoder é permitido porque esse modo e seu custo são transmitidos.
Esta especificação limita a capacidade do probe e pode deixar dependência
por explorar. Seu negativo continuará sem equivaler ao ótimo neural.

Reportar: NLL float, NLL da CDF inteira, bytes reais e diferença entre os três.
Um ganho apenas em NLL ou histograma empírico não autoriza F4.
Conferir q bit-exato, waveform bit-exato entre coders, bound físico, especiais,
resets, tails e portabilidade do decoder. Coder marginal genérico existente
pode servir de sanity check adicional, com framing identificado.

### 5.3 Controles que tornam o diagnóstico interpretável

Cinco controles sintéticos pequenos, mesma quantização, reset e contabilidade:

1. Gaussiano i.i.d.: não deve produzir ganho contextual sistemático fora do fit.
2. AR(1) com correlação conhecida: verificar dependência e degradação com resets.
3. Regime de variância Markov persistente, inovação gaussiana de média zero:
   correlação linear pode ser pequena e a distribuição condicional informativa.
4. Constante/zero: overhead e razão relativa indefinida ficam visíveis.
5. Valores float32 extremos, não finitos e zeros com sinal: escapes exatos.

Seeds fixas antes da execução, fit/check sintéticos independentes. A regra de
aprovação do método não exige “ganho em todo bloco i.i.d.” nem usa sorte como
evidência: publicar diferenças em repetições fixas e incerteza Monte Carlo.
Fracasso em AR/variância condicional torna um negativo FEBUS **inconclusivo para
esse diagnóstico**, não evidência de inexistência de dependência.
Esses controles não aumentam a diversidade de treinamento FEBUS.

Para execução reprodutível: seeds `1701,1702,1703,1704,1705`, com check usando
seed + 100.000; oito canais independentes × 131.072 samples por realização.
AR(1) estacionário com ρ=0,95 e variância marginal 1. No controle de variância,
dois estados equiprováveis, probabilidade de permanência 0,99, desvios 0,25 e 4,
com estado oculto ao coder. Usar as duas precisões primárias com escala
estimada no fit e congelada. Publicar primeiro efeito em NLL/CDF, depois bytes
e instalação: falta de amortização não é falha de detecção de dependência.
Nos casos constante/especiais, o objetivo é integridade e contabilidade.

### 5.4 Regra de decisão líquida

Nas duas precisões primárias predefinidas, exigir simultaneamente:

- Pelo menos **10% de economia em bytes completos** e **0,05 bpc** absolutos
  contra marginal no painel check, com reconstrução idêntica.
- Ganho no agregado de cada uma das quatro famílias e nenhuma configuração
  com mais de 1% de regressão de bytes; publicar todos os casos e overheads.
- Contabilidade explícita de instalação no objeto medido e curvas separadas
  de amortização. Só passar em transporte compartilhado não é vitória autocontida.
- Roundtrip e garantia sem violações; custo de encode/decode e memória medidos.

São thresholds de investimento escolhidos agora, não tolerâncias científicas de
uso. Estabilidade por arquivo/componente continua necessária; não tratar células
como réplicas independentes para fabricar intervalos de confiança estreitos.
Ganho em um único regime ou ε vira oportunidade localizada, não passe generalista.
Melhora nos demais sete pontos não substitui os dois primários.

**Limite da regra:** passar torna F4-A justificável, não vencedor presumido.
Não passar encerra esse probe; não refuta todos os modelos probabilísticos.
Falha por setup em objetos pequenos deve ser separada de falta de dependência.

## 6. F3 — O desenho que falta é de sessões, não de mais janelas

Antes de uma curva de dados, registrar sessões e relações fonte/derivado com
`config/new_febus_acquisitions.template.csv`, preservando simultaneidade de núcleos
e eventos. Uma sessão requer definição operacional de aquisição/reinstalação;
ID inventado por arquivo não satisfaz essa exigência.

Desenho mínimo informativo: repetir mesma montagem/excitação entre sessões
e variar ao menos um fator com repetição suficiente para separá-lo do dia ou
dispositivo. Sem cruzamento/repetição, “efeito de fibra” pode ser efeito de
excitação/montagem. Definir sessão de confirmação antes de inspecionar seu alvo.
Seeds de treino estimam variação do otimizador, não entre sessões.

Publicar dois estimandos distintos: curvas com compute total fixo e curvas com
exposição/convergência por sessão controlada. Arquitetura, geometria, avaliação,
precisão e população ficam fixas. A dose externa não substitui a dose FEBUS.
O exemplo de quatro sessões/família do 106 é planejamento logístico, não cálculo
de poder estatístico; o tamanho deve vir da variabilidade piloto entre sessões.

**Estado:** desenho científico definido em princípios; curva bloqueada pela
ausência das novas sessões. Teste fold0 e EDAS não substituem essa coleta.

## 7. F4 — Como escolher uma única candidata

| F1 | F2 | Próxima decisão científica |
|---|---|---|
| Sensibilidade material de contexto | Sem oportunidade líquida demonstrada | Priorizar um contraste de compatibilidade, se o controle puder ser reproduzido |
| Contexto sem efeito material no painel | Oportunidade líquida consistente | Priorizar distribuição neural pequena de valores quantizados |
| Ambos positivos | Ambos positivos | Comparar benefícios observados e custo; escolher apenas um, sem somar ganhos hipotéticos |
| Ambos negativos válidos | Ambos negativos válidos | Reduzir investimento arquitetural e documentar a fronteira atingível |
| Algum inconclusivo | Qualquer | Resolver a limitação identificada, sem transformar ausência de evidência em refutação |

F4-A deve mudar o modelo de probabilidade, mantendo quantização/cobertura e
controle marginal. Objetivo log-probabilidade, inferência causal reproduzível,
grupos paralelizáveis e coder com probabilidades discretizadas. Fallback deve
cobrar identificação e instalação. Uma prior ruim deve custar bytes, não
inventar a realização observada.

F4-B deve isolar compatibilidade treino/inferência; não trocar simultaneamente
backbone, quantizador, loss, dados e coder. Sensibilidade de uma rede congelada
não garante benefício após treino, e reduzir contexto pode sacrificar dependência.

Proposta de dose: 8.192 updates por braço, extensão até 32.768 só após revisão
de curvas/saúde/custos, com células e compute registrados. Esses números não
são licença para executar agora todos os braços ou uma busca de arquiteturas.
Uma GPU, braços sequenciais. Confirmação científica permanece limitada sem sessões.

## 8. Ordens de execução para o próximo modelo

1. **D0 — metadados:** executar/verificar o congelamento abaixo. Entregar hashes,
   painel e exclusões; não abrir sinais para alterar a seleção.
2. **D1 — runner F1:** implementar o cruzamento da seção 4, recuperar os pesos
   identificados, checar replay e registrar protocolo/hash antes da inferência.
   Entregar tabela longa por arquivo/modelo/contexto/norma e todos os custos.
3. **D2a — manifesto temporal F2:** fechar componentes fit/check, coordenadas,
   overlaps e orçamento da seção 5.1. Ambiguidade física retorna ao responsável
   científico; não inventar timestamps/unidades.
4. **D2b — coder e controles:** implementar seção 5.2, congelar detalhes binários
   e executar sintéticos. Entregar teste de causalidade/roundtrip e bytes, antes
   de qualquer score FEBUS. Seleção científica muda somente por versão explícita.
5. **D2c — curva FEBUS:** após os dois preflights, executar todos os pontos
   declarados, com mesmos dados, sem tuning nos checks.
6. Retornar F1/F2 para decisão da seção 7. F3 e treino F4 não são ações automáticas.

```bash
.venv/bin/python -m tools.freeze_feasibility_metadata \
  --output results/experiments/feasibility_definition_20261117/metadata_freeze.json
.venv/bin/python -m unittest tests.test_feasibility_metrics tests.test_feasibility_context tests.test_feasibility_metadata
```

Não reutilizar supervisores antigos de duas GPUs. Usar somente um dispositivo
remoto e execução sequencial. Preservar relatórios, protocolos, negativos e reservas;
não recriar mapas JSON de numeração/consolidação. O destino acima é novo e
recusa evidência diferente; uma revisão precisa de outro destino/versionamento.

## 9. Evidência e fechamento

Executado localmente:

- [Congelamento de metadados](../results/experiments/feasibility_definition_20261117/metadata_freeze.json),
  com hashes de manifesto, protocolo, proveniência, fontes e relatórios 1–106.
- Painel F1: **20 janelas / 20 arquivos / dez configurações**, duas por grupo.
  As larguras permitem **414 forwards** no cruzamento completo, abaixo do teto
  480. Uma janela de dois canais permanece registrada sem contraste espacial.
- Pool F2: **32 arquivos**, contra 44 ao excluir apenas os checks históricos.
  Outros **12 arquivos**, todos multicore, ficam fora por conexão transitiva a
  checks via timestamps de bloco. São relações de metadados, não cópias provadas.
- Componentes remanescentes: doped/5m SMF tem cinco; cada configuração
  single-mode e multimode/Configuration 7 tem três; multicore/C2 e C3 tem duas
  (quatro arquivos), e multicore/C1 e C4 tem **uma (dois arquivos)**.
- **Consequência:** C1 e C4 não admite fit/check por componentes nesse pool.
  Registrar esse grupo como bloqueado para esse estimando. Não separar seus
  dois arquivos, reaproveitar os checks excluídos ou chamar uma divisão temporal
  de confirmação por arquivo. Os demais grupos podem informar o diagnóstico,
  mas não autorizam um passe sobre as dez configurações. Qualquer pergunta
  intrassessão alternativa requer protocolo próprio e custo de adaptação.
- **25 testes aprovados** com `unittest`: novas métricas, contexto, congelamento,
  e regressão das métricas físicas/espectrais anteriores. O planner foi checado
  em larguras 1–1.024; identidade, apagamento, inversão, mudança de unidade,
  Nyquist e pooling foram verificados em controles sintéticos.
- Hashes congelados conferidos e numeração contínua 1–107. A checagem documental
  encontrou 183 links locais acessíveis; as duas referências históricas do índice
  a `archive/post_89_consolidation_20261115/` não estão disponíveis nesta cópia
  local. As referências foram preservadas, sem recriar o arquivo histórico.

Os módulos de teste usam sinais sintéticos apenas. Nenhum waveform FEBUS foi
lido, nenhum checkpoint carregado para inferência e nenhum treino/GPU remota
ou download foi iniciado. Os resultados históricos e suas decisões continuam
preservados; não houve novo score de codec ou promoção.

**Próxima ação delegável:** D1 (replay F1), em paralelo lógico à preparação de
D2a/D2b; uma única GPU para qualquer execução remota. A interpretação dos
resultados retorna à seção 7. F0 operacional, F3 e seleção de F4 seguem pendentes
das informações/evidências explicitadas, não de maior tamanho de modelo.
