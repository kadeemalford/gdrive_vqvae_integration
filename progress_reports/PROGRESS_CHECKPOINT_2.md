# PROGRESS CHECKPOINT 2 — Fidelidade por bit na reconstrução generalista FEBUS

**Consolidação: 05/11/2026.** Continuidade do
[checkpoint 1](PROGRESS_CHECKPOINT_1.md), cobrindo os **relatórios 60–79**, inclusive
as campanhas e controles posteriores reunidos nas seções 8–13 do relatório 79.
O primeiro checkpoint sintetizava principalmente 1–59, com atualizações sobre
60–64; esses resultados são retomados aqui para fechar a comparação arquitetural,
não contados como descobertas adicionais.

**Resultado geral:** houve avanços relevantes de reconstrução e de codificação,
mas **nenhum novo codec foi promovido a solução generalista operacional**. Não
foi demonstrada reconstrução de alta fidelidade a 700×. Validação cruzada, novos
recortes dos mesmos arquivos e stress externo de desenvolvimento têm alcances
diferentes e não constituem uma única confirmação independente.

## 1. O que mais avançou neste período

1. **Adaptação da rede transferida, sem aumentar a taxa do contraste:** a confirmação
   A/B/C chegou a **0,416764 de RMSE / 0,904839 de correlação**, contra
   0,457670 / 0,889594 do histórico compatível reavaliado, ambos a **40,367×
   nominais**. É redução de RMSE de **8,94%**, embora com regressões localizadas.
2. **Codificação lossless:** reduzir os bytes dos índices e transportar melhor
   mapas/metadados preservou a reconstrução existente. No mesmo container, o
   compacto passou de **153,680× para 171,994×**, e o denso de **41,127× para
   45,499×**. É avanço de compressão, não de fidelidade aprendida.
3. **Busca de índices com decoder congelado (E1):** mostrou oportunidade concreta
   de melhorar a codificação sem aumentar bytes por janela. Ganhos de RMSE de
   **3–5%** apareceram em diferentes pilotos; proteger também amplitude e espectro
   reduziu o ganho, e a restauração afim transmitida recuperou parte dele.
4. **Refinamento residual FSQ:** trouxe **~3% de redução de RMSE com guarda** e
   repetiu o benefício em outra rotação, mas com mais bits. A vantagem adicional
   sobre selecionar o modelo denso a bytes pareados foi muito menor.
5. **Preparação espacial e integridade:** melhorias localizadas de entrada foram
   medidas sobre o mesmo alvo retido, e não finitos passaram a ser preservados
   bit a bit. São avanços de preparação/contrato numérico, não autorização para
   contar exclusão de canais como compressão ou ganho neural.

As demais estratégias ajudaram principalmente a delimitar hipóteses: **mais
parâmetros, arquitetura anisotrópica, mudanças de quantizador, perdas/proxies de
taxa e controles clássicos não produziram o salto generalista pretendido**.
Isso não equivale a provar que essas famílias sejam inviáveis.

## 2. Ganhos relevantes de fidelidade — sempre dentro de cada protocolo

| Estratégia | Referência → resultado | Efeito em bits/cobertura | Limite principal |
|---|---|---|---|
| Adaptação transferida na confirmação A/B/C | RMSE **0,457670 → 0,416764**; correlação **0,889594 → 0,904839** | Mesmos **40,367× nominais**, alvo e avaliação do contraste | Caudas e configurações ainda regridem; relatório 64 |
| Preparação antes da inferência, FEBUS 4× no arquivo de 200 Hz | RMSE centrado **0,685150 → 0,596336**, ~13%; correlação **0,728732 → 0,805477** | Avaliação nos mesmos 505 canais, pesos congelados; muda a entrada | Resultado localizado, não reconstrução de toda a extensão bruta; relatório 61 |
| Residual FSQ com guarda, primeira coorte | RMSE **0,667586 → 0,647110**, ganho **3,07%** | Razão real **171,936× → 136,456×**: paga bits adicionais | Não é melhora à mesma taxa; relatório 71 |
| E1 com guarda predefinida em 63 recortes adicionais | RMSE **0,64547 → 0,62095**, ganho **3,80%** | Frames **97.031 → 96.801 bytes**, teto respeitado por janela | Guarda inicial cobre RMSE/correlação/p90, não toda a fidelidade física; relatório 78 |
| E1 + afim/OLS limitado, outro conjunto de 63 recortes | RMSE **0,645798 → 0,621873**, ganho **3,705%** | Frames **96.618 → 96.390 bytes**, incluindo coeficientes transmitidos | Guardas físicas mantidas, mas transferência externa pequena; relatório 79, seção 9 |

**Esta tabela não é uma curva de um único modelo.** Mudam preparação, coorte,
checkpoint, agregação, tamanho da avaliação e contrato de bits. Os percentuais
referem-se ao par de cada linha; não devem ser somados. Razões nominais não
incluem os mesmos itens que streams completos, e os pesos compartilhados
continuam um custo de implantação a declarar separadamente.

### 2.1 Preparação espacial: benefício real, porém condicionado ao alvo

A auditoria dos [relatórios 60–61](61-PERSISTENT_SPATIAL_BAND_FILTER.md) mostrou
faixas de alta energia muito diferentes entre famílias e arquivos. A política
`persistent-amplitude-v1` trata excesso persistente conservadoramente, mantém
mapas e não concatena regiões não contíguas. O corpus novo foi auditado contra
os brutos no [relatório 62](62-SPATIAL_CORPUS_AND_CONTROLLED_NEURAL_COMPARISON.md).

No contraste de 200 Hz da tabela, as duas reconstruções foram avaliadas nos
**mesmos canais fonte retidos**. Portanto, o ganho não veio apenas de retirar
canais ruins da métrica: a entrada diferente melhorou a saída nesses canais.
O efeito reúne normalização, geometria e contexto de borda; não isola cada causa.
No arquivo de 3.000 Hz, sem exclusão por amplitude, as saídas ficaram idênticas.

Sem mapa físico validado, a máscara permanece **heurística**, não prova de
invalidade de uma faixa. Ganhos no sinal retido não equivalem a compressão ou
preservação de toda a fibra. Essa ressalva tornou-se central no stress SMF.

## 3. Mudança de arquitetura: a anisotrópica foi de fato implementada e testada

Os [relatórios 62–64](64-CROSS_FOLD_CONFIRMATION_RESULTS.md) compararam:

- **A:** CNN residual nativa, inicializada com pesos offshore e adaptada a FEBUS;
- **B:** mesma CNN nativa, inicialização aleatória, treino FEBUS-only;
- **C:** `anisotropic-multiscale-v1`, inicialização aleatória, treino FEBUS-only.

C foi uma mudança estrutural, não apenas um novo nome para a CNN existente:
convoluções espaciais/temporais fatorizadas, redução e reconstrução por eixo,
ramos temporais locais/dilatados, ramo espacial e fusão multiescala. **Não houve
skip contínuo gratuito atravessando a quantização.** O gargalo VQ e as grades
foram controlados; C tinha ~26,3 M parâmetros contra ~28,0 M na nativa.

| Braço | Primeira coorte, denso: RMSE / correlação | Confirmação, denso: RMSE / correlação |
|---|---:|---:|
| **A — transferência** | **0,445187 / 0,888093** | **0,416764 / 0,904839** |
| B — nativa do zero | 0,482705 / 0,867561 | 0,441654 / 0,893037 |
| C — anisotrópica do zero | 0,495379 / 0,859700 | 0,456840 / 0,885001 |

Todos completaram **98.304 updates por braço em cada coorte**. As razões densas
foram 41,577× na primeira e 40,367× na confirmação, iguais entre A/B/C **dentro
de cada coorte**. Na taxa compacta da primeira, A também liderou: RMSE 0,667586,
contra 0,702018 de B e 0,700077 de C, todos a 160,211× nominais.

**Conclusão:** a anisotrópica não superou a transferência em nenhuma das duas
coortes. Foi também mais lenta apesar de ter menos parâmetros: na confirmação,
~6,65 h de épocas contra ~5,01 h de A. Sua pequena vantagem compacta sobre B
não demonstrou superioridade sobre a referência relevante A.

Isso não refuta arquiteturas anisotrópicas em geral: desenho, inicialização,
otimização e oportunidade de convergência limitam o contraste. O pré-treino
offshore também não é compute histórico gratuito ou pareado com o scratch.
Não foi demonstrado que um Transformer adicional resolveria o restante.

**A liderança de A também não promove seus pesos.** Na confirmação, por exemplo,
multicore e single-mode 11 regrediram contra o histórico; o p90 de single-mode 11
foi 1,37775 apesar da correlação agregada 0,95690. Ganho macro e correlação alta
não substituem preservação de amplitude, fase e canais.

## 4. Aumento do tamanho da rede: ganho modesto, não o salto procurado

Os [relatórios 65–66](66-CAPACITY_COMPACT_RESULTS.md) testaram **aumento de largura
da CNN nativa**, em scratch, mantendo o ponto compacto. Não foi uma troca de
arquitetura nem aumento da quantidade de índices.

| Braço | Parâmetros | RMSE macro | Correlação | Compressão nominal |
|---|---:|---:|---:|---:|
| Controle | 27.958.081 | 0,684932 | 0,716521 | 157,357× |
| Maior | 96.517.033 | **0,674562** | **0,724291** | 157,357× |

A rede ~3,45× maior reduziu RMSE em **1,51%**, com o mesmo orçamento final de
98.304 updates por braço. O ganho não foi zero, mas **não mudou o patamar de
fidelidade**, não preservou todos os grupos e não justificou escalar novamente.
O p90 de single-mode 11 piorou **0,198380**; houve cinco regressões nas guardas.

O tempo de épocas foi ~2,44 h contra ~0,94 h, **2,59× maior**, além do custo
dos pesos ampliados. A taxa nominal igual não significa implantação de mesmo
custo. Treino e validação ainda melhoravam; não há base para explicar o retorno
pequeno apenas por overfitting ou declarar convergência definitiva.

**Decisão:** tentativa relevante, porém malsucedida como estratégia de avanço
generalista forte. Não concluir que toda rede grande falharia, nem comparar seus
scores diretamente com A/B/C de outras condições como se fossem uma ablação única.

## 5. Melhor compressão sem degradar a reconstrução já obtida

### 5.1 Índices e metadados lossless: avanço consolidado de codificação

O [relatório 68](68-FROZEN_CODING_RESULTS_AND_TAIL_ROI.md) mediu redução do payload
fixo dos índices em **11,55% no compacto** e **9,81% no denso**, com roundtrip
exato de índices e side, sem perda adicional da reconstrução.

O transporte também importou: mapas/manifesto/política antes ocupavam ~4,68 MB
textuais. Foram preservados em archive lossless, com fronteiras e comprimentos
dos frames cobrados. **No mesmo container**, a comparação ficou:

| Ponto | Índices fixos | Huffman/fallback |
|---|---:|---:|
| Compacto | 153,680× | **171,994×** |
| Denso | 41,127× | **45,499×** |

Não se removeu informação para obter essa economia. Entretanto, melhorar
empacotamento/entropia conserva a **fidelidade que o reconstrutor já tinha**;
não corrige seus erros científicos. No compacto da coorte curta, incluir uma
vez os pesos reduzia a razão de 131,347× do transporte textual para ~19,385×:
a amortização do modelo depende do volume, não pode ser escondida.

### 5.2 Os incrementos posteriores foram pequenos ou não amortizaram o custo

- **Prior contextual neural, relatório 75:** economizou ~2,41% nos frames, mas
  ficou **0,75% maior no total** que o melhor controle ao incluir seus pesos.
  Não confirmou a economia líquida necessária para continuar no segundo fold.
- **Transições/repetições estáticas, relatórios 76–77:** economia adicional de
  **0,086–0,308%** contra marginal/range nos quatro pontos compacto/denso dos
  dois folds, preservando os códigos. Útil, mas sem mudança de patamar.
- **Escapes numéricos, relatório 79:** no stress externo, o side real ficou em
  **18.720 bytes**, contra 5.769.600 bytes de escapes sem essa compressão.
  Os **720.000 NaNs** foram preservados bit a bit. É integridade e serialização
  lossless, não recuperação neural de sinal nem redução de erro ordinário.

## 6. Como a fidelidade adicional foi obtida — e o que ela custou

### 6.1 FSQ residual e seleção de modos

O residual VQ quase não avançou: **0,0447% de RMSE sem guarda**, terminando com
três códigos usados. O FSQ residual, com 2.048 estados, foi mais eficaz nesse
refiner: **3,65% sem guarda / 3,07% com guarda** contra a base. O benefício
repetiu na outra rotação: RMSE **0,648189 → 0,633530** com guarda, mas a razão
real caiu de **164,997× para 134,243×** ([relatórios 70–73](73-RESIDUAL_CONFIRMATION_AND_WIRE_MATCHED_RESULTS.md)).

Mais estados usados também significaram mais informação transmitida; não foi
uma vitória automática de FSQ a bytes iguais. No controle final com **streams
quase pareados**, o híbrido melhorou apenas **0,023–0,356% de RMSE** frente à
seleção densa, dependendo de fold/teto, com regressão de cauda no primeiro fold
e possibilidade de exigir três modelos. A ROI inicial também melhorou a base
ao gastar mais bits, mas sua vantagem sobre alocação uniforme foi pequena.

**Síntese:** houve ganho de reconstrução comprando refinamento, e alguma
complementaridade, mas não uma nova fronteira generalista forte a custo igual.

### 6.2 E1: melhor escolha de índices com o mesmo decoder

E1 otimiza vetores apenas como auxílio de busca, projeta-os ao codebook e
transmite **índices discretos exatos**. Encoder/decoder/codebook não são
retreinados. O [relatório 78](78-REMOTE_REPRESENTATION_PILOTS.md) encontrou
~5% de redução de RMSE a teto real em duas coortes; parte desse ganho piorava
p90 por canal. Com guarda predefinida, novos recortes deram **3,80%**.

A guarda aplicada **durante** a busca, e não somente no final, recuperou outros
candidatos admissíveis: **4,014%** em 63 recortes adicionais no relatório 79.
Porém, ao proteger também magnitude FFT, fase e distância da razão RMS a 1,
um contraste pareado passou de **4,831% para apenas 1,299%** de ganho.
As guardas iniciais não asseguravam preservação física abrangente.

A restauração afim explicitamente transmitida recuperou parte dessa margem:
em 63 recortes, **2,971% contra origem**, ou **1,895% sobre o controle físico
sem afim**. Em outro conjunto pareado, acrescentar OLS limitado chegou a
**3,705% contra origem**, **0,654% adicional sobre as duas propostas afins**,
com sete guardas mantidas e frames não maiores. Não comparar os dois conjuntos
como se o refinamento explicasse toda a diferença entre seus scores.

Esse é um avanço de **codificação**, não um novo reconstrutor treinado. A busca
custou dezenas a centenas de segundos para pequenos lotes, conforme o protocolo;
isso não demonstra tempo real. A guarda evita regressão por construção, mas a
evidência é a magnitude/repetição do ganho, não apenas sua ausência de regressões.
Ainda faltam transferência forte e custo aceitável para justificar destilação
ou implantação ampla ([relatório 79](79-ENCODER_RATE_AND_TILE_WEIGHTING.md)).

## 7. Transferência externa e cobertura mudaram a interpretação dos ganhos

O relatório 79 avaliou seis trechos SMF contínuos de um segundo, **todos os
551 canais**, sem remover a faixa dominante e preservando os NaNs exatos:

| Modelo congelado | RMSE pooled por arquivo de 1 s | Razão real de stream |
|---|---:|---:|
| Compacto | 0,8323–0,8447 | 199,22× |
| Denso | 0,5196–0,5301 | 52,05× |

São taxas diferentes, com pesos compartilhados separados. O denso ser mais fiel
não demonstra vantagem a bytes iguais. As métricas pooled de um segundo não
são intercambiáveis com médias macro de tiles espaçados de desenvolvimento.

A restauração/busca afim transferiu apenas **~0,19%** nos subsets externos
compacto e denso; são recortes/agregações próprios, não uma confirmação pooled
de um segundo. O ganho interno de 3,70% não se generalizou nessa magnitude.

Re-encodificar as mesmas linhas que a heurística reteria reduziu RMSE em
**7,82–9,85% no compacto / 9,18–16,59% no denso** sobre o alvo comum retido.
Mas o particionamento que **entrega todas as linhas** não passou as guardas:
no denso, 5,39% menos RMSE veio com 5,15% mais bytes de bundles e RMS pior.
Melhorar o alvo filtrado não resolveu o produto de cobertura integral.

Todo o grupo SMF de 36 arquivos é agora **desenvolvimento externo**, inclusive
os arquivos restantes. Os dois grupos EDAS continuam sem avaliação de sinais
nesta campanha. Hashes/clocks e pastas distintas não provaram independência
física; esses dados não validam externamente as quatro famílias originais.

## 8. Demais hipóteses e controles — resultados úteis, menor prioridade

| Linha testada | Resultado e interpretação |
|---|---|
| Quantização escalar + prior, taxa–distorção conjunta | Seis braços no relatório 78 mostraram negociação real de bits/erro, mas nenhum superou os controles VQ reais; houve colapso em uma condição. Inicialização funcional evitou aquele colapso, não demonstrou superioridade geral. |
| Encoder com gradiente de taxa, temperatura e amostragem por células | Na avaliação completa do relatório 79, Rate-sharp ganhou só **0,115% de RMSE sobre o controle**; ponderação por células, **0,0184%**. Reduzir a proxy suave nem sempre reduziu índices duros/bytes. |
| Coadaptação full-VQ com taxa | No piloto curto, **0,0547% sobre o controle** e 0,906% menos bytes de frames; não mudou o patamar nem protegeu automaticamente espectro. |
| Corretor espacial, orientação/padding e contexto temporal | Corretor espacial: **0,00679%** sobre o pontual; padding: até **0,0548%** contra origem. Lags AR adicionais de 100/200 ms não sustentaram uma campanha temporal. Negativos limitados aos métodos testados, não prova de independência do sinal. |
| Grade × cardinalidade | Denso K64 usou frames quase iguais ao compacto K2048, mas teve **5,544% mais RMSE**. Alfabetos menores economizaram bytes perdendo fidelidade; adaptação curta ainda em melhora não prova convergência. |
| Produto/PCA e ajuste de centroides pelo decoder | PCA reduziu erro latente abaixo do VQ em um contraste, sem melhorar o waveform. Ajuste pela reconstrução ganhou só **0,00448% de distorção** sobre o controle latente; nenhuma troca admitida. |
| DCT e preditivo com erro limitado | Não venceram os tetos neurais testados. O preditivo obteve alta precisão a **2,5–4,5×**, com muito mais bits; não é solução a alta compressão nem limite inferior teórico. |
| Fatores espaciais/temporais de baixo posto transmitidos | Uma troca local reduziu RMSE **0,259 → 0,146**, com menos bytes, mas o ganho agregado guardado foi apenas **0,174% de distorção**. Busca de postos não ampliou o ganho; variante espectral não passou seu diagnóstico de oportunidade. |

As últimas linhas estão nas seções 8–13 do relatório 79. Vários probes finais
foram interrompidos **ainda no treino**, sem abrir validação, porque falharam
nos critérios de avanço. Percentuais de distorção normalizada MSE/L1 não são
percentuais de RMSE; casos isolados não são ganhos generalistas.

## 9. Estado consolidado da pesquisa

- **Referências preservadas:** controles VQ transferidos nas taxas compacta e
  densa; streams lossless eficientes; E1/afim como opção experimental de encoder;
  residual FSQ e controles clássicos como pontos de comparação, não produto final.
- **Ramos sem retorno suficiente para continuação automática:** mais largura,
  anisotrópica testada, corretores locais, priors/quantizadores específicos e
  refinamentos cuja vantagem desaparece ou fica pequena ao parear bytes/guardas.
- **Questões centrais restantes:** eficiência da informação efetivamente
  transmitida; compromisso físico entre erro global, canais fracos, RMS, fase e
  espectro; cobertura espacial válida; custo de codificação e amortização dos
  pesos; confirmação com proveniência independente.
- **Condição de nova aposta:** mecanismo distinto e comparação controlada de
  fidelidade/bytes, não repetir um negativo com mais passos ou mais bits. Falha
  de um piloto não prova impossibilidade da família nem fornece limite informacional.

Houve ganho não desprezível neste período, mas ele se divide entre **melhor
adaptação à mesma taxa**, **mais fidelidade pagando bits**, **menos bytes sem
mudar a reconstrução** e **melhor preparação de um alvo explicitamente definido**.
Confundir essas quatro categorias produz uma impressão de progresso maior do
que o efetivamente demonstrado. A meta continua sendo eficiência de representação
com fidelidade física, não somente uma média melhor ou uma razão elevada.

Protocolos, snapshots, dados identificados por hash, frames reais, escapes exatos
e readers sem o waveform original sustentam a reprodutibilidade; a suíte chegou
a **331 testes locais aprovados**. Isso é verificação de implementação, não
validação operacional. Este checkpoint é síntese documental, sem nova execução.

## Mapa dos relatórios cobertos

| Bloco | Evidência principal |
|---|---|
| [60](60-RANDOM_DRIVE_SPATIAL_ENERGY_BANDS.md)–[64](64-CROSS_FOLD_CONFIRMATION_RESULTS.md) | Seleção/cobertura espacial, corpus auditado, implementação anisotrópica, comparação A/B/C e confirmação. |
| [65](65-CAPACITY_COMPACT_SINGLE_GPU_PROTOCOL.md)–[66](66-CAPACITY_COMPACT_RESULTS.md) | Expansão da CNN nativa até ~96,5 M parâmetros a taxa compacta fixa. |
| [67](67-FROZEN_SYMBOL_CODING_AND_ROI_PROTOCOL.md)–[68](68-FROZEN_CODING_RESULTS_AND_TAIL_ROI.md) | Índices lossless, transporte e limites da alocação ROI. |
| [69](69-PROGRESSIVE_RESIDUAL_REFINEMENT_PROTOCOL.md)–[73](73-RESIDUAL_CONFIRMATION_AND_WIRE_MATCHED_RESULTS.md) | Residual VQ/FSQ, modos concorrentes, confirmação e controle de bytes reais. |
| [74](74-LEARNED_LOSSLESS_INDEX_PRIOR_PROTOCOL.md)–[76](76-STATIC_INDEX_TRANSITION_PROTOCOL.md) | Prior contextual neural e transições lossless; conclusão das transições consolidada no 77. |
| [77](77-RESEARCH_DIRECTION_REVIEW.md)–[78](78-REMOTE_REPRESENTATION_PILOTS.md) | Revisão da direção, diagnósticos e pilotos E1/E2. |
| [79](79-ENCODER_RATE_AND_TILE_WEIGHTING.md), seções 1–13 | Encoder/taxa/tiling, limites físicos, afim, stress SMF, grade/alfabeto, produto/PCA, preditivo e fatores de baixo posto. |

O [roadmap generalista](FEBUS_GENERALIST_ROADMAP.md) mantém as prioridades; os
relatórios e seus artefatos, não esta síntese, são a referência dos protocolos
e de cada versão efetivamente avaliada.
