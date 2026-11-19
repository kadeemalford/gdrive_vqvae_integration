# H1 — lifting, coder e alocação por subbanda

**Etapa consolidada dos registros 92, 93, 94, 95.** Quatro braços concluídos; economia lossless do coder não promoveu H1 sobre o VQ.

Protocolos, resultados e auditorias foram reunidos sem alterar resultados, budgets ou gates. As partes históricas seguem em ordem: estados de lançamento/execução descrevem aquele momento, não tarefas ainda pendentes. Consulte as últimas partes para o encerramento.

## Registros desta etapa

- [Registro 92](#report-92) — Learned lifting: contraste estrutural controlado e protocolo remoto
- [Registro 93](#report-93) — Lifting fixo versus aprendido: resultados iniciais em bytes reais
- [Registro 94](#report-94) — Lifting: ponto compacto concluído e comparação ampliada
- [Registro 95](#report-95) — Taxa hard, coder de símbolos e alocação congelada por subbanda

<a id="report-92"></a>

## Registro histórico 92 — Learned lifting: contraste estrutural controlado e protocolo remoto

Fonte original preservada no arquivo: `92-LEARNED_LIFTING_CONTROLLED_PROTOCOL.md`.

<a id="report-92-learned-lifting-contraste-estrutural-controlado-e-protocolo-remoto"></a>

**Registrado antes da campanha científica; 15/11/2026 UTC.**
Continuação de P0/P1 do [relatório 91](91-PROSPECTIVE_PREPARATION_AND_SPATIAL_OPPORTUNITY.md),
autorizada pelo usuário para até duas GPUs remotas. Não constitui promoção.

<a id="report-92-1-contraste-que-será-isolado"></a>

#### 1. Contraste que será isolado

Haar/lifting fixo versus a **mesma transformação inicial com resíduos locais
neurais de predição e atualização espaciais aprendidos**.

- Cinco níveis espaciais quando o shape permite; três níveis temporais fixos
  na última banda baixa. **Todos os detalhes são preservados e codificados**.
- Operadores espaciais locais com duas convoluções 3×5, oito features, SiLU
  e saída `0,25*tanh`, última convolução inicializada em zero.
- Não há GroupNorm, downsampling destrutivo, filtros de limpeza ou descarte de
  altas frequências. A redução de resolução é uma decomposição completa.
- Comprimentos ímpares: o endpoint sem par fica na banda baixa, sem inventar
  canal físico. Identidade contínua, contagem de coeficientes e bordas testadas.
- Quantização escalar por banda, forward arredondado real e gradiente STE;
  passo aprendível. Prior logístico fatorizado só para otimização.
- Coder real inicial: planos int32/zlib, idêntico entre braços. Diferença entre
  proxy diferenciável e bytes reais permanece explicitamente medida.
- Modo literal comum para constantes bit a bit: valor float32 e shape, com
  framing cobrado, preservando também o sinal de zero.

O objetivo inicial é localizar oportunidade de aprendizagem e calibrar taxa real,
não demonstrar superioridade imediata sobre a CNN com pré-treino OOI.

<a id="report-92-2-dados-e-otimização-congelados"></a>

#### 2. Dados e otimização congelados

Preparação fold 0 já auditada, sem novo tratamento do waveform. Excluir do treino
os 43 arquivos de checagem P1; usar todas as janelas dos demais arquivos de treino.
As checagens de aprendizagem usam um arquivo/janela determinístico por grupo,
dez grupos/quatro famílias. São checks internos, **não validação independente**.
Escalas prospectivas congeladas em P0; nenhuma estatística aprendida na validação.

Mesma seed 2111, mesmas janelas/ordem nos braços, batch de uma janela nativa,
sem recorte artificial para igualar shapes. Reportar células e tempo: updates
iguais não implicam FLOPs iguais. Os braços partem da mesma função Haar, não de
pesos VQ. O controle fixo aprende somente quantização/prior.

AdamW novo, LR constante `3e-4`, weight decay `1e-5`, clip global 1, FP32, TF32
desligado, algoritmos determinísticos PyTorch/cuDNN e
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Loss: MSE normalizada na máscara ordinária + lambda × taxa logística.
Lambda inicia em 0,1; a cada 32 updates:
`lambda *= exp(clip(0.02*(proxy_médio/alvo-1), -1, 1))`, limitada a `[1e-5,100]`.
O contrato mira inicialmente 0,8 e 0,2 bit/célula **no proxy**; só os frames
determinam a taxa real. Não declarar taxa pareada por igualdade de lambda/alvo.

Seleção interna predefinida:
`NRMSE*_macro_famílias² + 0,1 * bpc_stream/alvo`, incluindo inicialização.
É um critério de desenvolvimento, não o gate final da fronteira.
Setup comprimido de mapas/offsets dos checks entra em `stream_bpc`; pesos ficam
discriminados e sua amortização deve ser apresentada antes de qualquer promoção.

<a id="report-92-3-orçamento-e-decisões"></a>

#### 3. Orçamento e decisões

1. Smoke/integridade, separado de resultados científicos: 32 updates por braço.
2. Primeiro contraste científico em 0,8 bpc proxy:
   **32.768 updates por braço**, checagem a cada 2.048.
3. Segundo ponto de 0,2 bpc: mesmo orçamento e inicialização nova, após conferir
   saúde/estabilidade do primeiro ponto. Não começar automaticamente se o primeiro
   tiver falha de integração, NaNs ou treinamento interrompido.
4. Nenhuma extensão automática até 98.304. Preservar curva, endpoints,
   otimizador e RNG. Resume rejeita mudança de protocolo/fontes.
5. P3 só se houver candidato promissor na comparação **a bytes completos
   pareados**, quatro famílias e referência prática VQ. Seed/rotação adicionais
   não são disparadas apenas porque terminou o budget.

Teto planejado: 131.072 updates entre dois braços e dois pontos, fora smoke.
O segundo ponto permanece condicionado à inspeção do primeiro.
Guarda final continua a recomendação do relatório 90: ≥5% menos erro sem mais
bytes ou ≥10% menos bytes a fidelidade equivalente, com guardas por configuração.
Correlação, fase/bandas, amplitude, caudas/canais fracos e custo de execução
precisam acompanhar a avaliação completa; o score interno não os substitui.

<a id="report-92-4-implementação-e-integridade"></a>

#### 4. Implementação e integridade

- `finetuning/learned_lifting.py`: análise/síntese e objetivo escalar.
- `compression/lifting_stream.py`: frames, identidade do modelo, exportação e
  carregamento de receptor independente, constantes literais.
- `finetuning/run_lifting_study.py`: protocolo, treinamento, checagens físicas,
  frames reais, checkpoints/RNG e seleção.
- `tests/test_learned_lifting.py`: inversa, paridade, coeficientes, gradientes,
  função inicial comum, receptor separado e rejeição de pesos incompatíveis.

O receptor do controle fixo não exporta convoluções inativas. O export aprendido
inclui seus operadores. Quantizador/prior ainda estão no export; não é uma
minimização certificada de todos os parâmetros estritamente necessários.

Cada avaliação salva receptor, frames e métricas; o treino mantém `last.pt`
atômico, melhor receptor e endpoint. Checks de replay usam receptor carregado
separadamente, não features originais ocultas. Os pesos compartilhados são
identificados por SHA-256 dentro dos frames.

<a id="report-92-5-máquina-remota-e-sincronização"></a>

#### 5. Máquina remota e sincronização

Conexão: `murilo@aumo-gpu-001` (hostname atual `aumo-gpu-TRX50-WS`).
Cópia existente: `/home/murilo/source/das_febus/gdrive_vqvae_integration`.
Duas RTX 4090/24 GB, livres no preflight.

`tools/sync_research_remote.py` compara conteúdo, preserva **todo arquivo remoto
divergente antes de substituí-lo**, transfere o snapshot e confere SHA-256.
Backups e inventários ficam em `results/remote_setup/pre_p90_sync_*`.
Não há reset, clean, exclusão de arquivos remotos, commit ou push. O HEAD remoto
permanece histórico; o estado executável é o snapshot auditado, não uma falsa
alegação de que a árvore remota ficou limpa ou igual ao HEAD.
Credenciais SSH ficam exclusivamente na variável de ambiente já disponibilizada.

O primeiro smoke completou 32 updates em cada GPU, com replay/inversa finitos:
GPU 0 fixo (~0,60 s de loop) e GPU 1 aprendido (~0,89 s). Esses tempos curtos
não incluem toda a inicialização nem são benchmark de produção. Scores desse
smoke não são evidência de ganho científico. O runner ganhou persistência de
frames/setup completo após esse smoke; uma integração final precede o lançamento.

**Estado da campanha científica:** primeiro ponto lançado conforme registro
abaixo; não confundir smoke com os 32.768 updates.

<a id="report-92-integração-final"></a>

##### Integração final

A primeira comparação treino contínuo versus retomado detectou diferença máxima
de ~8,75e-8 nos parâmetros, sem NaN ou falha de replay. A execução foi preservada
em `lifting_integration_v2_fold0_20261114`; não passou o teste exigente de identidade
bit a bit. Ativamos os algoritmos determinísticos antes de qualquer campanha.

A integração versão 3 **passou** em ambos os braços: 64 updates contínuos versus
32 + resume até 64 produziram **diferença exatamente zero** nos parâmetros,
mesma lambda e mesmo RNG de seleção de janelas. Auditoria:
`results/experiments/lifting_integration_v3_fold0_20261114/INTEGRATION.json`.
Também foram conferidos por SHA-256 **todos os 11.760 NPZ de treino remotos**,
o manifesto e os mapas contra a auditoria local. Nenhum array reservado foi aberto.

O launcher `tools/run_lifting_pair.py` recusa GPUs com processos de compute,
registra PIDs/comandos, usa uma GPU por braço e não dispara outro ponto ou extensão.
O primeiro destino científico é
`results/experiments/lifting_p2_dense_fold0_20261114/{fixed,learned}/`.
O ponto compacto e a confirmação continuam condicionados, não lançados.

<a id="report-92-lançamento-confirmado"></a>

##### Lançamento confirmado

**15/11/2026, 01:36:53 UTC:** iniciado o primeiro contraste, alvo proxy 0,8 bpc,
**32.768 updates por braço / 65.536 no total**, sem extensão automática:

| Braço | GPU física | PID inicial | Diretório |
|---|---:|---:|---|
| Fixo | 0 | 592679 | `lifting_p2_dense_fold0_20261114/fixed/` |
| Aprendido | 1 | 592680 | `lifting_p2_dense_fold0_20261114/learned/` |

Ambos os protocolos e avaliações de inicialização foram conferidos e copiados
localmente. O lançamento confirmou uso das duas GPUs (~671/689 MiB no registro
inicial). `LAUNCH.json` registra comandos e PIDs; os logs ficam na raiz da
campanha. O supervisor grava `EXIT.json` e `PAIR_BUDGET_COMPLETE` somente ao
encerrar os dois processos com sucesso. No lançamento **não existem resultados
convergidos nem modelo promovido**.

A suíte local passou **393 testes**; as verificações remotas de identidade,
hashes e resume são adicionais. As execuções rodam sobre o snapshot sincronizado
e verificado, preservando divergências remotas em backups; não houve commit/push.

<a id="report-92-primeiro-orçamento-concluído"></a>

##### Primeiro orçamento concluído

Os dois braços do primeiro ponto completaram 32.768 updates, com código de saída
zero e replay exato em todas as checagens. Os resultados, limites de taxa e
lançamento condicionado do segundo ponto estão no
[relatório 93](92-LEARNED_LIFTING_STUDY.md#report-93).
O registro anterior descreve o lançamento, não uma campanha ainda inconclusa.

Os quatro braços previstos posteriormente completaram **131.072 updates
científicos** no total, sem extensão. O estado consolidado está no
[relatório 94](92-LEARNED_LIFTING_STUDY.md#report-94).

---

<a id="report-93"></a>

## Registro histórico 93 — Lifting fixo versus aprendido: resultados iniciais em bytes reais

Fonte original preservada no arquivo: `93-LEARNED_LIFTING_INITIAL_RD_RESULTS.md`.

<a id="report-93-lifting-fixo-versus-aprendido-resultados-iniciais-em-bytes-reais"></a>

**15/11/2026 UTC.** Continuação do [protocolo 92](92-LEARNED_LIFTING_STUDY.md#report-92).
Resultados de desenvolvimento, sem promoção e sem leitura de validação/teste.

<a id="report-93-1-primeiro-ponto-concluído"></a>

#### 1. Primeiro ponto concluído

Ambos os braços completaram **32.768 updates**, com 5.600 janelas de 44 arquivos
e mesma ordem de treinamento. Cada braço processou **2.531.708.000 células**.
Os 43 arquivos de checks P1 foram excluídos dos batches novos; os checks internos
do treinamento usam dez janelas/arquivos, um por configuração.

| Candidato | Update | NRMSE* macro entre famílias ↓ | Stream bit/célula ↓ |
|---|---:|---:|---:|
| Inicialização Haar comum | 0 | 1,022112 | 0,542984 |
| Fixo selecionado | 4.096 | 0,729375 | 1,299832 |
| Aprendido selecionado | 4.096 | 0,715812 | 1,300161 |
| Fixo endpoint | 32.768 | 0,791875 | 1,103377 |
| Aprendido endpoint | 32.768 | 0,760954 | 1,121030 |

A seleção segue o critério predefinido, incluindo bytes: não foi alterada para
favorecer um braço. O endpoint e o selecionado são reportados separadamente.
O aprendido selecionado reduz erro em **~1,86%**, mas custa 47 bytes a mais
nesse stream de checks. No endpoint, reduz **~3,90%** com **~1,60% mais bytes**.
Esses números **não demonstram o gate ≥5% sem aumento de bytes**.

O score selecionado não melhora todas as famílias:

| Família | Fixo selecionado | Aprendido selecionado |
|---|---:|---:|
| Doped | 0,831240 | 0,815761 |
| Multicore | 0,579353 | 0,586803 |
| Multimode | 0,700676 | 0,683285 |
| Single-mode | 0,806230 | 0,777397 |

Há regressão em multicore; não declarar ganho generalista. São métricas
prospectivas novas, não diretamente intercambiáveis com macros históricas.

<a id="report-93-2-saúde-taxa-e-custo"></a>

#### 2. Saúde, taxa e custo

O alvo **0,8 bpc era do proxy logístico**, não um limite garantido do stream.
No último bloco, proxy médio: **0,8072 / 0,8135 bpc** (fixo/aprendido).
Os streams reais de endpoint ficaram em **1,1034 / 1,1210 bpc** nos checks:
existe descalibração coder/proxy e a distribuição dos checks também não é
idêntica à média dos batches. Não atribuir toda a diferença a um único fator.

Nenhum NaN/overflow interrompeu o treinamento. O máximo de diferença absoluta
entre caminho hard e replay do receptor separado foi **zero em todas as
17 checagens de ambos os braços**.

Tempo de loop até o endpoint: **~306,7 s fixo / ~525,6 s aprendido**, incluindo
checagens dentro do loop, mas não toda inicialização/exportação. Não é benchmark
de latência encoder/decoder. Exportações compartilhadas:
**2.021 B fixo / 22.757 B aprendido**. O stream exclui esses pesos compartilhados,
mas inclui o setup dos checks; amortização dos pesos ainda precisa ser lida
por horizonte. Arquivos científicos e otimizador não são payload de transporte.

Artefatos locais e remotos:
`results/experiments/lifting_p2_dense_fold0_20261114/`.
Incluem protocolos/hashes, históricos, todos os checks, receptores, frames,
estados de retomada, logs, `EXIT.json` e `PAIR_BUDGET_COMPLETE`.
Os dois processos terminaram com código zero; os artefatos foram recuperados.

<a id="report-93-3-segundo-ponto-autorizado-e-lançado"></a>

#### 3. Segundo ponto autorizado e lançado

A inspeção do primeiro ponto confirmou aprendizagem, estabilidade e replay, sem
justificar promoção ou extensão. Prosseguimos com o **segundo ponto já previsto**,
não com mais updates dos modelos densos:

- alvo proxy **0,2 bpc**, nova inicialização, mesma seed/dados/otimização;
- **32.768 updates por braço**, sem extensão automática;
- lançamento **15/11, 01:46:44 UTC**;
- GPU 0 fixo, PID inicial 609114; GPU 1 aprendido, PID inicial 609115;
- `results/experiments/lifting_p2_compact_fold0_20261114/`.

Ao registrar esta seção, o ponto compacto está em execução, não concluído.
O teto da campanha permanece 131.072 updates científicos entre os quatro braços.

<a id="report-93-4-avaliação-seguinte-sem-outro-treinamento"></a>

#### 4. Avaliação seguinte, sem outro treinamento

Antes de qualquer confirmação P3:

1. Congelar o checkpoint escolhido pelo critério original de dez checks.
2. Avaliar os **129 checks/43 arquivos de treino** já separados em P1.
3. Medir curva hard com multiplicadores de passo
   `0,5 / 0,75 / 1 / 1,5 / 2 / 3 / 4`, sem retreino, mesmo coder/target.
4. Comparar apenas pontos discretos medidos; não certificar economia por
   interpolação, mesmo passo, mesmo proxy ou mesmo lambda.
5. Incluir referências VQ maduras compatíveis e cobrar suas tabelas/setup/pesos.

Essa calibração de taxa é **posterior aos primeiros resultados**, não uma
validação independente nem um gate prospectivo novo. As configurações de checks
continuam pertencendo ao treino, e o viés para tiles maiores permanece.
As referências VQ já viram esses arquivos no histórico e tiveram outra exposição.

Implementação adicionada: `finetuning/evaluate_lifting_curve.py` e
`finetuning/evaluate_prospective_vq_reference.py`. Ainda não há, nesta seção,
resultado dessa avaliação ampliada ou aceitação por guardas espectrais/físicas.
Ela também registra energia/erro por bandas de 0–1/1–5/5–10 kHz, amplitude
espectral e fase com cobertura explícita. São descritores Hann centrados,
com estatísticas agregadas por arquivo; não substituem tolerâncias operacionais
nem são equivalentes a todos os seletores espectrais históricos.
`finetuning/summarize_lifting_frontier.py` compara apenas pontos discretos a
bytes não maiores e conserva vetos por configuração; não promove automaticamente.

<a id="report-93-continuação"></a>

##### Continuação

O ponto compacto terminou; seus resultados e a avaliação ampliada estão no
[relatório 94](92-LEARNED_LIFTING_STUDY.md#report-94).
A seção 3 acima preserva o estado no lançamento, não indica treinamento pendente.

---

<a id="report-94"></a>

## Registro histórico 94 — Lifting: ponto compacto concluído e comparação ampliada

Fonte original preservada no arquivo: `94-LEARNED_LIFTING_COMPACT_AND_POSTFLIGHT.md`.

<a id="report-94-lifting-ponto-compacto-concluído-e-comparação-ampliada"></a>

**15/11/2026 UTC.** Continuação dos relatórios
[92](92-LEARNED_LIFTING_STUDY.md#report-92) e
[93](92-LEARNED_LIFTING_STUDY.md#report-93).

<a id="report-94-1-orçamento-científico-concluído"></a>

#### 1. Orçamento científico concluído

Os quatro braços concluíram **32.768 updates cada**, totalizando **131.072
updates científicos**, sem extensão. Cada braço recebeu a mesma sequência de
2.531.708.000 células. O total não inclui os smokes/testes de retomada.
Brutos e checkpoints históricos foram preservados; não houve inferência nos
conjuntos reservados.

O ponto compacto também terminou com código zero nos dois processos.
Todas as 17 checagens por braço tiveram diferença **zero** entre forward hard
e replay do receptor independente.
Uma auditoria local adicional conferiu os fontes congelados, os 1.024 registros
de histórico por braço e a identidade dos pesos selecionados com `best_receiver.pt`:
`results/experiments/lifting_training_budget_audit_20261115.json`.

<a id="report-94-2-resultado-do-alvo-proxy-02-bpc"></a>

#### 2. Resultado do alvo proxy 0,2 bpc

| Candidato | Update | NRMSE* macro entre famílias ↓ | Stream bpc ↓ |
|---|---:|---:|---:|
| Fixo selecionado | 0 | 1,022112 | 0,542984 |
| Aprendido selecionado | 0 | 1,022112 | 0,542984 |
| Fixo endpoint | 32.768 | 1,170653 | 0,367902 |
| Aprendido endpoint | 32.768 | 1,130765 | 0,372827 |

**O seletor predefinido conservou a inicialização nos dois braços.** Não escolher
outro checkpoint retrospectivamente para ocultar esse resultado. Os endpoints
permanecem disponíveis, mas não são os modelos selecionados.

Ao endpoint, o aprendido tem ~3,41% menos erro que o fixo, com ~1,34% mais bytes.
Ambos pioraram no score prospectivo em relação à inicialização, enquanto
diminuíram o stream. Isso caracteriza o trade-off encontrado nesta otimização;
não prova colapso neural ou impossibilidade do lifting.

| Família, endpoint compacto | Fixo | Aprendido |
|---|---:|---:|
| Doped | 1,168239 | 1,151670 |
| Multicore | 0,980345 | 0,980515 |
| Multimode | 0,991371 | 1,001181 |
| Single-mode | 1,542656 | 1,389695 |

Há regressões em multicore/multimode. O gate generalista não está demonstrado.
Último bloco de treino: proxy médio **0,19990 / 0,20204 bpc**,
tempo de loop **~307,3 / ~520,6 s**. Os streams reais dos checks ficaram em
~0,368/~0,373 bpc, portanto **atingir o proxy não atingiu 0,2 bpc de transporte**.

Artefatos: `results/experiments/lifting_p2_compact_fold0_20261114/`,
copiados integralmente do remoto. O orçamento termina aqui: não foi disparada
extensão para 98.304 nem nova seed.

<a id="report-94-3-postflight-de-inferência-separado-do-treino"></a>

#### 3. Postflight de inferência, separado do treino

Lançado após os dois orçamentos completos, com GPUs livres:
`tools/run_lifting_postflight.py`, diretório
`results/experiments/lifting_postflight_fold0_20261115/`.
Nenhum parâmetro recebe updates nessa etapa.

O plano congelou os selecionados do critério original e mede:

- 129 janelas de 43 arquivos de treino, os mesmos checks ampliados de P1;
- quatro selecionados × sete multiplicadores de passo, com frames reais;
- duas referências VQ maduras compatíveis com fold 0, coder marginal histórico
  ajustado apenas nos 132 recortes dos 44 arquivos de ajuste;
- RMSE*/caudas/correlação por canal e arquivo, bandas/amplitude/fase com cobertura;
- setup/tabelas e pesos compartilhados discriminados;
- comparação discreta a bytes não maiores, sem interpolação.

As referências VQ têm pré-treino/exposição distintos e historicamente já viram
os checks. A referência compacta é a exportação da época 24;
a densa vem de `spatial_abc_20261024_v2/high_A/best_macro.pt`, **época 23**,
com config/pesos exportados sem alteração e sem otimizador.
O arquivo de proveniência está em
`results/remote_setup/dense_transfer_fold0_weights_p93.pt.json`.
Não confundir época selecionada com a duração total da campanha histórica.

**Limites:** avaliação ampliada de desenvolvimento, não confirmação externa;
tiles maiores sobrerrepresentados; a curva de passos foi proposta depois do
primeiro resultado. O export VQ contabilizado contém também encoder e não é
um certificado do menor receptor possível.

O postflight concluiu com código zero, **sem updates de parâmetros**. Os
artefatos foram recuperados integralmente para a máquina local.

<a id="report-94-4-curvas-medidas-nos-129-checks"></a>

#### 4. Curvas medidas nos 129 checks

Todos os pontos cobrem as mesmas **12.762.000 células, 129 janelas, 43 arquivos,
dez configurações e quatro famílias**. O score continua macro entre famílias,
após pooling por canal físico e macro canais/arquivos/grupos.

Os selecionados do ponto denso apresentam:

| Multiplicador do passo | Fixo: bpc | Aprendido: bpc | Fixo: NRMSE* | Aprendido: NRMSE* |
|---|---:|---:|---:|---:|
| 0,5 | 3,070395 | 3,152211 | 0,468139 | 0,473482 |
| 0,75 | 2,275151 | 2,290383 | 0,616867 | 0,609617 |
| 1 | 1,715285 | 1,708485 | 0,732510 | 0,718137 |
| 1,5 | 1,068070 | 1,037563 | 0,901142 | 0,873857 |
| 2 | 0,727317 | 0,704969 | 1,021486 | 0,985738 |
| 3 | 0,402325 | 0,375991 | 1,155952 | 1,110444 |
| 4 | 0,251129 | 0,237169 | 1,201303 | 1,151774 |

Há ganhos modestos no trecho intermediário, mas não uniformes. No ponto de maior
fidelidade, o aprendido tem **mais bytes e mais erro** que o controle fixo.
A função de inicialização dos dois selecionados compactos é idêntica: suas
curvas coincidiram. Não atribuir seus pontos a aprendizagem neural.

Os números diferem dos dez checks de treinamento porque a população ampliada
tem outros arquivos/posições; não houve retreino ou alteração retroativa do
checkpoint. As curvas não são interpoladas nem presumidas monotônicas.

<a id="report-94-5-referências-práticas-no-mesmo-contrato"></a>

#### 5. Referências práticas no mesmo contrato

| Referência | Stream bpc | NRMSE* macro | Stream total | Tabelas incluídas |
|---|---:|---:|---:|---:|
| VQ compacto maduro | 0,248317 | 0,756812 | 396.127 B | 116.566 B |
| VQ denso maduro | 0,746333 | 0,491288 | 1.190.587 B | 115.882 B |
| Lifting aprendido denso, passo ×2 | 0,704969 | 0,985738 | 1.124.602 B | sem tabelas externas |

O lifting ×2 custa menos que o VQ denso, mas tem **~100,64% mais erro** nesse
score. Todos os dez grupos pioram nessa comparação. Sob o teto de bytes do VQ
compacto, o melhor score dos pontos aprendidos medidos foi **1,139576**, contra
**0,756812** da referência; esse candidato é a inicialização compacta no passo ×4,
não uma melhoria aprendida. Também há regressão nos dez grupos.

Portanto, **o protótipo H1 não deslocou a fronteira prática de stream**.
Isso não refuta learned lifting em geral: foram testados operadores pequenos,
profundidade específica, STE, prior logístico e zlib/planos int32, com budget
declarado. Nem equivale a uma confirmação independente das referências VQ.

<a id="report-94-pesos-compartilhados-e-horizontes"></a>

##### Pesos compartilhados e horizontes

O export completo VQ tem **111.873.813 B**; os exports de lifting têm **2.021 B**
(fixo) e **22.757 B** (aprendido). Para um único objeto curto que precise instalar
os pesos, o custo de modelo pode inverter o ranking de tamanho total. Por exemplo,
o lifting aprendido no passo ×0,5 tem erro macro **0,473482** com stream
~3,1522 bpc e objeto com pesos de **5.051.322 B**; o VQ denso soma
**113.064.400 B**, com erro macro 0,491288.

Isso **não autoriza promover o lifting**: há regressões em sete configurações
nessa comparação, e o export VQ inclui encoder, não só o mínimo receptor.
O ganho de footprint é real nessa exportação; não é ganho certificado de
fidelidade por bit em transporte com pesos já instalados, nem amortização
validada em aquisições contínuas maiores. Não esconder nenhum desses contratos.

<a id="report-94-6-decisão-e-limites-da-sequência"></a>

#### 6. Decisão e limites da sequência

Foram comparados **30 pontos medidos** (28 lifting e duas referências VQ),
com 32 comparações discretas, considerando streams e objetos com pesos.
Nenhuma comparação passou o filtro exploratório de **≥5% menos erro a bytes
não maiores sem regressões por configuração**. A checagem adicional de
**≥10% menos bytes a erro macro e por grupo não piores** também não encontrou
candidato. Essas condições são necessárias nesta triagem, não todas as guardas
físicas finais: nenhum resultado foi promovido.

**Encerrar esta campanha no orçamento registrado. Não lançar P3, extensão de
treino ou nova seed para este protótipo.** Não houve P4 com dados novos ou
semissintéticos; disponibilidade/licença de novas aquisições e arquitetura-alvo
continuam condicionantes. O diagnóstico original-versus-decodificado foi
implementado; o probe específico de correção usando somente reconstrução VQ
continua distinto da referência prática aqui medida.

Próximo investimento defensável, antes de mais treino longo:

1. Calibrar **taxa hard/coder e alocação por subbanda**, distinguindo erro do
   proxy, custo dos planos int32 e seleção do ponto operacional. Primeiro uma
   medição/ablação pequena com transformadas e pesos congelados.
2. Se houver ganho líquido de bytes a fidelidade fixa, testar sua integração
   isoladamente. Não combinar novo coder, nova arquitetura e novos dados em uma
   única continuação para reinterpretar este negativo.
3. A alternativa H2 mantém motivação pelo diagnóstico P1, mas precisa de seu
   próprio contraste em malha fechada; não foi treinada nem declarada aprovada.
4. Preservar a prioridade de aquisições FEBUS realmente independentes. Mais
   janelas dos mesmos arquivos não resolvem a limitação de diversidade.

Entregas efetivas: preparação/contrato prospectivos, diagnóstico espacial inicial,
implementação neural reversível, dois pontos treinados, receptores independentes,
curvas hard/bytes e referências práticas. Confirmação/expansão permanecem
condicionadas — não são etapas silenciosamente dadas como concluídas.

<a id="report-94-7-artefatos-e-verificação-final"></a>

#### 7. Artefatos e verificação final

Em `results/experiments/lifting_postflight_fold0_20261115/`:

- `lifting_curve/`: protocolo, 28 curvas/pontos, receptores e todos os frames;
- `vq_compact/`, `vq_dense/`: tabelas, frames, exports e resultados por canal/arquivo;
- `frontier.json`: comparação original; preservada;
- `frontier_v2.json`: acrescenta a verificação explícita do gate de economia de
  bytes, **sem alterar inferência, pontos ou seleção**;
- `EXIT.json`, `COMPLETE` e logs.

Os descritores de espectro, fase/cobertura, viés, caudas e erro máximo estão
nos resultados por arquivo/canal. A ausência de candidato nas condições escalares
e por grupo já impede P3; não declarar aprovação das guardas restantes.

**Fechamento:** 400 testes locais passaram, documentos e cobertura pareada foram
conferidos, e o snapshot final foi sincronizado preservando divergências remotas.
Os resultados relevantes existem localmente e no remoto. As duas GPUs terminaram
sem processos de compute, com utilização 0%; auditoria:
`results/remote_setup/lifting_closeout_20261115.json`.
Nenhum novo treino ficou enfileirado, não houve commit/push e os dados originais
permaneceram inalterados.

<a id="report-94-continuação-posteriormente-autorizada"></a>

##### Continuação posteriormente autorizada

O usuário concedeu autonomia para investigar e executar novas etapas.
O [relatório 95](92-LEARNED_LIFTING_STUDY.md#report-95) isolou posteriormente
economia lossless importante do coder, sem alterar os coeficientes ou este
resultado histórico. A sequência H2 está nos relatórios
[96](93-NEURAL_ANCHOR_INNOVATION_STUDY.md#report-96) e
[97](93-NEURAL_ANCHOR_INNOVATION_STUDY.md#report-97).
O fechamento acima descreve aquela campanha, não proíbe essa continuação autorizada.

---

<a id="report-95"></a>

## Registro histórico 95 — Taxa hard, coder de símbolos e alocação congelada por subbanda

Fonte original preservada no arquivo: `95-HARD_ENTROPY_AND_SUBBAND_ALLOCATION.md`.

<a id="report-95-taxa-hard-coder-de-símbolos-e-alocação-congelada-por-subbanda"></a>

**Continuação autorizada em 15/11/2026 UTC.** O usuário concedeu autonomia para
avaliar resultados e prosseguir com etapas justificadas, sem encerrar o trabalho
apenas pela conclusão de uma etapa. Mantêm-se proteção dos dados reservados,
registro de critérios e budgets e preservação dos resultados negativos anteriores.

<a id="report-95-1-hipótese-e-separação-causal"></a>

#### 1. Hipótese e separação causal

O [relatório 94](92-LEARNED_LIFTING_STUDY.md#report-94) rejeitou a promoção
do protótipo H1 sob seu coder e sua otimização. O negativo permanece válido.
A investigação seguinte separa:

1. **Recodificação lossless dos mesmos coeficientes**, sem modelo, quantização
   ou waveform diferentes.
2. **Alocação de quantização por subbanda com operadores congelados**, ajustada
   somente nos arquivos de ajuste, usando custos hard reais.
3. Só depois, decidir sobre novo treino H1 ou predição H2 em malha fechada.

Uma inspeção descritiva encontrou ~3,15 bpc de frames em um ponto de alta
fidelidade contra ~1,15 bpc na soma das entropias marginais empíricas das bandas.
O segundo número **omite tabelas/framing e não é um limite informacional**.
Justifica medir o coder completo, não presumir ganho.

<a id="report-95-2-experimento-a-mesmos-coeficientes-outro-coder"></a>

#### 2. Experimento A: mesmos coeficientes, outro coder

Implementação `compression/adaptive_coefficient_stream.py`, formato **FRC2**:

- range coding categórico por banda, via `constriction==0.5.0`;
- alfabeto e contagens exatas explicitamente transmitidos, com custo;
- bandas constantes e planos zlib competem pelo tamanho completo;
- fallback de frame inteiro FRC1, identificado pelo magic, quando menor;
- limites de alocação, CRC de frame e coeficientes, contagens verificadas;
- nenhum prior de teste/validação, tabela invisível ou feature não transmitida.

O programa `finetuning/audit_lifting_entropy.py` recodifica 21 pontos do
postflight: sete do fixo denso, sete do aprendido denso e sete da inicialização
compacta. A cópia compacta aprendida é omitida por ter a mesma função inicial.
Exige identidade bit a bit de arrays, shapes, descriptor e side.
Os waveforms e métricas anteriores permanecem exatamente os mesmos, sob a mesma
implementação de síntese. Nenhum sinal bruto precisa ser aberto nessa etapa.

Artefatos: `results/experiments/lifting_entropy_recode_20261115/`.
O protocolo é gravado antes da leitura dos frames e inclui hashes.

<a id="report-95-3-experimento-b-planejado-alocação-com-taxa-real-sem-treino-da-rede"></a>

#### 3. Experimento B planejado: alocação com taxa real, sem treino da rede

Antes de qualquer resultado desse ajuste, fixar:

- pesos selecionados previamente, fixo/aprendido do ponto denso, sem retreino;
- uma janela central dentre os três recortes P1 de cada um dos **44 arquivos
  de ajuste**, escolhida por metadados; os 43 arquivos de checks não ajustam passos;
- escala física prospectiva já congelada; mesma cobertura e normalização;
- multiplicadores por banda `0,25/0,5/0,75/1/1,5/2/3/4`;
- lambdas `0,01/0,03/0,1/0,3/1/3`;
- busca coordenada determinística, no máximo três passagens por lambda;
- objetivo: quadrado da macro NRMSE* aproximada no domínio físico + lambda
  vezes taxa FRC2 completa nos arquivos de ajuste;
- macro canais/arquivo/configuração/família; a aproximação no ajuste usa a
  relação afim de normalização. Avaliação final restaura float32 efetivamente;
- custos de bandas pré-calculados, mas tabelas, framing, passos e side cobrados;
- no máximo 1.296 avaliações de vetor de passos por braço; zero updates neurais;
- comparação final nos mesmos 129 checks/43 arquivos, com receptor independente.

Esse desenho não troca loss, arquitetura, fonte ou pesos junto com o coder.
Usar duas GPUs remotas, uma por braço, após sincronização preservando divergências.
A redução de contagens/tabelas é adaptação **transmitida por frame**, não ajuste
gratuito aos checks. Os passos globais são ajustados apenas nos arquivos de ajuste.

<a id="report-95-4-decisão-registrada"></a>

#### 4. Decisão registrada

Economia lossless do coder não promove arquitetura. Comparar candidatos ao
controle fixo e às referências VQ completas, discriminando setup e pesos.
Exigir os gates exploratórios já definidos por erro/bytes e configurações;
guardas físicas e confirmação continuam necessárias para promoção.

Se a calibração corrigir apenas parte do déficit, documentar sua extensão antes
de investir em mais treino. A alternativa H2 permanece uma campanha distinta,
com âncoras efetivamente decodificadas e inovação transmitida; não usar contexto
original gratuito nem confundir correção VQ com H2.

**Estado inicial:** A em execução; B ainda é protocolo. Nenhum treino adicional
foi lançado neste registro.

<a id="report-95-5-resultado-a-economia-lossless-confirmada"></a>

#### 5. Resultado A: economia lossless confirmada

A recodificação concluiu nos 21 pontos, exigindo identidade bit a bit de todos
os coeficientes, shapes, descriptors e sides dos 129 frames por ponto.
Não há novo waveform nem mudança de métrica.

| Ponto denso | FRC1 bpc | Novo coder com fallback bpc | Economia | NRMSE* inalterado |
|---|---:|---:|---:|---:|
| Fixo, passo ×0,5 | 3,070395 | 1,246524 | 59,40% | 0,468139 |
| Aprendido, passo ×0,5 | 3,152211 | 1,263207 | 59,93% | 0,473482 |
| Fixo, passo ×1 | 1,715285 | 0,709022 | 58,66% | 0,732510 |
| Aprendido, passo ×1 | 1,708485 | 0,702660 | 58,87% | 0,718137 |
| Fixo, passo ×2 | 0,727317 | 0,341429 | 53,06% | 1,021486 |
| Aprendido, passo ×2 | 0,704969 | 0,328127 | 53,46% | 0,985738 |

São custos completos de frames e setup, com alfabetos/contagens transmitidos.
Os pesos compartilhados permanecem os mesmos, não entram nessa taxa de stream.
Por exemplo, no aprendido ×0,5 os 129 frames escolheram FRC2; 1.091 bandas
usaram categórico e 70 usaram planos zlib. No aprendido ×1, um frame preferiu
FRC1; essa decisão é explícita pelo magic e não altera os coeficientes.

**Leitura:** o coder inicial respondia por parte importante do déficit de taxa.
O relatório 94 continua correto para seu stream, mas seu negativo não deve ser
generalizado a um coder eficiente. Ainda não há avanço sobre VQ denso: a alta
fidelidade de 0,473482 custa ~1,2632 bpc, versus 0,491288 a ~0,7463 bpc da
referência VQ. A economia lossless por si só não promove H1.

<a id="report-95-6-continuação-b-lançada"></a>

#### 6. Continuação B lançada

Em seguida foi lançado o ajuste congelado especificado na seção 3, com GPU 0
para o controle fixo e GPU 1 para os operadores aprendidos.
`tools/run_lifting_allocation.py` exige GPUs sem processos de compute,
registra comandos/PIDs, espera ambos os braços e grava códigos de saída.
Diretório: `results/experiments/lifting_hard_allocation_20261115/`.

A implementação é `finetuning/fit_lifting_allocation.py`. Os custos de bandas
são calculados por codificação real, e serão conferidos contra a serialização
integral dos 44 frames de ajuste em cada perfil. O receptor novo está em
`compression/adaptive_lifting_stream.py`, sem alterar o leitor FRC1 histórico.
Cada vetor de passos é gravado antes da leitura dos 129 waveforms de checks;
esses checks não ajustam passos ou operadores.

Neste lançamento há **zero updates neurais**: a nova otimização altera apenas
os nove passos de quantização explicitamente transmitidos. Resultados B ainda
não estão disponíveis nesta atualização.

<a id="report-95-7-resultado-b-e-próximo-diagnóstico-registrado"></a>

#### 7. Resultado B e próximo diagnóstico registrado

Os dois ajustes terminaram: **525 / 505 vetores únicos avaliados**
(fixo/aprendido), abaixo do teto previsto. Para cada um dos seis perfis por braço,
o custo pré-calculado FRC2 bateu com a serialização completa dos 44 frames de
ajuste; o receptor separado reconstruiu os checks. Os operadores ficaram congelados.

| Lambda | Fixo: bpc | Fixo: NRMSE* | Aprendido: bpc | Aprendido: NRMSE* |
|---|---:|---:|---:|---:|
| 0,01 / 0,03 / 0,1 | 1,934832 | 0,275760 | 2,029946 | 0,299558 |
| 0,3 | 1,301891 | 0,433502 | 1,134675 | 0,505474 |
| 1 | 0,526626 | 0,759954 | 0,484347 | 0,771490 |
| 3 | 0,209024 | 1,091270 | 0,157136 | 1,122183 |

Os três lambdas menores encontraram o mesmo limite inferior da grade de passos.
O ajuste movimentou a curva, mas não demonstrou ganho generalista frente ao VQ.
Não interpolar os perfis grosseiros para certificar economia a erro idêntico.
O aprendido não ganhou uma justificativa para continuação longa por esse ajuste.

**Diagnóstico C registrado antes da execução:** verificar o custo de predição
inteira reversível dos mesmos coeficientes, para identificar dependência ainda
explorável por um futuro coder condicionado. Métodos por banda:
identidade, diferenças temporal/espacial e Lorenzo 2D, com aritmética int64
intermediária e rejeição de overflow int32. Cada modo compete por bytes reais,
e modos, tabelas, checksums e wrapper entram no frame. Fallback integral ao melhor
FRC1/FRC2 impede regressão de tamanho. Não há novo alvo, quantização ou treino.

Implementações: `compression/predictive_coefficient_stream.py` (FRC3) e
`finetuning/audit_predictive_coefficients.py`. Aplicar aos mesmos 21 pontos,
exigindo identidade bit a bit dos coeficientes/side/descriptor.
Esse diagnóstico é distinto do preditivo histórico sobre o waveform: aqui o
objeto reversível é cada **subbanda quantizada** do codec neural.
Se a dependência restante render pouco, não iniciar um prior neural apenas
pelo nome; priorizar o contraste H2 em malha fechada com protocolo próprio.

<a id="report-95-8-resultado-c-e-transição-de-mecanismo"></a>

#### 8. Resultado C e transição de mecanismo

C concluiu nos 21 pontos, com identidade de coeficientes/side/descriptor.
A economia adicional ficou em **0–0,410%** sobre o coder adaptativo:

| Aprendido denso | FRC1/FRC2 adaptativo bpc | Com predição inteira bpc |
|---|---:|---:|
| Passo ×0,5 | 1,263207 | 1,261037 |
| Passo ×1 | 0,702660 | 0,701313 |
| Passo ×2 | 0,328127 | 0,326782 |

No passo ×0,5 apenas nove dos 129 frames escolheram o wrapper preditivo;
no ×1 foram sete e no ×2 seis. O fallback garante que o ganho pequeno não veio
de uma regressão de fidelidade. Isso não limita um contexto neural não linear,
mas não sustenta colocá-lo à frente da hipótese estrutural H2 agora.
Artefatos: `results/experiments/lifting_predictive_coefficients_20261115/`.

Concluídas A/B/C, prosseguir com âncoras neurais e inovação em malha fechada,
conforme o [relatório 96](93-NEURAL_ANCHOR_INNOVATION_STUDY.md#report-96).
As economias do coder ficam disponíveis para esses codecs; não são promoção
do H1 e não modificam seu resultado histórico.
