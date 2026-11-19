# Metadados físicos, clocks e proveniência

**Etapa consolidada dos registros 109, 110, 111, 112.** Cabeçalhos e relógios distinguem recortes, mas não certificam independência de aquisições.

Protocolos, resultados e auditorias foram reunidos sem alterar resultados, budgets ou gates. As partes históricas seguem em ordem: estados de lançamento/execução descrevem aquele momento, não tarefas ainda pendentes. Consulte as últimas partes para o encerramento.

## Registros desta etapa

- [Registro 109](#report-109) — H3: inventário físico antes de condicionar outro codec
- [Registro 110](#report-110) — Metadados físicos completos, diversidade e proveniência ainda limitadas
- [Registro 111](#report-111) — Proveniência: relógios e fingerprints de treino, sem reabrir reservas
- [Registro 112](#report-112) — Relógios distinguem recortes; não apareceu cópia nos fingerprints amostrados

<a id="report-109"></a>

## Registro histórico 109 — H3: inventário físico antes de condicionar outro codec

Fonte original preservada no arquivo: `109-PHYSICAL_METADATA_PREFLIGHT_PROTOCOL.md`.

<a id="report-109-h3-inventário-físico-antes-de-condicionar-outro-codec"></a>

**15/11/2026. Registrado antes do novo censo de cabeçalhos.**

<a id="report-109-escopo"></a>

#### Escopo

Somente os **87 IDs de arquivos do treino fold0** no manifesto preparado.
Abrir HDF5 locais correspondentes em `data/finetune/raw/` apenas para atributos,
nomes, shapes e dtype do dataset previamente selecionado e seus ancestrais.
Não ler valores de datasets, relógios ou waveforms. Não abrir HDF5 val/test/EDAS.
Metadados de inventário CSV podem ser consultados sem abrir sinais.

Conferir presença de dx/spacing, gauge length, pulse width, derivação e
identificadores/tempos de aquisição nos atributos. Preservar nomes/unidades
originais; não inferir unidades pela magnitude ou pela frequência no filename.
Não trocar o dataset previamente auditado por fallback ao primeiro dataset.

<a id="report-109-perguntas-e-decisão"></a>

#### Perguntas e decisão

1. Quais variáveis físicas realmente variam entre os dez grupos de treino?
2. Elas são completas, consistentes dentro dos arquivos e redundantes com
   rótulos de grupo? Não alegar causalidade com apenas dez configurações.
3. Há atributos suficientes para um vetor físico comum sem usar nomes de família,
   excitação, alvo ou estatísticas de validação?
4. Há proveniência suficiente para distinguir aquisições de cópias/arquivos?
   IDs Drive e nomes diferentes, sozinhos, não demonstram independência.

Publicar cobertura por campo/arquivo, combinações de geometria, ausências,
conflitos com dt/dx já preparados e custo potencial de transmitir metadados
novos. Cabeçalhos lidos não provam sincronização de núcleos.

Se a cobertura permitir H3, registrar **outro protocolo** para um adaptador
físico contra o mesmo modelo com vetor constante, mesma função inicial,
loss, taxa e budget. Se as variáveis forem ausentes/inidentificáveis, documentar
o bloqueio e priorizar aquisição/proveniência, não inventar valores ou prometer
generalização. Nenhum treinamento fica implícito neste censo.

Destino: `results/experiments/physical_metadata_preflight_fold0_20261115/`.
Zero updates, zero inferência neural, zero leitura de valores de datasets.

<a id="report-109-complemento-de-inventário-remoto-somente-metadados"></a>

#### Complemento de inventário remoto, somente metadados

Após o censo local, conferir se há material adicional nas quatro famílias FEBUS
da pasta DAS já usada pelo projeto: SMF Configurations, Multicore Fiber,
NP-doped Fiber e Multimode Fiber. Listar recursivamente apenas nomes, IDs,
tipos e tamanhos; não baixar HDF5 ou abrir conteúdo de dados. Não percorrer
EDAS Comparison nem modificar o particionamento reservado.

Comparar contagens por família e fingerprint do conjunto ordenado de IDs com o
inventário local de 207 HDF5. A fingerprint FNV-1a64 é um auxiliar de inventário,
não um checksum criptográfico de conteúdo ou evidência de independência.

<a id="report-109-resultado"></a>

##### Resultado

Censo de 87 cabeçalhos e listagem de inventário concluídos sem leitura de
valores de datasets. Metadados completos, porém constantes por configuração;
nenhuma nova aquisição bruta demonstrada na listagem das quatro famílias.
Decisão e contrato de novas aquisições no
[relatório 110](98-PHYSICAL_METADATA_AND_PROVENANCE.md#report-110).

---

<a id="report-110"></a>

## Registro histórico 110 — Metadados físicos completos, diversidade e proveniência ainda limitadas

Fonte original preservada no arquivo: `110-PHYSICAL_METADATA_AND_ACQUISITION_COVERAGE.md`.

<a id="report-110-metadados-físicos-completos-diversidade-e-proveniência-ainda-limitadas"></a>

**15/11/2026.** Resultado do protocolo 109. Sem inferência ou valores de datasets.

<a id="report-110-censo-local-de-cabeçalhos"></a>

#### Censo local de cabeçalhos

Os **87 HDF5 do treino** estão presentes. Spacing, GaugeLength, PulseWidth,
PulseRateFreq e DerivationTime existem em todos. O dx e dt do preparo coincidem
com os cabeçalhos sob a convenção histórica explícita `Spacing[1] × 1e-3`;
nenhum conflito foi encontrado. Outros campos mantêm semântica nativa:
não inventar unidades de pulso a partir da magnitude ou do filename.

- Oito combinações de dx/GL/pulso/derivação/frequência de pulso.
- Nove combinações incluindo sampling rate/resolution, oversampling e potência.
- **Um único vetor de hardware por configuração** nos dez grupos de treino.
- DerivationTime é constante nessa população; dt preparado também.
- Um registro comprimido dos campos numéricos por arquivo custaria **2.705 B**
  nesta representação preliminar, não um stream operacional já implementado.

É possível construir um adaptador com esses campos, mas seu efeito no treino
seria confundido com adaptação por configuração. O censo não demonstra que
FiLM físico deslocaria a fronteira ou transferiria a combinações novas.

<a id="report-110-proveniência-de-aquisição"></a>

#### Proveniência de aquisição

Há **dez combinações de Hostname/AcqStartTime** nos 87 arquivos: uma por grupo.
Não chamar automaticamente cada ID Drive de uma aquisição independente.
Também não concluir que todos os arquivos de cada grupo são duplicatas:
o horário pode ser herdado de exportações e não foi verificado contra relógios
ou registros de sessão. Nenhum valor de clock/dataset foi aberto neste censo.

Os estudos anteriores já declaravam checks de desenvolvimento historicamente
consumidos, não confirmação independente; essa evidência reforça o cuidado.
Não mudar retroativamente splits ou métricas para fabricar confirmação.

<a id="report-110-conferência-do-drive-derivados-não-são-dados-novos"></a>

#### Conferência do Drive: derivados não são dados novos

Listagem somente de metadados nas quatro famílias FEBUS conhecidas:

| Escopo | HDF5 |
|---|---:|
| Recursivo, incluindo Processados e FFT | 351 |
| Diretamente nas 23 pastas de configuração | **207** |
| Diferença em pastas derivadas | 144 |

Contagens dos brutos: single-mode 72, multicore 90, doped 27 e multimode 18.
A fingerprint ordenada dos IDs coincide com o inventário local de 207 arquivos.
Isso é uma checagem de inventário, não hash criptográfico de conteúdo.
As pastas adicionais têm nomes `Processados`/`fft_per_block_outputs`;
não foram tratadas como novas aquisições, nem seus sinais baixados.
EDAS não foi percorrido. Não afirmar que toda possível fonte do Drive foi esgotada.

<a id="report-110-decisão-e-próximo-requisito"></a>

#### Decisão e próximo requisito

**Não iniciar agora outro piloto de condicionamento físico nos mesmos grupos.**
Não é uma refutação de H3; faltam diversidade identificável e proveniência para
uma conclusão de transferência, e não apareceu uma vitória prática a confirmar.
Também não reciclar FFT/processados como pré-treino supostamente independente.

A próxima expansão com maior valor científico é **aquisição FEBUS independente
documentada**, preservando waveform/ruído, com sessões, excitações e geometria
verificáveis. Antes de incorporá-la:

1. identificar sessão e origem, vínculos de cópia/derivação e checksum;
2. declarar variável/unidade, dtype, clocks, dt, dx, GL, pulso e derivação;
3. separar novas sessões entre desenvolvimento e confirmação antes de treinar;
4. não criar continuidade artificial entre NPZ nem reamostrar o alvo para
   esconder diferenças físicas;
5. manter teste histórico e EDAS reservados, sem convertê-los em treino.

Foi acrescentado um **template e validador de manifesto** para essa entrada,
sem inventar sessões ou tocar dados existentes. Um template vazio não passa:
precisa de registros reais com unidades e proveniência explícitas.
Isso não certifica independência apenas por validar sintaxe; a origem ainda
precisa de conferência documental e de integridade.

Até obter essa evidência, manter a referência madura e os negativos arquivados,
em vez de consumir GPUs com novas variantes indistinguíveis. O bloqueio é
experimental/proveniência, não falta de capacidade de executar mais updates.

Artefatos:
`results/experiments/physical_metadata_preflight_fold0_20261115/`,
incluindo `CONSISTENCY_AND_PROVENANCE.json` e `DRIVE_METADATA_CENSUS.json`.

Verificação de engenharia final: **452 testes passaram**. Nenhum dado original,
split reservado ou resultado histórico foi descartado; sem commit/push.

<a id="report-110-aprofundamento-posterior"></a>

##### Aprofundamento posterior

A auditoria de clocks/amostras no [relatório 112](98-PHYSICAL_METADATA_AND_PROVENANCE.md#report-112)
encontrou 81 arrays de relógio distintos e nenhuma coincidência de fingerprint
não constante amostrado. Os dez AcqStartTime não devem ser confundidos com dez
arquivos idênticos ou uma contagem comprovada de sessões.
O teste H5 subsequente está em
[114](99-CLOCK_ALIGNED_MULTICORE_OPPORTUNITY.md#report-114). A normalização **interna** local,
hipótese separada do condicionamento físico aqui adiado, foi registrada em
[115](100-LOCAL_INTERNAL_NORMALIZATION_STUDY.md#report-115).

---

<a id="report-111"></a>

## Registro histórico 111 — Proveniência: relógios e fingerprints de treino, sem reabrir reservas

Fonte original preservada no arquivo: `111-TRAIN_CLOCK_AND_COPY_AUDIT_PROTOCOL.md`.

<a id="report-111-proveniência-relógios-e-fingerprints-de-treino-sem-reabrir-reservas"></a>

**15/11/2026. Registrado antes da leitura dos clocks e dos novos recortes.**
Continuação autorizada do relatório 110.

<a id="report-111-escopo-e-pergunta"></a>

#### Escopo e pergunta

Os 87 arquivos de treino têm dez combinações Hostname/AcqStartTime. Isso pode
representar sessões compartilhadas ou atributos herdados na exportação.
Verificar relógios e conteúdo antes de chamar isso de duplicação ou independência.
Não mudar splits, pesos, métricas ou resultados históricos.

Abrir **somente os 87 HDF5 de treino**, pelo ID e dataset já registrados.
Teste, validação e EDAS não serão abertos nesta etapa.

<a id="report-111-plano-fixo"></a>

#### Plano fixo

1. Ler o dataset `time` do Source correspondente, quando existir e tiver um
   valor por bloco. Preservar float64 e unidade nativa; publicar monotonicidade
   e coincidência de valores, sem inventar conversão de relógio.
2. Por bloco do dataset preparado [bloco, tempo, canal], usar até oito canais
   uniformemente distribuídos no conjunto **retido pelo mapa físico daquele
   bloco**, e três recortes temporais de até 256 amostras: início, centro e fim.
   Seleção exclusivamente por metadados; não procurar trechos parecidos.
3. SHA-256 dos valores nativos e coordenadas; registrar separadamente hashes de
   relógio. Amostras constantes/zero são marcadas e não evidenciam cópia.
4. Comparar dentro do treino, discriminando os papéis fit/check por arquivo já
   usados. Relógios comuns não provam valores idênticos; fingerprints distintos
   não provam aquisições independentes.
5. Se houver fingerprints **não constantes** coincidentes em arquivos diferentes,
   conferir os blocos completos de até **oito pares lexicográficos**, com hash
   streaming e teto de **1 GiB de valores lidos** nessa confirmação. A escolha
   dos pares não depende de scores do codec. Publicar qualquer cobertura limitada.

Nenhum resultado permite automaticamente afirmar cópia integral de arquivo:
hash completo do bloco confirma somente valores daquele bloco/dataset.
Não ler alvos reservados para procurar overlap; uma auditoria entre splits
precisaria de protocolo e autorização de escopo próprios.

<a id="report-111-decisão"></a>

#### Decisão

Se houver cópia/overlap comprovados nos checks de treino, publicar a limitação,
sem apagar negativos nem selecionar de novo modelos. Se clocks forem herdados
com conteúdo diferente, registrar que AcqStartTime não identifica sessões
independentes por si só. A documentação de aquisição continua necessária.
Uma expansão de corpus não poderá contar blocos/copias confirmados como novas
aquisições ou descartar ruído para criar diversidade artificial.

Destino: `results/experiments/train_clock_copy_audit_fold0_20261115/`.
Zero updates e zero inferência neural. A transição fingerprint → hash completo
limitado é automática e está autorizada por este protocolo.

<a id="report-111-resultado"></a>

##### Resultado

87 clocks e 486 fingerprints de bloco foram auditados. Nenhum candidato
não constante coincidiu, portanto não houve leitura de hash completo.
Conclusões e limites no [relatório 112](98-PHYSICAL_METADATA_AND_PROVENANCE.md#report-112).

---

<a id="report-112"></a>

## Registro histórico 112 — Relógios distinguem recortes; não apareceu cópia nos fingerprints amostrados

Fonte original preservada no arquivo: `112-TRAIN_CLOCK_AND_COPY_AUDIT_RESULTS.md`.

<a id="report-112-relógios-distinguem-recortes-não-apareceu-cópia-nos-fingerprints-amostrados"></a>

**15/11/2026.** Resultado do protocolo 111.

<a id="report-112-cobertura-e-resultado"></a>

#### Cobertura e resultado

Foram lidos os clocks dos **87 arquivos de treino** e fingerprints de **486
blocos**, somando **2.985.984 células amostradas**. Todos os clocks suportados
são float64 e estritamente crescentes dentro de seus arquivos.
Não houve leitura de validação, teste ou EDAS, nem treino/inferência neural.

- Os dez Hostname/AcqStartTime do cabeçalho correspondem a **81 arrays de
  clocks distintos**. AcqStartTime não delimita os recortes exportados.
- Seis pares de arquivos têm arrays completos de clock iguais; todos estão nos
  grupos multicore de treino. Quatro desses pares atravessam os papéis
  históricos fit/check por arquivo.
- **Nenhum par de fingerprints não constantes coincidiu**.
  Todos os 486 recortes amostrados foram não constantes.
- Não houve candidato à etapa de hash completo: ela corretamente leu zero bytes.

Isso não comprova independência de sessões e não é hash integral do corpus.
Também não há evidência aqui para chamar esses checks de cópias dos alvos:
mesmo clock não implica mesmo waveform, sobretudo entre canais de aquisição.

<a id="report-112-consequência-para-os-estudos-anteriores"></a>

#### Consequência para os estudos anteriores

Os checks eram explicitamente desenvolvimento de arquivos de treino vistos
pela fonte, não confirmação independente. A sobreposição nominal de relógio
reforça que separação por Drive ID não equivale a separação de sessão.
Não mudar scores/splits históricos nem reinterpretar negativos como positivos.

Não reduzir “87 arquivos” a “dez aquisições idênticas”, nem elevar “81 clocks”
a “81 sessões independentes”: ambas seriam extrapolações sem proveniência.

<a id="report-112-continuação-identificada-contexto-entre-canais-de-aquisição-multicore"></a>

#### Continuação identificada: contexto entre canais de aquisição multicore

Os arquivos simultâneos são candidatos à hipótese H5 do relatório 90, **distinta
de prever canais espaciais alternados**. Nessa dimensão, os encoders podem
continuar observando um tile espacial completo por núcleo.

Antes de investir em outra rede, registrar um diagnóstico limitado:
alinhar somente clocks exatamente comuns, conferir geometria e canais físicos,
separar os dois lados do mesmo par dos novos papéis fit/check e medir dependência.
Nenhum núcleo será considerado contexto gratuito num eventual codec.

Clocks comuns são alinhamento nominal, não prova de sincronismo físico fino;
não ajustar offsets pelos checks nem reamostrar o alvo para aumentar correlação.
O protocolo 113 fixa a seleção e o gate para avaliar contexto VQ decodificado.

Artefatos: `results/experiments/train_clock_copy_audit_fold0_20261115/`.
