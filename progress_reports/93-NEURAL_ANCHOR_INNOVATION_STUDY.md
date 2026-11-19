# H2 — âncoras neurais e inovação

**Etapa consolidada dos registros 96, 97, 98.** Preflight congelado e treino conjunto separados; ganho contra o controle, não contra o VQ maduro.

Protocolos, resultados e auditorias foram reunidos sem alterar resultados, budgets ou gates. As partes históricas seguem em ordem: estados de lançamento/execução descrevem aquele momento, não tarefas ainda pendentes. Consulte as últimas partes para o encerramento.

## Registros desta etapa

- [Registro 96](#report-96) — H2: âncoras neurais decodificadas e inovação efetivamente transmitida
- [Registro 97](#report-97) — H2 conjunto: codec neural de âncoras e inovação, com controle independente
- [Registro 98](#report-98) — H2 conjunto: mecanismo positivo, fronteira prática ainda não alcançada

<a id="report-96"></a>

## Registro histórico 96 — H2: âncoras neurais decodificadas e inovação efetivamente transmitida

Fonte original preservada no arquivo: `96-CLOSED_LOOP_NEURAL_ANCHOR_INNOVATION.md`.

<a id="report-96-h2-âncoras-neurais-decodificadas-e-inovação-efetivamente-transmitida"></a>

**Protocolo de continuação, 15/11/2026 UTC; registrado antes dos resultados.**

<a id="report-96-1-por-que-mudar-o-mecanismo-agora"></a>

#### 1. Por que mudar o mecanismo agora

O [relatório 95](92-LEARNED_LIFTING_STUDY.md#report-95) separou as causas:
o coder categórico economizou ~53–60% em pontos destacados, mas alocação congelada
não demonstrou avanço sobre VQ. O diagnóstico reversível posterior de diferenças
nos coeficientes rendeu apenas pequenas economias adicionais, sem mudar waveform.
Não há evidência suficiente para priorizar outro prior neural só pelo nome.

Avançar para H2 muda **o que o stream transmite**, não apenas os bytes dos
mesmos símbolos: uma CNN VQ codifica canais-âncora; os demais são representados
por inovação calculada a partir das âncoras **já decodificadas**.

<a id="report-96-2-preflight-congelado-e-controle"></a>

#### 2. Preflight congelado e controle

Primeiro, sem updates neurais:

1. Normalização robusta global comum de todo o tile, side FNE1 transmitido uma vez.
2. Canais pares adquiridos viram âncoras, codificadas pelo VQ maduro compacto ou
   denso do fold 0; sem decimação do alvo final e sem canais descartados.
3. Decoder separado recupera índices/vetores e reconstrói as âncoras.
4. Dois coeficientes de predição linear por configuração são ajustados somente
   nos 132 recortes dos 44 arquivos de ajuste, a partir desse contexto discreto.
5. Os coeficientes são transmitidos em cada frame. A inovação dos canais ímpares
   é transformada por Haar completo, quantizada e codificada com FRC1/FRC2.
6. Receptor reconstrói âncoras, previsão e inovação, intercala os canais e
   restaura a escala física completa. Escapes/constantes preservam seus bits.

Controle: mesmas âncoras e inovação/coder, mas predição identicamente zero.
Assim se isola a dependência aproveitada, sem oferecer uma base completa de
todos os canais gratuitamente a um corretor.

Grade fixa de passos de inovação: **0,25 / 0,5 / 1 / 2 / 4** no domínio
normalizado global. Dois anchors VQ × dois métodos × cinco passos = **20 pontos**.
Tabelas marginais dos índices são ajustadas apenas nos arquivos de ajuste,
serializadas e cobradas, como as referências práticas. Shapes, normalização,
coeficientes, digests, CRC, framing e mapas necessários também entram na taxa.
Não contar o restante do alvo como conhecido nem reutilizar seu residual original
como contexto do receptor.

<a id="report-96-3-dados-e-integridade"></a>

#### 3. Dados e integridade

Mesmos 129 checks/43 arquivos de treino de P1; mesmos targets e escalas prospectivas.
São desenvolvimento consumido, não validação externa. Sem leitura de waveforms
de validação/teste ou EDAS reservado nessa etapa.

Limitação específica: o espaçamento efetivo das âncoras é **2×dx**. O VQ
congelado não foi treinado especialmente nessa geometria ou na normalização de
subconjuntos. A matriz não é reamostrada para esconder isso. Um negativo congelado
não refuta H2 com encoders adaptados conjuntamente.

Exigir índices e replay exatos com receptor separado. Registrar também o erro
das âncoras com os demais canais hipoteticamente perfeitos, **só como diagnóstico
de piso de erro**, sem atribuir taxa/compressão a essa referência otimista.
Esse diagnóstico pode identificar um gargalo que um preditor dos ímpares não
resolveria mantendo as âncoras congeladas.

<a id="report-96-4-implementação-e-recursos"></a>

#### 4. Implementação e recursos

- `compression/anchor_innovation_stream.py`: formato FAI1 e receptor em malha fechada.
- `finetuning/run_anchor_innovation_probe.py`: ajuste train-only, controle e curva.
- `tests/test_anchor_innovation_stream.py`: paridade, bordas, canal único,
  constantes, escapes e replay com índices efetivamente codificados.

Duas GPUs remotas, uma por origem VQ, apenas inferência nessa fase.
Diretório planejado: `results/experiments/anchor_innovation_preflight_20261115/`.
Nenhum checkpoint histórico é alterado, nenhum treino fica implícito no preflight.

<a id="report-96-5-critérios-de-continuação"></a>

#### 5. Critérios de continuação

- Promoção continua exigindo comparação por bytes e guardas físicas por grupo,
  mais confirmação; esse preflight não promove modelos.
- Se houver oportunidade em malha fechada e proximidade à referência prática,
  considerar preditor neural treinado com o mesmo contexto/coder.
- Se o gargalo for adaptação das âncoras/inovação, testar H2 **conjuntamente
  treinável**, com controle independente de mesma capacidade/grades e budget
  explícito. Não estender o codec H1 antigo como se fosse essa hipótese.
- Um eventual piloto neural deverá registrar antes de rodar inicializações,
  otimizadores sem alias, sequência de dados, bytes, saúde dos dois quantizadores,
  limite de updates e critérios de extensão. Não prolongar por mera passagem de tempo.

**Estado inicial:** implementação/preflight registrados; ainda sem resultado
de H2 e sem updates neurais novos.

<a id="report-96-6-preflight-concluído-e-resultado"></a>

#### 6. Preflight concluído e resultado

Os 20 pontos foram codificados/decodificados nas duas GPUs, sem updates neurais.
Artefatos: `results/experiments/anchor_innovation_preflight_20261115/`.

| Âncoras / passo inovação | Independente bpc | Malha fechada bpc | Independente NRMSE* | Malha fechada NRMSE* |
|---|---:|---:|---:|---:|
| Compactas / 0,5 | 1,299488 | 1,227673 | 0,635119 | 0,635937 |
| Compactas / 2 | 0,607685 | 0,561432 | 0,886785 | 0,879831 |
| Densas / 0,25 | 2,037659 | 1,834684 | 0,471876 | 0,471789 |
| Densas / 0,5 | 1,568890 | 1,398353 | 0,543194 | 0,542457 |
| Densas / 2 | 0,877088 | 0,778712 | 0,794859 | 0,761812 |
| Densas / 4 | 0,657913 | 0,600576 | 0,944157 | 0,849043 |

Em densas/passo 4, o ganho de erro é **~10,07% com menos bytes e sem regressão
nos dez grupos**, contra o controle independente correspondente. É um positivo
exploratório **do mecanismo**, não superioridade sobre o VQ completo: a referência
prática densa do relatório 94 tem erro 0,491288 a 0,746333 bpc.
Outros pontos têm pequenas regressões por grupo; igual passo não é igual erro.

<a id="report-96-o-gargalo-não-é-apenas-o-preditor"></a>

##### O gargalo não é apenas o preditor

O diagnóstico de erro das âncoras, com não-âncoras ideais até o arredondamento
do roundtrip, teve macro **0,481683 compacta / 0,389758 densa**.
Na origem densa, mesmo esse diagnóstico fica acima do erro da referência VQ
completa em single-mode 11, 12, 4 e 7. O caso mais forte é configuração 11:
**0,855472 nas âncoras sozinhas versus 0,399561 do VQ completo**.
Um preditor dos ímpares não remove erro já fixado nos pares.

Portanto, não prolongar um preditor sobre essas âncoras congeladas. A oportunidade
de malha fechada e a mudança de geometria justificam o contraste seguinte:
**adaptação conjunta dos dois encoders/decoders, quantizadores e preditor**,
com controle independente igualmente adaptado. Esse é o
[protocolo 97](93-NEURAL_ANCHOR_INNOVATION_STUDY.md#report-97), não uma continuação
dos pesos H1 nem uma declaração de que o preflight congelado foi aprovado.

---

<a id="report-97"></a>

## Registro histórico 97 — H2 conjunto: codec neural de âncoras e inovação, com controle independente

Fonte original preservada no arquivo: `97-JOINT_NEURAL_ANCHOR_INNOVATION_PROTOCOL.md`.

<a id="report-97-h2-conjunto-codec-neural-de-âncoras-e-inovação-com-controle-independente"></a>

**15/11/2026 UTC. Protocolo anterior a smoke/piloto científicos.**
Continuação autônoma autorizada, baseada nos relatórios 95–96.

<a id="report-97-1-hipótese-controlada"></a>

#### 1. Hipótese controlada

A geometria de canais alternados prejudicou as âncoras VQ congeladas, apesar
de existir ganho de malha fechada sobre o controle. Adaptar conjuntamente:

- encoder, codebook K2048×128 e decoder das âncoras;
- preditor de canais ímpares que recebe exclusivamente as âncoras discretamente
  reconstruídas;
- encoder, codebook K2048×128 e decoder da inovação;
- perda no waveform completo reconstruído, mesma máscara física entre braços.

**Controle independente:** dois backbones iguais, mas previsão zero. O segundo
codifica diretamente os canais ímpares. **H2:** previsão começa como média de
âncoras vizinhas, acrescida de uma CNN 3×5, largura 32 e última camada zero.
Ambos usam duas CNNs densas, mesma origem, geometria, normalização, dose e
otimização. H2 tem ~17 mil parâmetros ativos adicionais no preditor; não alegar
igualdade exata de parâmetros treináveis. Esse contraste isola o mecanismo H2
completo, não exclusivamente a contribuição da CNN contra uma média fixa.

Os dois backbones (~28M cada) são cópias **sem alias** da origem densa fold 0,
época selecionada 23, `dense_transfer_fold0_weights_p93.pt`. Herdam a história de
pré-treino OOI/FEBUS; não são treino do zero. O encoder residual recebe inovação
no mesmo domínio normalizado global, sem estatísticas privadas por ramo.

<a id="report-97-2-discreto-real-em-treino-e-receptor"></a>

#### 2. Discreto real em treino e receptor

O forward usa **vetores exatos do codebook**. O gradiente do encoder passa
apenas por `z-z.detach()`, que vale exatamente zero antes da soma aos vetores.
Isso evita transportar ao preditor o pequeno residual numérico de um STE
convencional. Há teste explícito de igualdade do valor hard.

Transmitem-se os **dois streams de índices**, não um stream completo de todos
os canais mais uma correção gratuita. FNE1 global é transmitido uma vez,
com restauração de escapes e modo literal comum para constantes.
Paridade é local ao tile: linha 0 é âncora; shape e mapas fixam a interpretação.
Dimensões ímpares/canal único têm caminho explícito.

Formato **FJH1**, dois streams **FAK1** existentes (range/fixed fallback).
Registros de contagens exatas dos dois livros, ajustados somente no treino,
são comprimidos/transmitidos e cobrados. Não serializar tabelas condicionais
Huffman que esse coder não usa. A referência VQ madura é **medida novamente com o mesmo
FAK1 e registro compacto**, evitando atribuir ao H2 a economia de tabela de
um comparador antigo mais pesado. Cada livro possui namespace de identidade próprio.

Receptor separado é carregado de exportação real e reconstrói somente de índices,
pesos/registro e side. O replay precisa ser bit a bit igual no float32 físico
ao caminho hard, exceto pelo modo literal, que precisa ser bit-exato ao alvo.
O export completo dos dois backbones também contém encoders; seu custo é
discriminado, não chamado de menor receptor possível.

<a id="report-97-3-dados-perda-e-dose"></a>

#### 3. Dados, perda e dose

- 5.600 janelas dos 44 arquivos de ajuste do fold 0;
- excluir dos batches novos todos os 43 arquivos de checks P1;
- mesma seed **2115**, ordem de batches e uma janela nativa por update;
- calibração dos livros: uma janela central por arquivo de ajuste, 44 arquivos;
- checks do piloto: dez arquivos/janelas, um por configuração, quatro famílias;
- seleção prospectiva: `macro_famílias NRMSE*² + 0,1*bpc_stream/0,8`, incluindo inicialização;
- após o treino, ampliar automaticamente para os 129 checks/43 arquivos,
  sem retreino/reseleção e com referência prática no mesmo contrato.

Esses checks são desenvolvimento, historicamente vistos pela origem; não são
aquisições independentes. Não ler sinais de validação/teste ou EDAS reservado
para decidir esse piloto. Metadados físicos/mapas e escalas de P0 permanecem.

Loss comum: **0,5 MSE + 0,5 L1 mascaradas + média das duas perdas VQ**.
Não é uma loss RD explícita; índices nominais têm a mesma grade por ramo entre
os braços, mas **bytes completos reais** decidem seleção/gates.
AdamW novo nos dois braços, LR `1e-4`, weight decay `1e-5`, clip global 1, FP32,
TF32 desligado, algoritmos CUDA determinísticos. Sem scheduler, refresh,
novo dataset, loss familiar ou quantizador diferente combinado ao contraste.
Momentos AdamW não são herdados da origem; livros/contadores não podem aliasar.

<a id="report-97-4-orçamento-e-continuação-automática-limitada"></a>

#### 4. Orçamento e continuação automática limitada

1. **Smoke:** dois updates por braço, primeiro no maior tile e depois no menor.
   Verificar memória, gradientes finitos/não nulos do preditor, livros sem alias
   e replay real. Comparar também dois updates contínuos com um update + resume
   até dois: parâmetros, momentos e RNG devem ser idênticos. Avaliações preservam
   RNG de treino durante a construção do receptor. Sem significado científico dos scores.
2. **Piloto:** **8.192 updates por braço**, checks a cada 2.048.
3. Autorizar extensão **somente se todos** os critérios abaixo passarem:
   - selecionado H2 melhora ≥1% contra sua inicialização;
   - endpoint H2 melhora ≥0,2% contra o bloco anterior (6.144 → 8.192);
   - NRMSE* macro do selecionado H2 fica até 3% acima do independente selecionado;
   - stream H2 selecionado custa até 105% do independente selecionado;
   - perplexidade média de cada livro no último bloco é ≥16;
   - mesma cobertura de células e ausência de falha numérica/replay.
4. Se elegível, continuar **os dois braços até 32.768 updates totais por braço**,
   preservando optimizer/RNG/seleção acumulada. Não são 32.768 extras.
5. Independentemente da extensão, executar avaliação ampliada train-only dos
   selecionados/endpoints e da referência madura; depois decidir outra etapa.

Teto: **65.536 updates científicos no total**, incluindo o piloto, mais oito
updates técnicos entre smoke contínuo e teste de resume. Não há extensão além disso ou promoção automática.
O gate é investimento em aprendizagem, não aprovação das guardas finais.
A confirmação em outra seed/coorte depende de ganho real por bytes, grupos e
guardas físicas, não só do gate de extensão.

O usuário autorizou essa avaliação/continuação autônoma. O supervisor registra
o gate e só prossegue quando elegível; não espera uma confirmação adicional
para uma transição já prevista, nem roda indefinidamente.

<a id="report-97-5-implementação-artefatos-e-supervisão"></a>

#### 5. Implementação, artefatos e supervisão

- `finetuning/joint_anchor_vq.py`: representação conjunta, STE hard exato e loader próprio.
- `compression/joint_anchor_vq_stream.py`: receptor FJH1/FAK1 e registros cobrados.
- `finetuning/joint_anchor_evaluation.py`: calibração train-only, replay e referência pareada.
- `finetuning/run_joint_anchor_study.py`: treino, seleção e checkpoints/RNG atômicos.
- `finetuning/joint_anchor_gate.py`: gate de extensão, sem promoção.
- `finetuning/evaluate_joint_anchor_study.py`: expansão train-only sem updates.
- `tools/run_joint_anchor_campaign.py`: smoke → piloto → gate → eventual extensão
  → avaliação ampliada, esperando processos em vez de consultar repetidamente logs.

Destinos: `results/experiments/joint_anchor_vq_fold0_20261115/`.
GPU 0 independente; GPU 1 H2. Os mesmos dois dispositivos são usados no postflight.
Cada avaliação registra frames/tabelas/identidade; são preservados melhor export
e endpoint/optimizer. Não se guardam exports completos de todos os intermediários
como se fossem receptores mínimos. Scores intermediários não são promoção.
Código/documentos são sincronizados com backup de divergências remotas.

**Estado no registro:** implementação e testes locais; campanha ainda não lançada.

<a id="report-97-6-integração-passada-e-piloto-iniciado"></a>

#### 6. Integração passada e piloto iniciado

O supervisor foi lançado no destino registrado após **422 testes locais passarem**.
Os smokes nas duas GPUs passaram, incluindo o maior tile e o canal mínimo.
O teste contínuo versus retomado confirmou, em ambos os braços:

- parâmetros exatamente iguais;
- momentos AdamW exatamente iguais;
- RNG Python, CPU e CUDA exatamente iguais.

Evidência: `RESUME_INTEGRATION.json` na raiz da campanha. Os pesos técnicos
duplicados ficam preservados no remoto, sem necessidade de copiá-los todos para
o armazenamento local.

**Piloto científico iniciado automaticamente após essa integração:**
GPU 0 independente, PID inicial **714655**; GPU 1 H2, PID inicial **714656**,
8.192 updates por braço. Os comandos estão em `LAUNCH_pilot_*.json`.
O gate de extensão e o postflight ampliado permanecem automáticos dentro do
teto registrado, não exigem uma nova pausa por etapa e não promovem modelos.
Nenhum resultado científico desse piloto está disponível nesta atualização.

<a id="report-97-continuação-concluída"></a>

##### Continuação concluída

O gate passou e ambos os braços completaram 32.768 updates; o postflight ampliado
também terminou. Resultados, limites e hipótese seguinte estão no
[relatório 98](93-NEURAL_ANCHOR_INNOVATION_STUDY.md#report-98).
O registro anterior preserva o estado do lançamento, não indica treino pendente.

---

<a id="report-98"></a>

## Registro histórico 98 — H2 conjunto: mecanismo positivo, fronteira prática ainda não alcançada

Fonte original preservada no arquivo: `98-JOINT_NEURAL_ANCHOR_INNOVATION_RESULTS.md`.

<a id="report-98-h2-conjunto-mecanismo-positivo-fronteira-prática-ainda-não-alcançada"></a>

**15/11/2026 UTC.** Resultado completo do [protocolo 97](93-NEURAL_ANCHOR_INNOVATION_STUDY.md#report-97).
Sem acesso a waveforms de validação/teste ou EDAS reservado; sem promoção.

<a id="report-98-1-execução-e-integridade"></a>

#### 1. Execução e integridade

O piloto terminou e passou **todos os critérios registrados de extensão**:
aprendizagem contra inicialização e último bloco, proximidade ao independente,
bytes, saúde dos dois livros e cobertura. O supervisor continuou ambos os braços
até **32.768 updates totais**, não 32.768 extras.

- **65.536 updates científicos** entre dois braços, mais oito updates técnicos
  dos testes contínuo/retomado.
- **2.599.675.000 células por braço**, mesma sequência e máscaras.
- Fonte herdada: VQ denso fold 0; dois backbones por braço, com preditor adicional
  ativo apenas em H2. Pré-treino e exposição histórica permanecem declarados.
- Tempo somado dos loops piloto+extensão: **~2.661,8 s independente /
  ~2.672,2 s H2**, incluindo checks no loop, sem toda a inicialização/postflight.
- Parâmetros, momentos e RNG passaram identidade contínuo/resume no preflight.
- Replay físico do receptor separado teve diferença máxima **zero** nos checks
  do treino e nas avaliações ampliadas.
- Não houve colapso aparente dos livros: último registro de perplexidades
  âncora/inovação ~571/567 no independente e ~591/513 no H2. São médias por batch,
  não perplexidade global de um corpus concatenado.

Artefatos: `results/experiments/joint_anchor_vq_fold0_20261115/`, com
`EXTENSION_GATE.json`, logs, protocolos, histórico, checkpoints/otimizadores,
registros, frames, `TRAINING_COMPLETE` e `COMPLETE`.
Os artefatos científicos foram copiados localmente; exports técnicos duplicados
de smoke permanecem preservados no remoto.

<a id="report-98-2-seleção-e-avaliação-ampliada-congelada"></a>

#### 2. Seleção e avaliação ampliada congelada

O independente selecionou **30.720**; H2 selecionou **16.384**.
Não reescolhemos pelos 129 checks ampliados. As curvas H2 já não melhoraram
consistentemente depois do selecionado; não justificar outra extensão apenas
pela esperança de mais tempo.

Mesmo conjunto ampliado: **129 janelas, 43 arquivos, dez configurações,
quatro famílias, 12.762.000 células**. Todos pertencem ao treino histórico;
não são confirmação independente.

| Modelo | Update | NRMSE* macro famílias ↓ | Stream bpc ↓ | Stream total |
|---|---:|---:|---:|---:|
| Independente selecionado | 30.720 | 0,679812 | 0,761686 | 1.215.080 B |
| H2 selecionado | 16.384 | **0,628995** | **0,738904** | **1.178.737 B** |
| Independente endpoint | 32.768 | 0,679748 | 0,761683 | 1.215.075 B |
| H2 endpoint | 32.768 | 0,634683 | 0,748785 | 1.194.499 B |
| VQ denso maduro, FAK1 comum | — | **0,491288** | **0,676730** | **1.079.553 B** |

A taxa inclui frames, registro de contagens comprimido e mapas/transporte
necessários; pesos compartilhados ficam discriminados. FAK1 foi aplicado também
à referência madura: não comparar contra tabelas Huffman condicionais antigas
mais pesadas para fabricar uma vitória H2. A fidelidade da referência permaneceu
igual à do relatório 94; mudou o custo de serialização realmente medido.

<a id="report-98-3-o-mecanismo-h2-funcionou-contra-seu-controle"></a>

#### 3. O mecanismo H2 funcionou contra seu controle

O selecionado H2 reduziu **~7,48% do erro e ~2,99% dos bytes** contra o
independente selecionado. Melhorou nos **dez grupos**:

| Família | Independente | H2 |
|---|---:|---:|
| Doped | 0,730443 | 0,697951 |
| Multicore | 0,687774 | 0,650593 |
| Multimode | 0,714265 | 0,642262 |
| Single-mode | 0,586767 | 0,525174 |

Isso passa a condição escalar/configurações de investimento **contra esse
controle**. Sustenta a utilidade da predição/inovação conjunta, não atribui
isoladamente o ganho à CNN do preditor versus uma média fixa treinada.
Amplitude/fase/caudas por canal e suas coberturas estão nos arquivos por grupo.
Não declarar aprovação de todas as guardas físicas por essa tabela.

Uma auditoria pareada dos registros ampliados confirmou essa distinção:
contra o independente, H2 teve **um dos 43 arquivos** com NRMSE* maior e
**cinco arquivos** com p90 de canais maior, embora as dez macros por grupo tenham
melhorado. Entre 30 pares configuração×banda, houve duas regressões de erro
espectral e 14 na distância da amplitude RMS a 1. Os dois aumentos descritivos
de fase têm coberturas explicitadas; suportes de bins podem diferir e não
certificam uma guarda de fase comum. Artefato: `PHYSICAL_CONTROL_COMPARISON.json`.

<a id="report-98-4-por-que-ainda-não-promover-ou-iniciar-confirmação-p3"></a>

#### 4. Por que ainda não promover ou iniciar confirmação P3

Contra o VQ maduro reserializado, H2 permanece **~28,03% pior em erro** e
**~9,19% maior em bytes**, com erro pior nos dez grupos.
Os dois backbones H2 também custam mais pesos compartilhados: export completo
**223.808.607 B**, incluindo encoders, não o menor receptor possível.
O ganho contra o controle não deslocou a fronteira prática.
Contra a referência, os 43 arquivos têm NRMSE* maior e 42 têm p90 maior;
`PHYSICAL_REFERENCE_COMPARISON.json` registra também bandas/amplitude/fase.

Logo, **não promover, não iniciar P3 e não estender esta campanha além de
32.768 por braço**. Preservar a evidência positiva do mecanismo e o negativo
frente ao produto de referência, sem confundi-los.

<a id="report-98-5-próxima-hipótese-observação-do-encoder-não-outra-dose-igual"></a>

#### 5. Próxima hipótese: observação do encoder, não outra dose igual

Nos dois braços desta campanha, cada encoder começou observando só uma fase
espacial: pares ou inovação dos ímpares. O VQ completo observa os canais
adjacentes antes da redução de resolução. A perda de contexto físico na análise
é uma explicação testável para parte do déficit; **ainda não foi demonstrada
causalmente por estes resultados**.

O próximo contraste deve oferecer as duas fases adquiridas ao encoder, mantendo
os mesmos livros, grades, decoder, preditor e stream. Isso não oferece dados
gratuitos ao receptor: informação adicional só pode chegar por índices pagos.
Uma conexão de contexto de encoder inicialmente zero permite partir da mesma
função H2 selecionada e comparar contexto desligado/ligado com o mesmo budget.
Não combinar essa hipótese com nova normalização, coder ou padding.

Também registrar a diferença de posições/caudas de padding dos dois ramos;
não atribuir toda a diferença de bytes ao prior ou à observação do encoder.
O protocolo desse contraste será separado antes de seu treinamento.

<a id="report-98-auditoria-de-posições-e-framing-sem-inferência"></a>

##### Auditoria de posições e framing, sem inferência

`finetuning/audit_joint_anchor_wire.py` verificou os CRCs dos frames existentes
e decompôs os custos do selecionado H2 e da referência:

| Quantidade | VQ completo | H2 |
|---|---:|---:|
| Posições de índice | 864.000 | 936.000 |
| Bits de payload por posição | 9,80174 | 9,59210 |
| Framing e side dos frames | 14.448 B | 47.361 B |

H2 transmite **8,33% mais posições** nesse layout de ramos/padding, apesar de
ter menor custo por símbolo. Também paga mais cabeçalhos. Isso explica parte
da taxa, não o déficit de fidelidade. Não implica que posições possam ser
descartadas sem erro: padding/latentes participam da síntese e normalizações.
Arquivo: `PADDING_WIRE_AUDIT.json`. O próximo contraste mantém esse layout para
não confundir mudança de observação do encoder com economia de padding.

<a id="report-98-6-preservação-do-experimento"></a>

#### 6. Preservação do experimento

Antes de ampliar os loaders/runner para a hipótese seguinte, os fontes listados
no protocolo foram conferidos por SHA-256 e preservados em
`SOURCE_SNAPSHOT/`, incluindo vendor e supervisor. O manifesto guarda os hashes.
Os resultados e pesos de 97 não são reinterpretados por loaders novos.
Uma reprodução do código histórico deve usar esse snapshot e um destino novo;
as guardas de resume continuam rejeitando mudança de código/protocolo.

O próximo contraste foi registrado no
[relatório 99](94-ENCODER_OBSERVATION_STUDY.md#report-99).
