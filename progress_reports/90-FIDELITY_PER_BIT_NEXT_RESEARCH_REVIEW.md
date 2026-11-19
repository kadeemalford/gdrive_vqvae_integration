# Próxima direção: inovação espacial e representação neural multirresolução

**Revisão: 14/11/2026, posterior ao encerramento do relatório 89.**

## 1. Recomendação executiva

**Minha principal aposta é um codec neural com transformação espacial reversível
e detalhes explicitamente codificados, treinado para taxa–distorção.** Uma
implementação por *learned lifting* é a primeira candidata. A alternativa mais
próxima é predição neural de canais a partir de âncoras já decodificadas, seguida
da codificação da inovação. São hipóteses, não ganhos demonstrados em FEBUS.

O investimento deve começar por três entregáveis:

1. **Uma preparação canônica verificada, acompanhada de um contrato prospectivo
   de métricas e taxa que funcione em sinais constantes, canais fracos e aquisições
   completas.** O pré-processamento recomendado está explicitado na seção 4.2.
2. **Um diagnóstico de dependência espacial com contexto efetivamente disponível
   ao decoder**, incluindo os custos de transmitir esse contexto e os detalhes.
3. **Um único contraste estrutural com oportunidade real de aprendizagem**, em
   vez de uma nova sequência de adaptadores curtos da CNN transferida.

Em paralelo, recomendo ampliar **aquisições FEBUS independentes e regimes de
excitação**, antes de investir novamente em pré-treino indiferenciado de dados
externos. QuantX e PoroTomo já foram testados; seus resultados não justificam
repetir a mesma ponte.

A perspectiva central é preservar também o ruído **explorando a parte dele que
é redundante entre canais**. A realização da inovação que continuar imprevisível
precisa de bits. Nenhuma arquitetura remove essa necessidade por ser maior.

## 2. Base desta revisão

Foram consultados README, roadmap completo, os dois checkpoints de progresso,
relatórios de resultados e protocolos relevantes, com aprofundamento em 7,
77–80 e 83–89; implementações de arquitetura, geometria, quantização, normalização,
seleção e transporte; artefatos locais das campanhas de representação e produto VQ;
documentação da pasta DAS via MCP; e as fontes externas da seção 10.

A revisão também executou uma **auditoria descritiva CPU**, sem inferência ou
treinamento neural:

- recontagem do manifesto espacial do fold 0;
- 20 janelas de treino, de 20 arquivos e dez configurações, escolhidas por
  metadados antes da leitura: dois IDs lexicográficos por grupo, maior tile com
  pelo menos 16 canais e primeira janela desse tile;
- leitura dos atributos geométricos dos HDF5 correspondentes;
- conferência da janela constante apontada no relatório 89.

Artefatos locais:

```text
results/experiments/research_review_20261114/
  audit_review.py
  protocol.json
  audit.json
```

Reprodução:

```bash
.venv/bin/python results/experiments/research_review_20261114/audit_review.py
```

Essa amostra favorece tiles maiores e não estima a distribuição integral do
corpus. Os descritores centram canais para análise, sem modificar os alvos.
Não foram abertos sinais de validação/teste ou dos grupos EDAS reservados.
O MCP foi usado para listar a raiz e consultar README, diagrama e extrator;
isso não constitui um novo censo numérico dos HDF5 remotos.

## 3. O que o projeto já demonstrou

### 3.1 A percepção sobre comprar fidelidade com bits é correta, mas incompleta

| Evidência | Leitura para a próxima decisão |
|---|---|
| Relatório 22: RMSE ~0,810 → 0,720 à mesma taxa nominal | Houve ganho importante de otimização/adaptação sem mais índices. LR e exposição não foram isolados naquele contraste. |
| Confirmação A/B/C: 0,457670 → 0,416764 à mesma taxa nominal, relatório 64 | Outro avanço de ~8,94% de RMSE, com regressões localizadas. O problema não é exclusivamente quantidade de bits. |
| Streams compactos: ~166–173×, correlação ~0,73–0,75; densos: ~44–46×, ~0,89–0,90 | A grande diferença recente de fidelidade acompanha mais informação transmitida. Modelos e coortes diferentes entre folds; não é uma curva de uma rede única. |
| CNN ~28M → 96,5M: ~1,51% menos RMSE, relatório 66 | Aumentar largura isoladamente tem baixa prioridade. |
| E1 com guardas físicas + afim: até ~3,70% no desenvolvimento, ~0,19% no stress SMF | Existe lacuna de codificação, mas não evidência de que destilar essa busca produzirá o salto generalista. |
| FSQ residual e seleção híbrida, relatórios 70–73 | Melhoram a base comprando bits; vantagem sobre controles a bytes pareados é pequena. |
| Escalar/RD, prior e contexto, relatórios 78–88 | Houve testes reais dessas famílias. A proposta “usar taxa–distorção/atenção/hyperprior” precisa explicar a diferença concreta. |
| Produto VQ, relatório 89 | O piloto terminou; a fase longa não começou. A seleção foi dominada por alvo constante. Não usar o ranking como refutação conclusiva de PQ. |

As razões acima usam streams históricos com pesos compartilhados separados.
O transporte explícito de outras campanhas muda a taxa: por exemplo, o
relatório 78 mede ~0,261/~0,790 bit/célula no contrato comum compacto/denso.

### 3.2 Há uma concentração de hipóteses no mesmo mecanismo

A CNN nativa usa convoluções com stride, GroupNorm, um dicionário VQ de 2.048
vetores de dimensão 128 e uma única grade de índices. A anisotrópica foi uma
mudança real de blocos, mas manteve o gargalo de grade única. O escalar E2 também
reutiliza as transformadas residuais nativas, com largura menor e outro quantizador.

Assim, o histórico não esgotou arquiteturas neurais: ele caracterizou
extensivamente **uma família de análise/síntese e suas adaptações**.

Dois pontos merecem atenção:

- **Mais canais latentes contínuos não equivalem a mais informação transmitida.**
  Depois do VQ, cada posição continua identificada por um símbolo.
- **O contínuo não fornece um teto causal confiável da qualidade.** O bypass
  altera a distribuição recebida pelo decoder; o histórico multimode mostrou isso.
  Tampouco há prova de que todo erro anterior à quantização seja inevitável.

Fontes de código: `../smartgrids/vqvae/train_vqvae.py`,
`finetuning/{model_factory,latent_geometry,anisotropic_vqvae,scalar_rd}.py`.

## 4. Prioridade zero: preparação dos dados e decisão experimental bem definidas

### 4.1 Resolver o domínio das métricas, prospectivamente

Em `finetuning/evaluate_checkpoints.py:33–38`, `relative_rmse` divide pelo RMS
do alvo com epsilon `1e-12`. A janela zero do relatório 89 foi confirmada:
10 × 1.000 valores float32 exatamente zero. Um erro bruto ~5e-9 tornou-se uma
razão normalizada ~5e9, responsável por mais de 99,99998% do macro do seletor.

`physical_validation.py` possui outro denominador (`+1e-6`) e pula canais com
desvio ≤1e-6 em determinadas métricas. Esses pisos têm efeitos diferentes
conforme a unidade e a escala; não constituem tolerâncias físicas universais.

**Proposta para uma nova versão do protocolo:**

1. Agregar primeiro SSE, contagem e energia de referência por canal/aquisição,
   e só então formar razões. Publicar também as macros entre configurações/famílias.
2. Definir uma escala de referência física positiva por produto/regime, a partir
   da aplicação ou de uma calibração de treino congelada. Por exemplo,
   `NRMSE* = RMSE / max(desvio_do_alvo, escala_de_referência)`.
   A definição e a unidade dessa escala fazem parte da métrica.
3. Manter RMSE absoluto, viés/DC, erro máximo ou quantis de erro e cobertura dos
   canais fracos. Um piso reduz a explosão numérica; o erro absoluto impede que
   isso esconda uma falha física.
4. Tratar correlação/fase sem energia suficiente como **indefinidas**, com
   contagem de cobertura, em vez de transformá-las automaticamente em zero/sucesso.
5. Validar a métrica com identidade, alvo constante, constante perturbada, canais
   fracos e conversão coerente de unidades antes de selecionar outra campanha.

Isso exige protocolo novo, mantendo os scores e decisões antigos arquivados.
Não se trata de trocar epsilon até o produto VQ passar. A nova seleção também
deve ter representatividade de arquivos/regimes, além de contagem de janelas.

Uma ablação pequena de engenharia é **modo literal para constantes exatas**:
tipo, shape e valor/bits originais, com custo cobrado. Beneficia integridade e
evita reconstruir artificialmente zero por uma CNN; não explica a fidelidade
do restante do corpus. Aplicá-lo igualmente aos codecs comparados.

### 4.2 Pré-processamento: o que aproveitar e o que acrescentar

**Complemento solicitado após a revisão inicial:** recomendo uma etapa explícita
de preparação antes de P1/P2. Parte já estava incluída no contrato de alvo e nas
hipóteses de representação; o histórico também resolveu problemas importantes.
O trabalho prioritário agora é consolidar uma entrada comum para os novos
codecs e verificar as condições específicas de seus operadores.

#### O que já foi realizado

| Tratamento | Evidência e consequência |
|---|---|
| Auditoria de extremos, normalização com estatísticas em float64 e escape exato | Relatórios 7–8 e 32–35. Reutilizar a proteção versionada; os escapes são bit-exatos, mas o roundtrip das células ordinárias normalizadas pode ter erro de arredondamento. |
| Mapas físicos, faixas contíguas e tiling com cobertura declarada | Relatórios 12 e 60–62. As 28.780 janelas foram verificadas contra os brutos; não há motivo para refazer essa preparação sem uma alteração concreta. |
| Normalização global versus por canal, com restauração e side contabilizados | Relatórios 11–14. A versão por canal melhorou métricas de cauda, mas piorou RMSE macro e mais que dobrou o payload parcial no contraste do relatório 14. Não é uma melhoria pendente que deva virar default. |
| Diagnóstico temporal contínuo e preparação externa | Relatórios 79–80. Há ferramentas para extrair contexto real dos HDF5 e conferir clocks; os NPZ espaçados não podem ser concatenados como uma sequência contínua. |
| Preservação dos bits de NaNs e cobertura integral experimental | Relatório 79, formatos em `compression/numeric_restoration.py` e `partition_bundle.py`. São caminhos específicos; sua existência não torna toda preparação histórica compatível com não finitos. |

A política `persistent-amplitude-v1` permanece uma heurística de cobertura,
com exclusões registradas. Ela não identifica definitivamente todo canal inválido.
Para o contraste principal, manter o alvo retido comum. Uma extensão de cobertura
integral usa contrato próprio e os mesmos canais em todos os braços.

#### Preparação recomendada para a próxima campanha

1. **Manifesto físico explícito.** Consolidar variável/unidade, dtype de origem,
   sample rate armazenado, dx, GL, pulse width, derivation time, coordenadas e
   proveniência por aquisição. O preparo atual registra dt/dx, mas não reúne
   todos os campos necessários ao condicionamento H3. Para novos formatos,
   selecionar dataset e orientação pelos metadados/esquema: o fallback ao primeiro
   dataset e a inferência de eixos pelo tamanho em `prepare_finetune_samples.py`
   foram auditados no corpus existente, não validados para qualquer fonte externa.
   Preservar a precisão de origem antes de centralizar/escalar.

2. **Máscaras com semântica separada.** Distinguir amostra física ordinária,
   valor preservado por escape, região fora da cobertura e padding artificial.
   Zeros reais e canais de baixa energia continuam sendo dados. Na rede,
   preenchimento determinístico de posições sem alvo não deve contribuir para
   a loss ou ser contado como célula adquirida; sua influência nas ativações
   também precisa ser medida. Máscaras deriváveis de shape/mapa não exigem bytes
   duplicados, mas toda decisão adicional necessária ao decoder entra no stream.

3. **Tiling compatível com a geometria e com a inversa.** Preservar regiões
   contíguas, coordenadas de origem e comprimentos reais. Para lifting/âncoras,
   fixar paridade, extensão nas bordas e tratamento de canais/tempos ímpares;
   validar os tiles pequenos e os resíduos de divisão. Separar lacunas reais
   de padding evita predizer através de vizinhanças físicas inexistentes.
   Isso é integração do novo codec, não repetição da busca de orientação/padding
   já encerrada no relatório 79.

4. **Continuidade temporal quando necessária.** Para transporte por segundo,
   avaliação pooled ou contexto entre blocos, recortar diretamente dos HDF5
   com timestamps, offsets e convenção de overlap explícitos. O preparo atual
   usa `linspace`, produzindo, por exemplo, inícios 0/1052/2105 para janelas
   de 1.000 amostras. Interpolar seus intervalos ou simplesmente juntá-las
   inventaria continuidade. Overlap de clock também não garante valores
   idênticos: amostras diferentes continuam preservadas se o contrato cobre
   todos os valores armazenados. Um transporte de 1 s pode conter várias
   janelas de 50 ms; aumentar o contexto neural é uma ablação separada.

5. **Roundtrip de preparação antes do modelo.** Usar o caminho identidade
   `raw → normalização/side → restauração`, medindo erro absoluto e relativo
   em regiões ordinárias, canais fracos e constantes, além da identidade dos
   bits escapados. Reutilizar os testes/auditorias existentes e ampliar apenas
   para o novo formato, tiling ou fonte. A normalização deve usar os parâmetros
   efetivamente disponíveis no receptor. Estatísticas aprendidas ficam no
   treino; estatísticas calculadas por aquisição na codificação são permitidas,
   desde que sua representação seja transmitida e cobrada.

6. **Inventário de constantes e regimes, sem exclusão por dificuldade.**
   Registrar por aquisição a incidência de constantes, não finitos, extremos,
   faixa dinâmica e tamanho de tiles. Usar isso para cobrir casos nas verificações
   e impedir que a seleção seja dominada por denominadores patológicos.
   Aplicar o eventual modo literal da seção 4.1 de forma comum. O inventário
   não justifica remover canais fracos nem retomar automaticamente a ablação
   de sampler por células, já realizada.

**Entrega concreta de P0:** manifesto e política versionados, mapas/máscaras,
relatório de cobertura, erro do roundtrip e bytes de side. Para dados existentes
sem mudança de entrada, basta verificar compatibilidade e reutilizar evidências.
Se uma nova preparação alterar a entrada, comparar primeiro os pesos congelados
sobre **o mesmo alvo físico**, para separar ganho de preparação de ganho aprendido.
Depois, o contraste arquitetural usa a preparação comum.

#### Transformações adicionais: onde entram na hipótese

O lifting/subbandas de H1 já é uma forma de **pré-condicionamento para compressão**,
mas pertence ao codec: sua inversa e todos os detalhes necessários fazem parte
da reconstrução. O controle de transformada fixa previsto em P2 permite medir
quanto vem dessa organização e quanto vem dos operadores neurais.

Manteria a normalização externa global com proteção como controle inicial.
Normalização por canal já foi testada; normalização interna local é a ablação
distinta H3. Uma futura remoção de tendência ou componente comum só preserva o
objetivo se transmitir/reconstruir também essa componente e cobrar seu custo.
Isso vale igualmente para whitening ou outra transformação reversível: diminuir
a loss no domínio transformado não demonstra menor erro físico nem menos bytes.

Denoising, clipping, filtros que eliminem bandas e decimação simples alterariam
o alvo de reconstrução. Para este projeto, o pré-processamento deve preservar
o conteúdo medido e melhorar sua representação/acessibilidade. Não há evidência
para supor que uma nova “limpeza” genérica resolverá a limitação atual.

### 4.3 Fixar horizonte e contabilização de taxa

O bundle de 18 checks do relatório 89 cobra ~904,7 kB de metadados comprimidos
para ~29–32 kB de frames. O código em `codec_wire_evaluation.py:57–61` inclui
manifesto/mapas completos da preparação, não somente os checks transmitidos.

Isso é uma contabilidade reproduzível de arquivo de pesquisa, mas a fração de
economia de um piloto depende fortemente desse custo fixo. Para estudar o codec:

- definir transporte comum por aquisição ou bloco contínuo, por exemplo 1 s;
- contabilizar todas as janelas/caudas, mapas necessários, escalas, escapes e
  inicializações desse contrato;
- medir custo variável e custo de instalação do stream;
- publicar também o arquivo científico completo e os pesos compartilhados;
- distinguir exportação integral do modelo de uma exportação mínima do receptor.

Formalmente, `B(N) = B_setup + B_payload(N) + B_side(N) + B_framing(N)`.
Uma curva por horizonte mostra quando a economia se paga. Projeção de
amortização continua sendo projeção, até medir o volume correspondente.

### 4.4 Separar triagem de aprendizagem de aceitação final

O arquivo novo de um modelo não precisa vencer uma referência com ~98 mil
updates FEBUS e pré-treino OOI após alguns centenas/milhares de updates para
merecer investigação. Alguns negativos curtos rejeitam uma **continuação
específica**, mas não estimam desempenho convergido.

Recomendo:

- smoke para integridade;
- orçamento mínimo para aprendizagem e calibração hard/bytes;
- critérios predefinidos de extensão por curva, saúde do quantizador e custo;
- comparação final com a fronteira prática;
- confirmação por aquisição/seed.

Os ≥5% de redução de RMSE ou ≥10% de economia continuam referências úteis de
investimento. Não são requisito de superioridade após cada bloco curto nem
tolerância física validada. Guardas antigas devem permanecer reportadas; uma
política futura de equivalência precisa ser definida antes dos resultados.

## 5. O sinal favorece uma hipótese espacial específica

O E0 histórico encontrou similaridade mediana adjacente ~0,621, temporal lag1
~0,096 e ~44,1% da potência em 5–10 kHz (relatório 78). O AR contínuo não
encontrou ganho consistente ao acrescentar lags de 100/200 ms (relatório 79).
Isso favorece começar pelo eixo espacial, sem afirmar independência temporal.

A auditoria atual acrescenta exemplos da **forma** dessa dependência:

| Exemplos de treino, duas janelas por configuração | Similaridade espacial lag1 | Lag5 | Similaridade temporal lag1 | Potência 5–10 kHz |
|---|---:|---:|---:|---:|
| Doped / 5m SMF | 0,594–0,606 | −0,136 a −0,117 | 0,045–0,074 | 41,2–45,1% |
| Multimode / Configuration 7 | 0,767 | −0,035 a −0,025 | −0,214 a −0,170 | 46,5–51,5% |
| Single-mode / Configuration 4 | 0,997 | ~0,990 | ~0,944 | 13,5–13,8% |

Similaridades são medianas dos cossenos entre pares de linhas centradas, não
coerência espectral estimada em aquisições longas. A coluna de potência usa
agregação energética e Hann; as agregações não são equivalentes.

No exemplo doped, cinco canais equivalem a GL=1 m, com dx=0,2 m. No multimode,
cinco canais equivalem a GL=20 m, com dx=4 m. O padrão é **compatível com uma
dependência influenciada pela resposta espacial instrumental**, mas não identifica
causalmente sua origem, nem prova que seja todo ruído ou que o mesmo operador
sirva para todas as fibras.

Na mesma amostra, há GL/dx de 5, 6,25 e 25. Uma CNN que vê apenas pixels não
recebe explicitamente essa diferença. O manifesto completo contém dx de
0,2 a 5 m; todos os datasets selecionados nessa preparação estão **rotulados
Strain Rate**, inclusive Ch1/Ch2. Isso torna a ponte a partir de fase bruta
fisicamente diferente do corpus atual, mesmo que o suporte geral aceite Strain.

### Diagnóstico adicional realmente útil

Fixar arquivos de treino e comparar, fora do trecho/arquivo usado no ajuste:

1. previsão com vizinhos originais — referência otimista;
2. previsão com âncoras quantizadas/decodificadas — condição implementável;
3. previsão apenas da reconstrução VQ — mecanismo próximo dos corretores já negativos;
4. efeito de dx, GL, pulse width, lag e banda física.

Medir o custo real de **âncoras + inovação + parâmetros + framing**, não apenas
R² ou queda de variância. Os ganhos de ~0–28% do probe sobre residual original
do roadmap não dão esse custo: o decoder não recebe aquele residual original.
PSD plana, rank e falha de um preditor também não medem entropia irredutível.

## 6. Hipóteses arquiteturais priorizadas

### H1 — Transformação neural reversível espacial/multirresolução

**Prioridade: principal aposta estrutural.**

Separar canais pares/ímpares e aplicar operadores neurais de predição e atualização:

```text
e = x_even
d = x_odd - P(e; metadata)
s = e + U(d; metadata)

decoder:
e_hat = s_hat - U(d_hat; metadata)
odd_hat = d_hat + P(e_hat; metadata)
```

Sem quantização, a transformação é invertível algebricamente; sua realização
numérica precisa ser verificada. Com quantização, o erro é medido após a inversa.
**Isso não promete codec lossless nem fidelidade gratuita.**

O stream transmite os coeficientes quantizados de `s` **e** `d`, em todos os
níveis, além dos parâmetros auxiliares necessários. O mecanismo esperado é
compactar a dependência espacial antes de pagar pelos detalhes, inclusive
componentes de ruído correlacionadas.

Desenho inicial:

- lifting espacial com operadores locais pequenos e inicialização estável;
- contexto temporal curto, com todas as amostras preservadas;
- se houver decomposição temporal, usar banco completo de subbandas/polifases,
  incluindo altas frequências; menor resolução de uma subbanda não é descarte;
- quantização escalar por subbanda e taxa diferenciável, com avaliação hard;
- prior inicialmente simples por subbanda; condicionamento entre subbandas
  somente depois de medir redundância líquida recuperável;
- perda no waveform reconstruído, com normalização/ponderação comum entre braços;
- toda configuração física usada pelo receptor deve constar do contrato.

O primeiro contraste pode usar metadados constantes e comparar **transformada
fixa versus a mesma transformada com passos neurais aprendidos**, mantendo coder,
quantização, alvo e loss. Condicionamento físico é ablação posterior. A referência
VQ madura permanece comparador prático em cada taxa.

**Diferença frente ao histórico:** multiescala dentro de blocos convolucionais
não é decomposição reversível com streams de detalhes em múltiplas resoluções.
O escalar E2 manteve a CNN de stride/GroupNorm e não testou esse mecanismo.
Também não é SVD por tile: os operadores são compartilhados e locais, sem
transmitir uma base espacial/temporal inteira para cada janela.

**Risco principal:** detalhes espacialmente imprevisíveis podem continuar caros,
e a inversa não linear pode amplificar erro de quantização. Por isso, medir
sensibilidade por subbanda, estabilidade e bits reais é parte do piloto.
Uma identidade contínua excelente, sozinha, não constitui resultado de compressão.

### H2 — Predição neural espacial em malha fechada + inovação transmitida

**Prioridade: alternativa a H1, se o diagnóstico com âncoras decodificadas for favorável.**

Codificar um conjunto de canais-âncora; prever os demais usando **as âncoras
reconstruídas**; transmitir a inovação que falta para reconstruir cada canal:

```text
anchors_hat = decode(anchor_stream)
prediction = neural_predictor(anchors_hat, geometry)
residual = original_other_channels - prediction
other_hat = prediction + decode(residual_stream)
```

A escolha de âncoras e a divisão do orçamento precisam ser fixadas ou codificadas.
Treinar encoder, preditor e representação da inovação conjuntamente, com
quantização no caminho usado em produção.

**Diferença frente ao refiner congelado:** não se paga uma base completa para
todos os canais e depois se acrescenta mais informação. Desde o começo,
o orçamento comum é repartido entre contexto e inovação; a base se organiza
para ajudar a representação condicional.

**Diferença frente ao prior de índices:** muda o conteúdo transmitido e a
síntese, não apenas a probabilidade atribuída aos mesmos IDs.

O controle deve codificar os mesmos canais independentemente, com taxa e
capacidade compatíveis. Avaliar propagação de erro, bordas e latência; preferência
inicial por grupos paralelizáveis, não um loop serial por célula a 20 kHz.
Não avançar com ganho observado apenas quando o preditor recebe âncoras originais.

### H3 — Condicionamento físico compartilhado e normalização interna

**Prioridade: ablações pequenas, após identificar mecanismo.**

Dois contrastes distintos:

1. **Metadados:** condicionar operadores/quantização em dx, GL, pulse width e
   produto, usando embeddings ou FiLM pequenos. Comparar ao mesmo modelo com
   metadados constantes. Família nominal sozinha é uma variável menos informativa
   que a configuração física; exigir teste com combinações reservadas e metadados
   ausentes. Não reamostrar a fibra para fingir uma geometria comum.
2. **Normalização interna:** testar alternativa local à GroupNorm que agrega
   sobre todo o tile, como normalização só entre features por posição ou blocos
   sem normalização espacial global. Isso é diferente da normalização reversível
   de entrada já testada. A melhora de preparação no alvo comum e a sensibilidade
   a tiles dão uma motivação, mas não provam que GroupNorm seja a causa.

Trocar GroupNorm rompe a função de pesos transferidos; um teste congelado após
substituição arbitrária não seria uma ablação válida. Dar adaptação/controlar
inicialização e manter a normalização externa idêntica.

Essas hipóteses podem melhorar robustez e amplitude sem mais side por janela
quando usam metadados já transmitidos. O ganho generalista permanece incerto.

### H4 — Codec neural com erro controlado e residual quantizado

**Prioridade: alta se “fiel” significar erro estrito; secundária para o ponto ultracompacto.**

O encoder reconstrói sua própria saída, mede o erro e codifica uma correção
quantizada suficiente para satisfazer um contrato físico. Um modelo neural
prediz/codifica essa inovação condicionado na base e nos resíduos já decodificados.

A literatura DLPR fornece um desenho concreto. A novidade frente ao refiner
FSQ é otimizar **o custo da correção necessária para uma precisão definida**,
com contexto causal reproduzível, em vez de esperar que um pequeno número fixo
de códigos adicionais melhore todos os canais.

Comparar a mesma fidelidade final e cobrar base, correção, tolerâncias e modelo.
A garantia precisa ser verificada depois do roundtrip, inclusive arredondamento
e restauração física. Um limite de RMSE por bloco não é limite ponto a ponto,
nem garante fase de componentes quase nulas.

Esse caminho pode exigir taxas muito superiores a 0,2 bit/célula. O preditivo
histórico a alta precisão, ~2,5–4,5×, já alerta para esse custo; ele não é um
limite inferior teórico. O objetivo desta hipótese seria deslocar a curva a
precisão fixada, sem prometer manter 170× ou 700×.

### H5 — Compressão conjunta de núcleos simultâneos

**Prioridade: oportunidade específica de multicore, condicionada à sincronização.**

Usar um núcleo/canal de aquisição já codificado como contexto para outro, ou
transformada neural multivariada com componentes compartilhadas e inovações.
Isso explora uma dimensão de informação diferente de atenção na matriz de uma
única aquisição.

Primeiro confirmar clocks, sobreposição física e alinhamento; nomes “C1 and C3”
e arquivos Ch1/Ch2 não bastam. Medir dependência fora dos trechos de ajuste.
Cobrar **o stream conjunto inteiro**, inclusive a referência. O ganho de um
núcleo condicionado não pode receber o outro gratuitamente.

É uma extensão de domínio, com fallback quando só existe um núcleo disponível;
não é uma solução sozinha para as quatro famílias.

### Alternativas de menor prioridade

RVQ/hierarquia latente ponta a ponta continua diferente do refiner congelado e
do PQ 32×64. Ainda assim, outra fatoração na mesma grade teria justificativa
menor que H1/H2. Para retomá-la, exigir distribuição multiescala de informação,
controle de taxa total e comparação com códigos planos, não só mais estágios.

Outro Transformer, Mamba/LSTM longo, INR por arquivo, MoE familiar ou rede maior
não recebe prioridade pelo nome da arquitetura. Contexto temporal longo requer
evidência condicional adicional; INR/adaptação por aquisição requer cobrança de
pesos e compute. GAN/LPIPS ou geração de ruído plausível não atendem à fidelidade
da realização observada. PINN não é parte desta recomendação.

## 7. Dados: investir em diversidade física e informação de aquisição

### 7.1 O problema não é simplesmente quantidade de NPZ

O corpus histórico tem 207 HDF5/23 grupos, mas a auditoria temporal estimou
~896 s na soma das uniões de clock por grupo. Não é duração independente exata.
O manifesto atual contém 28.780 janelas, incluindo tiling; isso não acrescenta
aquisições. Existem só três grupos doped e três multimode.

Priorizar **novas sessões/dias/remontagens**, não apenas mais recortes dos mesmos
tons. Um programa pequeno de aquisição poderia ser:

- quatro famílias;
- três sessões independentes por família, variando remontagem/dia/condição;
- três regimes de 60 s por sessão: fundo, excitação larga/transiente e mistura
  de tons/chirps com amplitude e fase variadas;
- sincronização com excitação de referência, mapa da fibra e parâmetros reais
  de aquisição, incluindo derivação, GL, dx, pulse width, potência e firmware;
- pelo menos uma sessão por família reservada antes da exploração.

Isso soma **36 min em 12 sessões**, como exemplo de planejamento, não tamanho
estatisticamente suficiente. Em 512 canais × 20 kHz × float32 seriam ~88,5 GB
de payload antes de container; ajustar duração/canais ao orçamento disponível.
Sessões realmente diversas podem valer mais que horas adicionais da mesma condição.

Fundo e ruído também são alvos de reconstrução. Gravações repetidas e referências
servem para entender componente comum, drift e inovação, não para substituir
o waveform por uma média mais fácil.

### 7.2 Fontes externas: priorização após os negativos existentes

| Fonte | Próximo uso defensável |
|---|---|
| Novas aquisições FEBUS do mesmo laboratório | Maior prioridade: preservam readout/bandas e permitem controlar diversidade, mapa e instrumentação. |
| EDAS do Drive | Útil para mudança de domínio. SMF já é desenvolvimento externo; Polyimide/Acrylate permanecem reservados segundo o histórico. Confirmar proveniência; não contam como validação das quatro famílias. |
| PoroTomo já preparado | Evidência de transferência condicional: ~4,54% menos RMSE com mesma dose FEBUS curta, mas mais bytes e pior que investir updates totais no alvo. Uma futura mistura por bandas/regimes seria outra hipótese, não continuação automática. |
| QuantX já auditado | Fase, 10 kHz e drift diferem do alvo. Ponte bruta já negativa; conversão/calibração teria de ser fisicamente fundamentada. |
| Ponte A8 / Mercerat et al., 2025 | Nova pista: FEBUS A1, strain-rate a 8 kHz, GL=4 m, campanhas de verão/inverno. Artigo declara dados mediante solicitação; análise publicada usa decimação para 500 Hz. Solicitar o bruto, termos de uso e metadados. Nenhum acesso aos waveforms foi confirmado aqui. |
| BFO / Forbriger et al., 2026 | Referência especialmente útil para acoplamento, calibração e ruído comum entre interrogadores. O produto público previamente levantado não equivale ao FEBUS 20 kHz. |
| PubDAS/EGS e fontes urbanas | Selecionar por readout, banda armazenada e independência. “Pulse rate 20 kHz” não significa saída a 20 kHz; intensidade não equivale a strain-rate. |

Um dataset amplo de classificação, denoising ou espectrogramas não é uma
substituição direta do waveform bruto. A busca atual não confirmou uma coleção
pública pronta que cubra simultaneamente as quatro famílias e o produto 20 kHz.

### 7.3 Dataset semissintético voltado à medição

Uma alternativa barata é combinar **fundo FEBUS de treino medido** com campos
sintéticos variados e uma resposta instrumental declarada:

```text
x_train = response(physical_wavefield; geometry, instrument) + measured_background
target  = x_train
```

Gerar ondas com diferentes velocidades/direções, múltiplas fontes, impulsos,
chirps, eventos fracos e mudanças de amplitude. A resposta pode aproximar a
diferença espacial sobre GL e a derivação temporal, quando apropriadas ao produto;
parâmetros incertos precisam ser randomizados e a aproximação documentada.
Ruído comum/espacial e heteroscedasticidade devem vir de evidência de treino.

O alvo continua sendo **o sinal composto completo**, não a versão limpa.
Separar sessões de fundo e seeds entre partições: reutilizar o mesmo ruído
debaixo de eventos diferentes pode criar vazamento. Evitar que um gerador
gaussiano simplificado substitua toda a diversidade real.

Contraste mínimo: FEBUS-only versus FEBUS+semissintético, mesma arquitetura e
orçamento; avaliar exclusivamente o ganho em FEBUS real reservado. O simulador
é um mecanismo de cobertura/testes, não evidência de generalização.

Também vale uma curva 25/50/100% dos **grupos de treino**, mantendo validação fixa
e documentando updates, células e passes. Ela distingue benefício de diversidade
de simplesmente repetir dados ou gastar mais compute.

## 8. Campanha sugerida

Os budgets abaixo são propostas de planejamento, sem execução automática.

| Etapa | Entregável e decisão |
|---|---|
| **P0 — preparação e contrato** | Preparação canônica da seção 4.2: manifesto físico, mapas/máscaras, tiling e roundtrip; continuidade quando necessária. Métricas prospectivas, casos constantes/fracos e transporte comum. Reutilizar a preparação auditada compatível; recomputações históricas recebem nome novo e interpretação retrospectiva. |
| **P1 — oportunidade espacial** | Dezenas de arquivos de treino, estratificados por geometria/regime; comparação original versus contexto decodificado; curva de custo de âncoras/detalhes. Nenhum ganho só de MSE vira economia de codec. |
| **P2 — H1 principal** | Transformada fixa versus learned lifting, mesmos coeficientes/coder/loss; VQ compacto/denso como referências práticas. Começar perto de 0,2 e 0,7–0,8 bit/célula em transporte comum. |
| **P3 — confirmação** | Segunda seed e rotação compatível, avaliação completa por configuração/arquivo, bandas, amplitude, fase e canais; somente para candidato que desloca a fronteira. |
| **P4 — dados** | Expansão FEBUS ou semissintético em ablação separada, sobre a arquitetura escolhida e seu controle. Reservar sessões externas desde o início. |

Para P2, sugiro planejar inicialmente **32.768 updates por braço**, com checks
de integração/saúde no início e uma extensão predefinida até 98.304 se a
aprendizagem justificar. O número deve ser ajustado no preflight ao batch,
células e custo; updates iguais não são FLOPs iguais. Uma GPU e braços
sequenciais são suficientes para o desenho, sujeitos à medição de memória.
O pré-treino OOI da referência prática deve continuar declarado.

O controle fixo de H1 faz parte da ablação da transformada; não abre uma campanha
de benchmarks clássicos nem substitui o objetivo neural. Não combinar de início
nova arquitetura, fonte, loss, scheduler e hyperprior.

**Critério final de investimento proposto:** ≥5% menos erro no score prospectivo
a bytes completos não maiores, ou ≥10% menos bytes a fidelidade equivalente,
com análise das quatro famílias e confirmação. Preservar a leitura das guardas
históricas; tolerâncias físicas novas devem ser fixadas previamente.

A curva deve trazer também piores grupos e custo de encoder/decoder. Intervalos
de incerteza usam aquisições/configurações como unidades; milhares de janelas
correlacionadas não criam milhares de réplicas.

## 9. O que esperar de alta compressão preservando ruído

Sobre float32, `R = 32/CR`. Portanto:

| Compressão | Bits por célula |
|---|---:|
| 40× | 0,800 |
| 160× | 0,200 |
| 320× | 0,100 |
| 700× | 0,0457 |

Em 700× há aproximadamente um bit para 22 valores, antes de reservar orçamento
para side. É uma meta muito agressiva para preservar uma realização de ruído.

Para uma fonte **hipotética gaussiana i.i.d.**, `D/variância = 2^(-2R)`.
Em 0,0457 bit/amostra, o RMSE normalizado ideal é ~0,969. Essa conta já apareceu
no relatório 77 e **não é um limite medido dos dados FEBUS**.

Como extensão ilustrativa: se uma fração `q` da potência fosse inovação
gaussiana independente e todo o restante fosse conhecido gratuitamente, ainda
teríamos `NRMSE ≥ sqrt(q) * 2^(-R)`. Para NRMSE de 0,1 a 700×, essa hipótese
exigiria `q ≤ ~1,07%`. Não foi estimado que o FEBUS satisfaça esse modelo.

Correlação espacial de ruído é justamente uma oportunidade: ruído pode ser
codificável. O objetivo do próximo diagnóstico é descobrir quanto dessa
dependência continua utilizável depois de transmitir contexto quantizado.
Matriz de covariância/curva gaussiana pode orientar experimentos, mas não é
certificado de impossibilidade nem lower bound empírico geral.

Um salto útil inicial seria obter fidelidade do ponto denso a uma taxa menor,
ou melhorar materialmente o compacto. Aproximar 700× deve ser consequência
de uma curva medida, não premissa que determine o resultado esperado.

## 10. Literatura e projetos que realmente informam a proposta

Busca dirigida, não revisão sistemática. Os resultados publicados não foram
reproduzidos neste repositório.

1. **Li, Naman e Taubman — Exploration of Learned Lifting-Based Transform
   Structures… (2024).** [Texto integral](https://arxiv.org/html/2402.18761v1).
   A fonte mais diretamente útil para H1: estrutura reversível, comparação
   de operadores e inicialização preservando uma transformada funcional.
   O artigo também mostra que profundidade adicional não garante ganho e
   discute amplificação de quantização. Seus números são de imagens naturais.

2. **Ma et al. — iWave++, End-to-End Optimized Versatile Image Compression
   With Wavelet-Like Transform.**
   [Artigo](https://ieeexplore.ieee.org/document/9204799),
   [projeto oficial](https://github.com/mahaichuan/Versatile-Image-Compression),
   [implementação PyTorch indicada pelos autores](https://gitlab.com/iWave/iwave).
   O README oficial foi consultado; implementação não foi executada/auditada
   integralmente nesta revisão. Aproveitar lifting/codificação de subbandas,
   adaptando eixos, unidades, shapes e loss para DAS.

3. **Meyer, Prativadibhayankaram e Kaup — Efficient Learned Wavelet Image and
   Video Coding (2024).** [Texto integral](https://arxiv.org/html/2405.12631v1).
   pWave++ oferece contexto em etapas paralelizáveis, importante para evitar
   custo serial por amostra. Tem medições de velocidade e trade-off de taxa
   em imagens; não prevê throughput ou economia em FEBUS.

4. **Bai et al. — Deep Lossy Plus Residual Coding, TPAMI 2024.**
   [Artigo/resumo](https://arxiv.org/abs/2209.04847),
   [código oficial](https://github.com/BYchao100/Deep-Lossy-Plus-Residual-Coding).
   Fonte para H4: base e residual, contexto e modos com erro limitado.
   O projeto fornece encode/decode separados. Dados de imagem inteiros e
   métricas do artigo exigem adaptação para float32 físico e escapes FEBUS.

5. **Zhu, Ranka e Rangarajan — Residual Modeling for High-Fidelity Learned
   Compression of Scientific Data (preprint 2026, v2).**
   [Texto integral](https://arxiv.org/html/2606.05389v2).
   NGLR condiciona predição em base reconstruída e vizinhos residuais já
   decodificados, cobrando pesos adaptados à coleção. Seu NRMSE usa range
   e garantia por bloco, não o denominador atual nem erro ponto a ponto.
   É referência de mecanismo, não evidência de transferência das taxas para DAS.

6. **Majumder et al. — Neural Residual Modeling… under Guaranteed Error Bounds
   (preprint 2026).** [Texto integral](https://arxiv.org/html/2609.23185v1).
   Examina RVQ + U-Net + correção PCA. Não priorizo copiar esse pipeline:
   corretores e fatores transmitidos já tiveram pouco retorno localmente.
   A estrutura do residual em simulações não demonstra previsibilidade do
   residual FEBUS a partir da reconstrução.

7. **Forbriger et al. — Cemented fibers as a testbed for DAS (2026).**
   [Resumo consultado](https://arxiv.org/abs/2603.24353).
   Evidência física de calibração/acoplamento e componente comum de ruído.
   Remover ruído comum, como numa análise instrumental, é diferente de
   codificá-lo e reconstruí-lo; aqui interessa a segunda opção.

8. **Mercerat et al. — Tracking the normal modes of an overpass highway bridge
   using DAS (2025).** [Texto integral](https://arxiv.org/html/2510.24212v1).
   Pista de aquisição FEBUS em duas campanhas, com bruto declarado a 8 kHz
   e análise a 500 Hz. Acesso mediante solicitação, não dataset público
   já disponível ou auditado.

9. **ZipDAS / AI4EPS.** [Projeto](https://github.com/AI4EPS/ZipDAS).
   Precedente DAS útil, já analisado no relatório 77. A consulta atual ao README
   não elimina as ressalvas anteriores sobre preparação, objetivo e splits.
   O ramo escalar local já explorou parte dessa direção; copiar defaults não
   constitui uma nova hipótese.

## 11. Decisão recomendada

**Priorizar preparação verificada + métrica válida → diagnóstico espacial com
contexto decodificado → learned lifting em taxa comum → confirmação, enquanto
se obtêm aquisições FEBUS mais diversas.**

Essa direção altera a forma de representar e transmitir dependências do sinal.
Tem motivação no conteúdo observado e uma diferença verificável frente aos
experimentos concluídos. A alternativa com predição/âncoras é apropriada se o
diagnóstico mostrar inovação compacta; o caminho de residual com erro controlado
ganha prioridade quando os requisitos de fidelidade forem estritos.

O relatório registra uma proposta e os descritores CPU desta revisão. Nenhum
modelo foi treinado, reclassificado como aprovado ou promovido.

### Continuação autorizada

Após esta revisão, o usuário autorizou implementar a sequência e usar as duas
GPUs remotas. A preparação e o diagnóstico inicial estão no
[relatório 91](91-PROSPECTIVE_PREPARATION_AND_SPATIAL_OPPORTUNITY.md);
o protocolo e o estado do contraste estrutural estão no
[relatório 92](92-LEARNED_LIFTING_STUDY.md#report-92).
A declaração acima descreve o momento da revisão inicial, não o estado posterior
dessas execuções.

## 12. Situação das propostas após as execuções

**Atualização de acompanhamento; não altera os protocolos ou resultados originais.**
Os links abaixo usam a numeração atual das etapas.

| Proposta | Evidência atual | Limite da conclusão |
|---|---|---|
| P0/P1: preparação, métricas e oportunidade espacial | [91](91-PROSPECTIVE_PREPARATION_AND_SPATIAL_OPPORTUNITY.md), concluídos | Preparação auditada não certifica independência das sessões. |
| H1: lifting | [92](92-LEARNED_LIFTING_STUDY.md), contraste e coder concluídos | Sem avanço prático; não refuta toda arquitetura reversível. |
| H2: âncoras/inovação | [93](93-NEURAL_ANCHOR_INNOVATION_STUDY.md), congelado e conjunto concluídos | Ganhou do independente, não do VQ maduro; validação consumida em [95](95-FROZEN_DEVELOPMENT_VALIDATION.md). |
| H3: normalização interna | [100](100-LOCAL_INTERNAL_NORMALIZATION_STUDY.md), contraste concluído | Sem avanço; variante local testada, não toda normalização possível. |
| H3: condicionamento físico | [98](98-PHYSICAL_METADATA_AND_PROVENANCE.md), censo; três braços concluídos em [105](105-PHYSICAL_CONDITIONING_STUDY.md) | Gate reprovado no contraste restrito; não refuta toda forma de condicionamento nem certifica independência de sessão. |
| H4: precisão controlada e residual neural | [103](103-ERROR_BOUNDED_RESIDUAL_STUDY.md), duas fases concluídas | Garantia verificada; ganho causal de 0,378% insuficiente, sem extensão. |
| H5: multicore | [99](99-CLOCK_ALIGNED_MULTICORE_OPPORTUNITY.md), triagem linear e magnitude condicional concluídas | Ambos reprovados; magnitude usou 24/24 fallbacks por braço. Não refuta toda dependência não linear ou todo modelo de entropia. |
| P4: composição semissintética | [104](104-SEMISYNTHETIC_AUGMENTATION_STUDY.md), contraste pareado concluído | Pior que controle; portadores medidos não eram fundos puros certificados. |
| P4: curva de diversidade | [104](104-SEMISYNTHETIC_AUGMENTATION_STUDY.md), 4/6/10 configurações concluídas | Não monotônica; regimes não são sessões independentes e orçamento curto não mede convergência. |
| P4: novas sessões FEBUS | Manifesto/validador preparados em [98](98-PHYSICAL_METADATA_AND_PROVENANCE.md) | Nenhuma nova aquisição independente incorporada. |
| P3: confirmação | Não iniciada | Nenhum candidato elegível; teste e EDAS permanecem reservados. |

Portanto, não declarar que todas as variantes dessas famílias foram avaliadas
ou refutadas. Os desenhos autorizados de H4 e dados foram fechados sem promoção;
o teste restrito de condicionamento foi fechado em 105 e o de magnitude
condicional H5 em 99, sem extensão. Transferência entre sessões
e novas aquisições ainda dependem da evidência descrita em 98.
Não repetir os gates negativos nem transformar conclusão de etapa em obrigação
de gastar outro budget. Nova continuação precisa de mecanismo distinto ou
novos dados documentados, sem reabrir reservas para procurar uma vitória.
