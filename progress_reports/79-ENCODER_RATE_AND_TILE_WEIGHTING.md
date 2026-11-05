# Encoder VQ com taxa diferenciável e ponderação dos tiles

**Protocolo e execução: 31/10/2026. Consolidação: 02/11/2026.** Campanha concluída,
em continuidade ao [relatório 78](78-REMOTE_REPRESENTATION_PILOTS.md): quatro braços
com 8.192 updates cada (**32.768 atualizações**), buscas com guarda e diagnóstico
temporal. O protocolo foi registrado antes dos treinamentos. Nenhum modelo foi
promovido. Na retomada de 02/11 foram realizadas somente auditorias de inferência,
leitura/sincronização de artefatos e documentação; **nenhum treinamento novo naquela
consolidação**. A seção 8 registra a campanha local posterior, sob nova autorização.

## 1. Decisão de pesquisa

O próximo contraste não será outra escala da CNN ou outro codec escalar sem
controle. E1 mostrou ganho recuperável ao escolher índices diferentes; E2 não
superou os controles e teve colapso em uma condição de otimização. Portanto:

1. **Isolar o encoder:** decoder, codebook, normalização, geometria e tabelas
   ficam congelados. Comparar adaptação do encoder com e sem gradiente de taxa.
2. **Isolar o tiling:** testar amostragem por células dentro da configuração,
   preservando exatamente a sequência de configurações do controle.
3. Comparar custo real, reconstrução discreta e caudas; não tratar melhora de
   loss ou probabilidade suave como melhora do codec.

Essas etapas têm autorização do usuário para implementação e execução nas
RTX 4090 remotas. A ausência de ganho em uma etapa não implica lançar automaticamente
uma rede maior; uma continuação deve responder à hipótese seguinte.

## 2. Controles e partições

- Máquina: `aumo-gpu-001`, duas RTX 4090. Autenticação exclusivamente via
  `SSH_PASSWORD`, sem registrar seu valor.
- Preparação completa: `data/finetune/spatial_band_v1_fold1_20261026`, com
  14.620 janelas de treino e 6.940 de validação. O teste do protocolo não é usado.
- Inicialização: `results/remote_setup/compact_transfer_fold1_weights.pt`, a mesma
  referência compacta transferida do relatório 78. Cada braço parte dos mesmos pesos.
- AdamW novo para o encoder, `lr=1e-5`, `weight_decay=1e-5`, clip de gradiente 1,
  batch 1, FP32. Decoder/codebook não entram no otimizador.
- Distância/índices e síntese usam o codebook original; não há vetores contínuos
  transmitidos nem skip gratuito. O decoder original deve reproduzir exatamente
  a reconstrução obtida dos índices transmitidos.

## 3. Taxa diferenciável sem trocar o VQ

As probabilidades marginais são ajustadas **somente no treino**, com um recorte
por arquivo/região, seed 2081 e suavização de contagem +1. As mesmas tabelas ficam
congeladas em todos os braços. Para cada símbolo, `c[k] = -log2 p[k]`.

O valor de taxa no forward é a soma dos custos dos índices duros por célula.
Seu gradiente aproximado usa a expectativa dos custos sob
`softmax(-distância_quadrática / temperatura)`, sobre os 2.048 códigos. A operação
é explicitamente um estimador de gradiente: o stream e a avaliação continuam duros.

A temperatura é a mediana, entre janelas de treino, da mediana do intervalo entre
as duas menores distâncias, com piso `1e-5`. Em exemplos de treino por família e
tamanho, mede-se `||grad(D+VQ)||/||grad(R)||`. O coeficiente de taxa é 0,1 vezes
a mediana dessas razões, limitado a `[1e-4, 10]`. Nenhum valor é escolhido pela
validação. As normas e escolhas ficam em `calibration.json`.

Objetivo: `0.5*MSE + 0.5*L1 + VQ_histórico + coeficiente*R`. O gradiente de
commitment histórico é preservado; o termo de codebook não atualiza o embedding.
O controle usa coeficiente zero. O experimento não testa um prior adaptativo,
hyperprior ou codebook treinado conjuntamente: isola o custo de símbolos chegando
ao encoder da representação existente.

## 4. Amostragem: contraste separável

O controle sorteia uma janela uniformemente dentro da configuração. A variante
usa probabilidade proporcional a `canais * amostras` dentro da mesma configuração.
O sorteio das configurações mantém as probabilidades históricas por contagem de
janelas, sem novo balanceamento de famílias.

Ambos os samplers consomem os mesmos sorteios e produzem **a mesma sequência de
configurações** para budgets iguais. Só muda a CDF dentro da configuração. Todos
os tiles mantêm probabilidade positiva. Exposição por configuração, faixa de
tamanho e células processadas será registrada. Mesmo número de updates não
garante mesmo compute ou número de células: isso será declarado, não ocultado.

## 5. Série e critérios predefinidos

| Braço | Encoder | Gradiente de taxa | Amostragem |
|---|---|---|---|
| Controle | Treinável | Não | Janela |
| Rate | Treinável | Sim, coeficiente calibrado no treino | Janela |
| Cell | Treinável | Não | Células dentro da configuração |

Controle e Rate são a primeira etapa, um por GPU. Cell é a etapa seguinte para
testar a lacuna de ponderação observada; não recebe pesos escolhidos de Rate.
Se houver evidência favorável em ambos os mecanismos, um quarto braço combinado
é uma extensão condicional, não uma interação já demonstrada.

Cada braço tem **8.192 updates iniciais**, blocos de 2.048 e no máximo **16.384**.
Uma única extensão exige simultaneamente ≥0,2% de melhora relativa de RMSE com
guarda no último bloco e ≥0,2% de ganho contra a origem. Sem isso, encerrar.
O investimento-alvo permanece ≥5% de redução de RMSE macro com guarda a bytes não
maiores frente ao controle pareado, seguido de confirmação separada.

Validação piloto: dois recortes por arquivo/região, seed 2081, selecionados por
metadados. Checkpoint inicial elegível e seleção pelo menor RMSE macro com guarda;
checkpoint de melhor RMSE sem guarda também é preservado para diagnóstico.
Ao final, avaliação completa das 6.940 janelas, sem seleção usando teste.

### Contrato de bytes e guarda

O encoder pode comparar os códigos da referência e da variante usando o original,
disponível ao compressor. O frame novo só é escolhido se **naquela janela**:

- não aumentar os bytes reais;
- não aumentar RMSE ou p90 por canal;
- não reduzir correlação.

Caso contrário, transmite os índices originais. Isso exige executar o encoder
original como fallback na codificação, mas **não** outro decoder no receptor ou
um sinal contínuo extra. Os frames já identificam o mesmo decoder/tabelas; não
há custo oculto de roteamento. O encoder novo é custo de codificação e seus pesos
não são necessários ao receptor. Isso não dispensa medir a latência posteriormente.

São preservados resultados **sem guarda** e **com guarda**. Esta última impede
regressões nas métricas protegidas por construção; a evidência experimental útil
é a magnitude do ganho e sua repetição, não apenas a ausência de regressão.
Amplitude, fase e espectro continuam auditados, sem prometer sua preservação a
partir dessas três guardas. Framing, side, tabelas, mapas e transporte entram nos
bytes completos. Pesos compartilhados do decoder/codebook são separados.

## 6. Implementação e rastreabilidade

- `finetuning/vq_encoder_rd.py`: forward duro, gradiente suave e samplers pareados.
- `finetuning/run_encoder_rd_study.py`: calibração de treino, pilotos limitados,
  checkpoints, exposição, stream real e avaliação física.
- `tests/test_vq_encoder_rd.py`: valor duro/gradiente não nulo, congelamento,
  replay exato, dose de configurações e probabilidade positiva de tiles pequenos.

Os protocolos guardam hashes de código, dados e checkpoint. Cada série tem saída
própria; nenhum artefato do relatório 78 será sobrescrito. O repositório remoto
permanece em `aac0b8d`; as mudanças locais ainda não commitadas são sincronizadas
explicitamente, preservando o índice Git e as alterações prévias do usuário.

## 7. Resultados e decisões

**Calibração concluída no treino:** temperatura **6,171459** e coeficiente de
taxa **0,139882**. O controle usa zero. As duas execuções de encoder começaram
em GPUs separadas, com os mesmos pesos, seed e sequência de janelas.
Diretórios remotos em `results/experiments/`:

- `encoder_rd_calibration_fold1_20261031/`;
- `encoder_control_window_fold1_20261031/`;
- `encoder_rate_window_fold1_20261031/`.

### Primeira comparação completa: encoder e taxa

Ambos os braços encerraram em 8.192 updates pela regra predefinida. As doses
por configuração e o número de células processadas foram **exatamente iguais**
(902.599.000 células). Nenhum parâmetro do decoder/codebook mudou.

| Braço | Passo selecionado | RMSE sem guarda | RMSE com guarda | Ganho com guarda contra origem | Janelas alteradas admitidas |
|---|---:|---:|---:|---:|---:|
| Origem congelada | 0 | 0,648189 | 0,648189 | — | — |
| Controle | 8.192 | 0,641600 | 0,645764 | 0,374% | 1.396/6.940 |
| Rate | 2.048 | 0,642629 | 0,646657 | 0,236% | 1.024/6.940 |

O pacote completo da origem, com as novas tabelas de treino compartilhadas,
tem **15.062.332 bytes**. Controle usa 15.061.911 sem guarda e 15.056.984 com
guarda; Rate usa 15.082.431 sem guarda e 15.059.527 com guarda. Não houve violação
de bytes/RMSE/correlação/p90 em nenhuma janela após a guarda. A economia pequena
da política não compensa o ganho de fidelidade abaixo do alvo de investimento.

**Decisão:** não estender estes dois braços nem promover Rate. Os budgets de treino
e as doses foram pareados, mas os checkpoints selecionados têm exposições
diferentes (2.048 versus 8.192): esta é uma comparação da mesma política de seleção,
não de dois checkpoints finais com idade idêntica. No endpoint comum de 8.192,
o piloto também favoreceu o Controle (RMSE com guarda 0,643570 versus 0,645430).
A etapa Cell permanece um contraste separado, motivado pelo tiling,
e o diagnóstico hard/soft abaixo determina se uma ablação de gradiente é cabível.

### Ponderação por células concluída

Cell também encerrou em **8.192 updates**, com a mesma dose de cada configuração
do Controle. O número de células processadas subiu de **902.599.000 para
1.394.338.000** (aproximadamente 54,5%). Updates de tiles de até três canais
caíram de 644 para cinco, enquanto os de 129–512 canais passaram de 1.034 para
2.041. Isso é mudança de exposição, não descarte de regiões da avaliação/stream.

O checkpoint selecionado foi o de 8.192 passos. RMSE sem guarda **0,641419**;
com guarda **0,645646**, ganho de **0,392%** contra a origem. O pacote com guarda
usa **15.056.121 bytes**. Frente ao Controle, o ganho de RMSE com guarda é só
**0,0184%**, para mais células/compute. O contraste não sustenta ampliar essa
campanha ou combinar mecanismos automaticamente.

Essa é evidência limitada à adaptação do encoder com reconstrutor congelado,
uma seed e uma coorte; não demonstra que a ponderação seja irrelevante em todo
treinamento ponta a ponta. O desequilíbrio de exposição existe, mas corrigi-lo
nesse piloto não mudou o patamar de fidelidade.

### Diagnóstico temporal paralelo, sem treinamento

Foi examinada a lacuna E3, sem lançar um codec temporal: primeiro arquivo
de treino por configuração, primeiro bloco disponível, todas as regiões espaciais
retidas, até um segundo **contínuo de HDF5**. A primeira metade ajusta preditores
AR ridge (`0.001`), e a segunda metade mede previsão one-step. O contexto curto
usa lags até 1.000 amostras (50 ms); o longo adiciona 2.000/4.000 (100/200 ms).
Ambos usam os mesmos alvos/instantes de ajuste. O baseline é a média por canal
ajustada no prefixo. Targets ordinários não são descartados por contexto inválido:
nesses casos há fallback à média; escapes são contados separadamente.

O preditor recebe passado **original**, não reconstruído. Portanto seu ganho é
um diagnóstico otimista de dependência, não resultado de codec, economia de bits,
teste de independência ou autorização automática para E3. Essa análise roda em
CPU local e não executa atualizações de parâmetros neurais. Fonte:
`finetuning/audit_continuous_context.py`.

**Resultado concluído:** 15 regiões de 11 arquivos/configurações de treino,
cada recorte com um segundo contínuo e 0,5 s de ajuste/0,5 s de avaliação. Foram
avaliadas 24.590.000 células futuras, sem escapes nesses recortes. A mediana da
razão MSE AR curto / média de prefixo foi **0,94557**. A mediana do ganho relativo
de RMSE ao adicionar lags de 100/200 ms foi **−0,0167%**; o maior ganho por região
foi **0,869%**. Regiões constantes não recebem ganho fictício por erro zero.

Isso não sustenta lançar uma campanha neural de contexto longo por enquanto.
Há previsibilidade temporal de curto alcance em parte do sinal, mas os lags
adicionais desse preditor não mudaram o patamar. O resultado não descarta modelos
temporais não lineares, contexto espacial ou uma arquitetura recorrente; tampouco
prova independência. Artefatos locais:
`results/experiments/continuous_context_train_fold1_20261031/`.

### Extensão E1: guarda dentro da busca

Após Controle/Rate, foi testada uma limitação concreta do E1 anterior:
buscar o melhor candidato global e só então aplicar fallback pode descartar
outros candidatos que melhorariam o sinal sem violar caudas. A extensão filtra
**cada candidato discreto** durante a busca, preservando o original e os mesmos
tetos/guardas. Trajetória Adam, 48 passos, projeções, tabelas e calibração de LR
no treino são mantidas. Não há atualização de pesos neurais.

Primeiro, comparação exploratória nos mesmos 63 recortes usados pelo E1 com
guarda do relatório 78. Depois, verificação em 63 recortes adicionais, seed 2082,
excluindo as 126 janelas dos dois pilotos anteriores. A segunda avaliação é
predefinida aqui, mas ainda usa arquivos/configurações conhecidos, não novas
aquisições. A comparação deve considerar RMSE, bytes, caudas e custo de busca,
sem presumir que o filtro novo domine todas as métricas do resultado antigo.

**Resultados concluídos:**

| Recortes | RMSE origem → busca | Redução relativa | Frames origem → busca | Correlação origem → busca |
|---|---:|---:|---:|---:|
| 63 conhecidos, seed 2079 | 0,645465 → 0,618201 | 4,224% | 97.031 → 96.774 bytes | 0,751847 → 0,772102 |
| 63 adicionais, seed 2082 | 0,644675 → 0,618798 | 4,014% | 96.743 → 96.498 bytes | 0,752256 → 0,771683 |

Nos recortes conhecidos, a política anterior com guarda apenas ao final obtinha
**3,798%**; a guarda dentro da busca recupera candidatos antes perdidos. No novo
conjunto, o p90 macro cai de **1,027963 para 0,987137**. Os tetos/guardas são
preservados por janela, e o decoder continua recebendo apenas índices discretos.
O resultado melhora E1, mas ainda não atinge 5% com guarda nem representa nova
aquisição. Não justifica uma campanha longa de destilação ou uma promessa a 700×.
Artefatos: `guard_aware_search_{known,fresh}_fold1_20261031/`.

### Diagnóstico do estimador e extensão de temperatura

Controle e Rate concluíram 8.192 updates cada, sem acionar extensão. A validação
completa encontrou ganhos com guarda de apenas **0,374%** e **0,236%** contra a
origem, respectivamente: o gradiente de taxa não superou o controle.

Uma auditoria adicional, restrita às mesmas 135 janelas de **treino** que ajustam
a tabela, revelou uma hipótese mais específica: a probabilidade suave melhora,
mas o índice duro não acompanha. Na origem, as taxas hard/soft eram
**0,162331 / 0,194284 bit/célula**. No último checkpoint Rate, passaram para
**0,162912 / 0,191636**; os frames aumentaram de **369.435 para 370.753 bytes**.
Logo, ter gradiente não nulo e reduzir a proxy não comprova reduzir o stream.
Não se deve explicar isso apenas como penalidade insuficiente e aumentar lambda.

Foi testado um único contraste adicional **Rate-sharp**: temperatura igual a
0,1 da regra original, com o coeficiente recalibrado no treino pela **mesma** regra
de norma de gradiente. Tabelas, pesos iniciais, loss, sampler, seed e encoder-only
são mantidos. O orçamento fica fixo em **8.192 updates**, igual ao Controle; não
há extensão automática nem busca de várias temperaturas pela validação. A
auditoria deve verificar novamente proxy/hard/bytes antes de recomendar qualquer
ampliação. Fonte do diagnóstico: `finetuning/audit_rate_surrogate.py`.

### Resultado final de Rate-sharp

A calibração de treino produziu temperatura **0,617146** e coeficiente
**0,100313**. O braço concluiu os 8.192 updates previstos, selecionando o último
checkpoint. Sua dose de configurações, janelas e células é a mesma do Controle.

| Política | RMSE macro | Correlação macro | P90 macro por canal | Bytes completos |
|---|---:|---:|---:|---:|
| Origem | 0,648189 | 0,749319 | 1,024954 | 15.062.332 |
| Rate-sharp sem guarda | 0,641510 | 0,753820 | 1,025564 | 15.021.546 |
| Rate-sharp com guarda | 0,645022 | 0,751676 | 1,014708 | 15.042.038 |

A política com guarda admite alterações em **3.523/6.940 janelas**, sem violação
de bytes/RMSE/correlação/p90 por janela. Seu ganho de RMSE é **0,489% contra a
origem** e **0,115% contra o Controle**, com aproximadamente 0,10% menos bytes
que o Controle. A versão sem guarda tem RMSE agregado melhor, mas p90 macro
ligeiramente pior que a origem; não é promovida por essa média.

As guardas não cobrem toda a fidelidade física: mesmo após aplicá-las, o erro
de magnitude FFT cresce em doped/5m SMF e multicore/C1 and C4; neste último,
de **0,594185 para 0,595587**. A razão RMS fica ligeiramente mais distante de 1
em três configurações, embora a fase agregada por configuração melhore nas seis.
Esses trade-offs foram mantidos na análise, sem ampliar retroativamente as
garantias da política.

O estimador mais nítido melhorou o braço Rate original e os bytes medidos,
mas ficou muito abaixo do alvo de investimento de 5% contra o controle pareado.
Não foi iniciado o braço combinado Cell+Rate, outra temperatura, hyperprior ou
continuação desses treinamentos.

### Auditoria de fechamento e rastreabilidade

A comparação final está em
`results/experiments/encoder_comparison_final_20261102/`. Ela verifica os mesmos
6.940 caminhos de validação e o mesmo checkpoint de origem nos quatro braços,
além de ausência de violações nas guardas por janela. Os quatro budgets concluídos
e as doses por configuração são iguais; somente Cell altera a exposição em células.
Checkpoints selecionados podem ter idades diferentes, como declarado acima.

As GPUs estavam ocupadas por outros processos na retomada. A auditoria adicional
de Rate-sharp usa **CPU**, `CUDA_VISIBLE_DEVICES=""`, pesos congelados e
`torch.no_grad()`, sem otimizador ou atualização de parâmetros. A origem é
reavaliada na mesma CPU para não atribuir diferenças CPU/GPU ao método. Métricas
de validação publicadas acima são as da execução original nas RTX 4090.

Nos mesmos **135 recortes de treino / 17.490.000 células**, com a temperatura
mais nítida e a mesma tabela original, a auditoria de CPU concluiu:

| Checkpoint | Custo marginal duro (bit/célula) | Proxy suave (bit/célula) | Bytes dos frames |
|---|---:|---:|---:|
| Origem, CPU | 0,16233162 | 0,16386088 | 369.445 |
| Rate-sharp selecionado, CPU | 0,16170114 | 0,16297202 | 368.099 |
| Rate-sharp último, CPU | 0,16170114 | 0,16297202 | 368.099 |

Aqui proxy, custo duro e bytes melhoram na mesma direção; os frames caem cerca
de **0,364%**. O selecionado e o último correspondem ao mesmo passo, embora os
arquivos tenham hashes diferentes por conterem metadados/otimizador diferentes.
A origem na auditoria GPU anterior produzia 369.435 bytes, contra 369.445 na CPU:
não se presume igualdade bit a bit entre plataformas. A comparação do método
acima é inteiramente CPU/CPU e não substitui a avaliação original em GPU.
Artefatos: `results/experiments/encoder_surrogate_sharp_cpu_audit_20261102/`.

Foram sincronizados os protocolos, logs, métricas completas, tabelas e snapshots
dos experimentos concluídos para `results/experiments/` local, sem sobrescrever
arquivos locais diferentes. Checkpoints e streams completos permanecem no remoto.
Conferidos **28 hashes** de snapshots contra os quatro protocolos originais,
os budgets concluídos e as flags de congelamento/replay discreto. A comparação
final não encontrou violações de guarda nas 6.940 janelas de nenhum braço.
A suíte local passou em **262 testes**, executada com CUDA desabilitada.

A única adaptação de código nesta consolidação foi permitir `--device cpu` na
auditoria de taxa e registrar dispositivo/zero updates. O código original da
auditoria remota foi preservado antes da atualização. Não foram alterados pesos,
partições, tolerâncias, protocolos de treinamento ou o índice Git do usuário;
nenhum commit ou push foi realizado.

### Decisão consolidada e próximos passos — sem execução automática

1. **Manter os controles VQ existentes.** Nenhum novo encoder atingiu o critério
   de investimento; não repetir essas continuações nem ampliar a CNN por inércia.
2. **Preservar E1 com guarda durante a busca como opção experimental.** O ganho
   de cerca de 4% foi repetido em recortes adicionais, mas ainda exige avaliação
   ampliada, custo de codificação e aquisições independentes antes de promoção.
   Os pilotos E1 e a validação completa dos encoders não são amostras pareadas
   para atribuir diretamente a diferença entre seus ganhos aos métodos.
3. **Se houver novo treinamento autorizado, isolar representação/quantização
   ponta a ponta e a fidelidade do estimador de taxa.** A proxy suave precisa
   acompanhar índices duros e bytes reais; gradiente não nulo não é prova disso.
   Decoder/codebook congelados limitam o alcance deste resultado negativo: não é
   uma refutação de todo aprendizado conjunto ou de outras quantizações.
4. **Manter contexto longo como caminho condicional.** O diagnóstico linear
   contínuo não forneceu ganho suficiente dos lags adicionais; não lançar uma
   rede temporal apenas porque as janelas atuais têm 50 ms.

São recomendações de pesquisa, não autorização ou agendamento de novas execuções.
As guardas e o alvo exploratório permanecem inalterados. Nenhum resultado aqui
estabelece fidelidade operacional suficiente ou recuperação a 700×.

## 8. Continuidade local autorizada: residual e informação disponível ao decoder

Esta seção registra uma campanha posterior às decisões acima. O usuário autorizou
prosseguir entre etapas, exigindo **RTX 4060 local para qualquer treinamento**.
Não serão iniciados trabalhos de compute ou treinamento no remoto. Protocolos,
snapshots e artefatos novos têm diretórios próprios, sem modificar as campanhas
encerradas. O roadmap contém a síntese da comparação pareada E1/encoders e da
auditoria inicial das bordas; esta seção documenta o próximo contraste.

### Protocolo antes da execução

Os probes anteriores usavam o residual **original** do canal vizinho, indisponível
ao receptor. A próxima hipótese é mais limitada: **há correção espacial útil a
partir somente da saída já decodificada?** Uma correção pequena pode testar isso
sem ampliar a CNN, mudar índices, acrescentar latentes ou esperar outra campanha
ponta a ponta. Não equivale a demonstrar toda a redundância espacial restante.

1. Reconstituir os mesmos frames da origem e auditar erro absoluto/relativo por
   canal, separando bordas/interior e canais fracos. Fraqueza diagnóstica:
   `std_canal < max(0,001*max_std_da_janela, 1e-12)`, no alvo ordinário normalizado.
   Nenhum canal sai do alvo ou da avaliação.
2. Treinar **dois corretores residuais pequenos**, inicializados exatamente como
   identidade: MLP pontual com duas camadas ocultas de 16 unidades. O braço espacial
   recebe cinco taps espaciais da reconstrução (±2 canais); o controle repete o
   canal central cinco vezes. Ambos recebem distâncias truncadas às duas bordas,
   têm a mesma arquitetura/parâmetros nominais e inicialização. A dimensão efetiva
   dos features difere; o controle separa vizinhança de ajuste pontual/posição.
3. Encoder, codebook e decoder originais permanecem congelados. Só os corretores
   recebem gradiente, **na GPU local**, com batch 1, Adam `1e-3`, FP32, clip 1 e
   `0,5*MSE + 0,5*L1` nas células ordinárias normalizadas. Sem filtro no alvo, clipping,
   GAN, alvos perceptuais ou acesso ao residual original no receptor.
4. Usar os 135 recortes de treino do diagnóstico, com separação interna de arquivos
   por configuração (seed 2084, aproximadamente 20% dos arquivos para seleção).
   Os dois braços recebem a mesma sequência de exemplos e **1.024 updates cada**.
   Sem extensão automática. Preservar checkpoints no passo zero e avaliar
   256/512/1.024; selecionar pela menor distorção macro normalizada **com guarda**
   no holdout interno de treino, sem denominador relativo instável de canais
   constantes. A separação por arquivo não garante independência de aquisição.
5. Avaliar os 63 recortes de validação só depois de selecionar ambos os braços.
   Preservar scores sem guarda e com fallback à origem; não reduzir RMSE/correlação/
   p90 da origem por janela. Métricas indefinidas rejeitam o candidato alterado.
   Amplitude/fase/espectro continuam diagnósticos concorrentes, não garantias.
6. O receptor precisa do corretor compartilhado, além do decoder original. O
   envelope experimental **FSC1** cobra 13 bytes por frame para identificação de
   modo, tamanho e CRC, inclusive no controle de origem. O payload original é
   preservado; modo zero usa origem e modo um usa correção. Tabelas, mapas,
   transporte, registro e arquivo do corretor são contabilizados explicitamente.
   Pesos compartilhados da base ficam separados. Não alegar taxa igual à origem
   sem envelope nem tratar novos pesos compartilhados como custo inexistente.
7. Critério de investimento: buscar **≥5% de ganho de RMSE com guarda contra o
   controle pontual**, com payload enquadrado igual, inspeção física e declaração
   do custo adicional de implantação. Ganho menor não justifica ampliar esse
   corretor nem uma campanha de tuning. O ponto inicial continua elegível.

Implementação: `finetuning/run_spatial_output_study.py`,
`finetuning/spatial_output_correction.py` e
`compression/spatial_correction_stream.py`. O smoke usa orçamento separado e
diretório próprio; seus scores não selecionam arquitetura ou hiperparâmetros.
Uma etapa negativa será registrada e encerrada, sem impedir um contraste distinto
e controlado se os diagnósticos sustentarem outra hipótese.

### Contraste congelado adicional: orientação e preenchimento espacial

Registrado antes da execução, sem depender de um ganho favorável do corretor:
comparar **reversão da ordem espacial** no encoder, desfeita após o decoder, e
**replicação apenas das linhas extras de entrada** necessárias ao múltiplo de 16.
O padding temporal continua zero; convoluções internas, pesos, codebook e grade
latente não mudam. A segunda alternativa é identidade nos shapes sem linhas
extras. Isso testa alternativas específicas de codificação, não prova a causa
dos erros nem é ablação de todo o padding da rede.

Os mesmos 135/63 recortes recebem somente essas duas alternativas, predefinidas.
Cada uma e uma política conjunta conservam o frame original e aceitam candidatos
apenas sob teto de bytes, RMSE, correlação e p90 por janela. A política conjunta
minimiza a distorção temporal ordinária entre os admitidos; não busca parâmetros
na validação. Resultados sem guarda também são preservados. O envelope FSC1
identifica orientação direta/reversa, com custo igual de 13 bytes em todos os
controles; o registro declara o transformador determinístico, sem pesos novos.
Tabelas/mapas/transporte/registro entram nos bytes completos; os pesos da base
são compartilhados e separados. A fonte, a mistura e as alternativas usam o
mesmo contrato. Manter o alvo/cobertura e testar replay do stream sem original.

O critério exploratório é ≥5% de ganho de RMSE com guarda contra origem,
a bytes não maiores, seguido de confirmação. Sem isso, não ampliar por inércia
essa família de alternativas. Implementação:
`finetuning/run_spatial_boundary_coding.py`. Toda inferência será local.

### Próximo contraste predefinido: coadaptação VQ com/sem taxa

Se as duas intervenções pequenas não atingirem o critério de investimento,
prosseguir com uma triagem distinta e delimitada: **encoder, decoder e codebook
treináveis nos dois braços**, inicializados do mesmo compacto funcional. O controle
usa a distorção temporal/VQ histórica; Rate soma custo entrópico hard no forward
com estimador suave de gradiente. Tabelas de probabilidade, grade, normalização,
cobertura, seed, exemplos e orçamento são comuns; o prior não é treinado.
Assim, uma diferença entre braços não pode ser atribuída simplesmente a liberar
o decoder/codebook. O resultado encoder-only anterior não testa esse contraste.

- **Somente RTX 4060 local**, FP32, checkpointing de ativações, TF32 desabilitado,
  AdamW novo `lr=1e-5`, `weight_decay=1e-5`, batch 1 e clip 1. Orçamento fixo de
  **1.024 atualizações por braço**, sem extensão ou sweep. Um smoke separado de
  dois updates por braço inclui o maior shape para verificar memória/gradientes.
- Amostrar o treino completo do fold 1, excluindo todos os arquivos usados pelo
  holdout interno de seleção anterior. Mesma sequência nos dois braços, seed 2085.
  A seleção usa os mesmos 22 recortes desses arquivos de treino e os pontos
  0/256/512/1.024, por menor distorção macro normalizada com guarda. O orçamento
  pequeno é uma triagem, não prova de convergência ou impossibilidade.
- Temperatura train-only já fixada em Rate-sharp, sem escolha pela validação.
  Recalibrar o coeficiente no **escopo completo** dos parâmetros por
  `0,1*mediana(||grad(D+VQ)||/||grad(R)||)`, limitado a `[1e-4,10]`, em oito
  exemplos de treino selecionados por família/tamanho e **fora** dos arquivos de
  seleção interna. Calibração não atualiza pesos. A temperatura histórica foi
  estimada no treino, incluindo dados desse holdout interno; este não é uma
  auditoria independente de todas as escolhas históricas.
- Validar os dois checkpoints selecionados nos mesmos 63 recortes somente após
  encerrar ambos os treinos. Guardar scores brutos e com guarda, taxas hard/soft,
  frames reais, bytes de implantação, passos selecionados e memória/tempo.
  Mesma política de seleção não garante a mesma idade dos checkpoints escolhidos.
- Agora os códigos do candidato requerem **seu próprio codebook/decoder**. O
  envelope identifica explicitamente modelo original versus candidato, e o
  registro associa os dois hashes. As tabelas internas conservam a identidade
  do alfabeto original; sozinhas não identificam o novo reconstrutor. Fallback
  exige os dois modelos no receptor, sem fingir que o novo decoder é gratuito.
- Alvo exploratório: **≥5% de ganho de RMSE com guarda de Rate contra o controle
  full-VQ**, a frames não maiores e com inspeção das métricas concorrentes.
  Abaixo disso, encerrar sem ampliar automaticamente o orçamento, a rede, o
  hyperprior ou a busca de temperatura. Nenhum candidato será promovido por esse
  piloto sem confirmação e aceitação física.

Implementação: `finetuning/run_joint_vq_rate_study.py`. O novo orçamento e seus
artefatos são separados dos 32.768 updates encoder-only da campanha original.

### Verificação adicional predefinida: custo de proteger amplitude/espectro em E1

Além dos pilotos acima, executar uma comparação congelada em **63 recortes
adicionais**, seed 2086, excluindo os recortes de validação dos protocolos E1
anteriores disponíveis e da comparação pareada local. Os arquivos/configurações
continuam conhecidos; não é auditoria externa. Fixar LR de calibração histórica
de treino, 48 passos, projeção a cada quatro e as mesmas tabelas/tetos de bytes.

Comparar a guarda histórica a uma guarda **aninhada mais restritiva**: também
não aumentar erro de magnitude FFT nem distância da razão RMS a 1, e não diminuir
alinhamento de fase, por candidato/janela. São critérios diagnósticos de
não-regressão exata, **não tolerâncias físicas validadas da aplicação**. Não
reinterpretam nem alteram retroativamente os experimentos encerrados. Registrar
quanto ganho de RMSE resta, bytes, frequência de fallback e custo de busca; não
retunar tolerâncias na validação para salvar o resultado. Preservar frames e
replay discreto. Implementação: `finetuning/run_physical_guard_search.py`.

### Resultados locais: correção e alternativas congeladas

**Auditoria por canal:** nas configurações single-mode 12/4/5, nenhum canal de
borda foi classificado como fraco pela regra diagnóstica de `0,001*max_std`.
As razões pooled de RMSE por energia centrada, descritivas sobre os recortes
normalizados, foram aproximadamente **0,571/0,197**, **0,573/0,171** e
**0,559/0,383** em borda/interior, respectivamente. Portanto o efeito não se
explica apenas por denominadores quase nulos sob essa regra. Isso não elimina
heterogeneidade de amplitudes, não valida universalmente esse limiar e não prova
causalidade do padding.

**Corretores locais:** 113 recortes de treino e 22 de seleção interna, com arquivos
disjuntos. Os dois braços concluíram 1.024 updates na RTX 4060, selecionando o
passo 1.024. Cada corretor tem **417 parâmetros**. Resultados nos mesmos 63 recortes:

| Política | RMSE macro | Ganho contra origem | Frames com envelope | Stream com auxiliares/registro | Corretor compartilhado adicional |
|---|---:|---:|---:|---:|---:|
| Origem enquadrada | 0,644823 | — | 98.194 | 401.246 | 0 |
| Pontual com guarda | 0,644280 | 0,0842% | 98.194 | 401.308 | 4.889 bytes |
| Espacial com guarda | 0,644237 | 0,0910% | 98.194 | 401.308 | 4.865 bytes |

Os pesos da base são compartilhados e separados. O corretor espacial melhora
o pontual em somente **0,00679%** de RMSE; não atinge o critério de investimento.
Admite alterações em 36/63 janelas, contra 35/63 do controle. O erro de magnitude
FFT ainda piora em cinco janelas com o corretor espacial guardado. Não ampliar,
retunar ou promover esses corretores. O negativo se limita ao pós-processamento
local de saídas já decodificadas, não à exploração espacial na representação.

Artefatos: `results/experiments/spatial_output_local_20261102/`. O smoke separado
usou oito updates por braço e não escolheu hiperparâmetros. CPU e GPU não são
presumidas bit-idênticas. O reader independente
`tools/decode_spatial_experiment_frame.py` foi executado em CPU usando apenas
checkpoint base, corretor serializado, contagens, registro e frame, sem NPZ/HDF5.

**Orientação/preenchimento congelados:** a reversão não admitiu melhoria na
validação; a replicação e a política conjunta admitiram alteração em **1/63
janelas**, com redução de RMSE de somente **0,0548%** (0,644823 → 0,644470).
Todos usam 98.194 bytes de frames enquadrados e 401.147 bytes com os mesmos
auxiliares/registro; não há pesos novos. A ausência de economia de frames não
é erro de contabilidade: o único frame alterado manteve seu tamanho. Não ampliar
essas alternativas por inércia. Artefatos:
`results/experiments/spatial_boundary_coding_local_20261102/`.

O controle de origem foi reavaliado com TF32 desabilitado nas duas comparações;
o RMSE macro difere ligeiramente do diagnóstico pareado anterior (0,644823281).
Os novos contrastes usam a mesma política numérica em ambos os lados; não
atribuir essa diferença entre execuções aos métodos novos.

O smoke do contraste full-VQ passou no maior shape de treino, com pico alocado
de aproximadamente **2,91 GB**. A calibração de escopo completo fixou temperatura
**0,617146** e coeficiente **0,414711** antes do treino de produção. Esses valores
não foram escolhidos pelo resultado de validação do smoke.

### Resultado local: coadaptação VQ com/sem taxa

Ambos os braços concluíram **1.024 updates na RTX 4060**, sem extensão, e
selecionaram o passo **256** pelo holdout interno de treino. Havia 11.740 janelas
elegíveis do treino completo após excluir os arquivos de seleção; cada braço
recebeu a mesma sequência de 1.024 sorteios. Não se deve confundir essa exposição
com a dose histórica dos checkpoints de origem ou com um treino convergido.

| Política | RMSE macro | Correlação macro | P90 macro por canal | Frames enquadrados | Stream com auxiliares/registro |
|---|---:|---:|---:|---:|---:|
| Origem | 0,644823 | 0,752140 | 1,023089 | 98.194 | Referência comum |
| Controle full-VQ com guarda | 0,641539 | 0,754841 | 1,006311 | 98.078 | 401.261 |
| Rate full-VQ com guarda | 0,641188 | 0,755084 | 1,006316 | 97.189 | 400.372 |

Rate ganha **0,5637% contra origem** e **0,0547% contra o controle**, com **0,906%**
menos bytes de frames que o controle (~0,222% no stream com auxiliares/registro).
O ganho está muito abaixo de 5%. O p90 macro é ligeiramente pior que o controle,
apesar de não piorar a origem em nenhuma janela admitida. Também há configurações
em que Rate perde para o controle; não alegar dominância universal.

Sem guarda, Controle/Rate têm RMSE **0,639846/0,639976** e payloads de
**97.213/95.897 bytes**, respectivamente: há negociação de taxa, não um salto
de fidelidade. A guarda admite 36/63 e 38/63 janelas. Na média descritiva
**não ponderada por janela** da validação, hard/soft bpc são
**0,391410/0,392663** no Controle e **0,385092/0,386396** no Rate. Proxy, custo
duro e bytes melhoram na mesma direção neste contraste; esses números não são
taxas pooled da coorte nem incluem side/transporte. Não usar a calibração como
substituto dos bytes reais.

O escopo treinável foi encoder/decoder/codebook nos dois braços. O novo receptor
com fallback precisa do checkpoint original e do candidato; os arquivos completos
selecionados medem aproximadamente **111,87 MB cada**, além da base. Incluem também
encoder e metadados: é o tamanho do pacote implementado, não o tamanho mínimo
teórico de decoder/codebook extraído. Esses custos de implantação são separados
do stream, explicitamente não gratuitos. O envelope associa cada frame ao modelo
correto. O reader independente executou o modo candidato em CPU sem NPZ/HDF5.

O treino levou ~202/204 segundos por braço, com picos alocados de ~2,93/3,07 GB.
Após a guarda, o erro de magnitude FFT piorou em 11/63 e 14/63 janelas,
respectivamente. Isso reforça a necessidade do diagnóstico físico, não justifica
relaxar guardas nem promover médias favoráveis. CPU/GPU não são presumidas
bit-idênticas. Artefatos: `results/experiments/joint_vq_rate_local_20261102/`.

**Decisão:** encerrar este contraste sem ampliar treino, taxa, temperatura ou
hyperprior. Há evidência técnica de negociação taxa–distorção com representação
treinável, mas não de retorno suficiente para investimento. O negativo é uma
triagem curta de VQ com prior congelado, não refutação de toda coadaptação,
quantização ou arquitetura neural espacial.

### Preparação externa por metadados, sem avaliação de formas de onda

Além dos experimentos, foi implementado o inventário read-only
`acquisition/audit_external_drive_candidates.py`: 108 candidatos EDAS/SMF de
seis pastas foram comparados aos 207 arquivos do desenvolvimento por ID e por
**MD5/tamanho reportados pelo Drive**. Todos os hashes estavam disponíveis; não
houve coincidência com desenvolvimento nem duplicatas entre os 108 candidatos.
Isso exclui cópias binárias óbvias, mas não prova independência de aquisição ou
ausência de reprocessamentos correlacionados. Nenhum arquivo do Drive foi alterado,
e credenciais OAuth não foram impressas nem regravadas.

Uma triagem HDF5 de um arquivo por pasta, escolhido pelo menor ID e não por sinal,
usou somente leituras HTTP Range com limite de 2 MiB por arquivo; consumiu apenas
**32.768 bytes por arquivo, 196.608 bytes no total**. Nenhum dataset de sinal foi
lido como array. Os seis headers apresentam strain rate float32, espaçamento
espacial 0,2 m, temporal 0,05 ms (**20 kHz**) e gauge length 1 m. Shapes são
[3 ou 4 blocos, 21.000 amostras, 551 canais]. Isso é compatibilidade de esquema
nos arquivos amostrados, não validação de fidelidade ou cobertura das quatro famílias.

Os headers repetem `Hostname/AcqStartTime` entre orientações de uma mesma fibra.
Não interpretar pastas como aquisições independentes. A etapa seguinte, registrada
antes de executar, amplia **somente metadados** aos 108 candidatos e aos 207 HDF5
locais do corpus: ler atributos, shapes e a pequena tabela de clock
`<instrumento>/SourceN/time` (1D, float, no máximo 256 entradas). Não ler arrays
de strain/strain rate nem rodar modelos nesses novos dados. Comparar conservadoramente
instrumento/início de aquisição e clocks, sem afirmar que coincidência de início
seja prova de simultaneidade ou que clocks distintos garantam independência.
Páginas Range podem conter bytes adjacentes a metadados; eles não são interpretados
como sinal nem usados para treino/avaliação. Manter limite de 2 MiB por candidato.

Implementação: `acquisition/audit_external_hdf5_headers.py`. Artefatos iniciais:
`results/experiments/external_drive_hash_screen_20261102/` e
`results/experiments/external_hdf5_header_screen_20261102/`. A ampliação terá
diretório próprio; nenhum holdout final é declarado ou usado para selecionar modelos.

### Resultado local: E1 com guardas físicas aninhadas

A comparação foi concluída nos 63 recortes adicionais, com cinco protocolos de
exclusão registrados. Ambos os braços usam exatamente o mesmo orçamento de busca,
LR histórico de treino, código/decoder congelados e teto real por janela.

| Política | RMSE macro | Ganho contra origem | Correlação macro | P90 macro por canal | Frames | Janelas alteradas |
|---|---:|---:|---:|---:|---:|---:|
| Origem | 0,646550 | — | 0,750837 | 1,048499 | 96.068 | — |
| Guarda histórica | 0,615317 | 4,831% | 0,774077 | 1,003228 | 95.768 | 63/63 |
| Guarda física aninhada | 0,638151 | 1,299% | 0,757205 | 1,035878 | 95.950 | 38/63 |

As duas políticas preservam os tetos/guardas históricas por janela. A busca
histórica piora magnitude FFT em **6/63**, fase em **2/63** e distância da razão
RMS a 1 em **51/63** recortes. A guarda aninhada não apresenta regressão nessas
métricas por construção; a evidência útil é o ganho que ainda sobra: **~1,30%**,
e não uma garantia nova de fidelidade universal. A razão RMS macro é
0,764780 na origem, 0,745581 com a guarda histórica e 0,766626 com a física.

O custo foi aproximadamente 159/160 segundos por política na GPU local,
incluindo avaliação dos candidatos. Os frames excluem custos compartilhados
iguais de tabelas/mapas/transporte/pesos; não são uma razão completa do codec.
Artefatos: `results/experiments/physical_guard_search_local_20261102/`.

**Consequência para a direção:** os ~4–5% de E1 não devem ser chamados de ganho
físico geral. Parte importante do ganho disponível sob o seletor temporal deixa
de ser admitida quando outras métricas são protegidas. Isso não identifica cada
restrição como causa isolada — foram acrescentadas conjuntamente — nem demonstra
que a aplicação deva exigir essas desigualdades exatas. Definir os compromissos
físicos aceitáveis passa à frente de nova destilação ou busca de hiperparâmetros.
Não reduzir exigências retrospectivamente nem ampliar o orçamento para salvar
uma conclusão positiva.

### Auditoria e orçamento da continuidade local

Foram executadas **4.096 atualizações neurais de produção** (2.048 nos corretores
e 2.048 no full-VQ), além de **20 updates de smoke** (16+4), todos na RTX 4060
local. Calibrações, probes e buscas de latentes não atualizam pesos neurais.
Não houve compute/treinamento remoto, nem continuações fora dos budgets.

A auditoria `finetuning/audit_local_spatial_campaign.py` verificou **25 hashes de
snapshots**, CRC/estrutura de **1.170 frames FSC1**, **774 decisões de guarda**
e decodificação de **126 frames E1** com as tabelas congeladas. Conferiu a
separação de arquivos de treino/seleção, sua ausência na calibração full-VQ,
os budgets concluídos e alteração efetiva de **54 tensores do
encoder, 62 do decoder e um do codebook**, em cada braço full-VQ selecionado.
O checkpoint de origem permaneceu inalterado. Resultado:
`results/experiments/local_spatial_campaign_audit_20261102.json`.

O reader CLI foi executado em CPU nos três contratos — corretor, orientação e
modelo completo — usando frames/modelos/tabelas/registro, sem carregar NPZ/HDF5.
As rotinas de avaliação também conferiram escapes e replay. A suíte local passou
em **291 testes**, com CUDA desabilitada. Nenhum novo modelo foi promovido;
nenhum commit/push ou alteração do staging do usuário foi realizado.

### Fechamento da proveniência externa por metadados

A ampliação foi concluída: **108 candidatos e 207 arquivos locais do desenvolvimento**
foram inspecionados com sucesso, sem ler arrays de sinal. Foram lidos somente
atributos/shapes e clocks pequenos explicitamente permitidos. As páginas HTTP
totalizaram **4.661.352 bytes (~4,45 MiB)** para os candidatos, abaixo do limite
por arquivo. O corpus local não foi alterado ou reprocessado.

Todos os 108 candidatos apresentam spacing `[0,2 m, 0,05 ms, 1]` e gauge length
1 m. Não há coincidência de `Hostname/AcqStartTime`, clocks registrados exatos
ou sobreposição de intervalos temporais com os 207 arquivos de desenvolvimento
sob a convenção: `Source/time` em segundos, `Zone Origin/Spacing` temporal em
milissegundos e extensão do bloco derivada do shape. Nenhum arquivo ficou sem
intervalo. Essa ausência é uma triagem de metadados, **não prova de independência
física ou de eventos**, e a convenção precisa continuar rastreada à aquisição.

Os 108 arquivos colapsam em **três grupos conservadores de instrumento/início de
aquisição**, cada um com 36 arquivos, contendo Horizontal e Vertical da mesma
fibra: SMF, EDAS Polyimide e EDAS Acrylate. As uniões de clock por grupo são
aproximadamente 104,8/106,8/118,8 s; não são contagens de réplicas independentes.
Os grupos foram reservados documentalmente sem separá-los entre treino e teste,
sem escolher split final e sem executar modelos nas formas de onda. A coleção
não oferece, por si, validação externa das quatro famílias originais.

Artefatos: `results/experiments/external_acquisition_metadata_full_20261102/`,
incluindo `provenance_summary.json`, produzido por
`acquisition/summarize_external_acquisitions.py`. Testes verificam o limite de
leitura, rejeição de respostas HTTP que ignoram Range, whitelist exclusiva de
clocks e cálculo de uniões/sobreposições. Nenhum token ou header de autenticação
foi registrado nos artefatos.

**Decisão final da rodada:** as hipóteses delimitadas foram executadas e os
budgets encerrados pelos critérios registrados, não por falta de acesso à GPU.
Não promover ou estender os ramos de ganho marginal. A próxima prioridade é
fixar o compromisso físico aceitável e completar a revisão de proveniência dos
três grupos candidatos, antes de mais seleção no mesmo desenvolvimento. Uma nova
proposta de representação precisa de mecanismo/controle adicional; não iniciar
outra rede, temperatura, sampler ou continuação apenas para ocupar o orçamento.

## 9. Nova rodada autorizada: restauração afim e avaliação externa delimitada

### Hipótese e protocolo registrados antes da execução

A diferença entre E1 temporal (~4,83%) e E1 fisicamente guardado (~1,30%) motiva
um contraste diferente dos treinos encerrados: **separar escolha dos índices de
restauração de ganho/offset por janela**. Os relatórios 47–48 testaram ganho global
ajustado somente no treino; não testaram dois coeficientes por janela calculados
no compressor e efetivamente transmitidos, nem sua interação com a busca E1.
Não se presume que aumentar RMS melhore forma de onda, fase ou canais fracos.

- Primeiro diagnóstico exclusivamente nos 135 recortes de **treino**, reutilizando
  os índices congelados de origem/E1 da comparação pareada. Sem atualizar pesos.
  Duas alternativas analíticas fixas: mínimos quadrados afins e ganho de RMS
  unitário com correção de média. OLS não é chamado de ótimo da perda MSE/L1
  mista, nem do conjunto de métricas físicas. Original sempre elegível.
- Exigir teto real por janela e todas as guardas físicas aninhadas anteriores.
  Selecionar menor distorção mista ordinária normalizada entre admitidos. Registrar
  coeficientes, fallback, erro absoluto/relativo e custo. Tiles constantes não
  recebem correlação fictícia e permanecem protegidos por fallback.
- **Contrato novo FAS1:** os dois float64 do side são offset/escala **da saída**,
  não média/std da normalização original. Mesmo layout/tamanho de FNE1, mas magic
  próprio, descriptor explícito e decoder que distingue as duas semânticas.
  Não sobrescrever a definição de FNE1 nem alegar que parâmetros adaptativos são
  gratuitos: eles ocupam os campos transmitidos. Threshold, índices/valores de
  escape e geometria são preservados. O receptor não recebe o alvo nem a
  normalização de entrada original, desnecessária para essa reconstrução.
- Verificar igualdade do tamanho efetivo dos frames, replay da restauração física
  e preservação bit-exata de escapes; contabilizar descriptor/tabelas/mapas e
  transporte em comparação comum. Não contar mudança de cobertura como ganho.
- Ganho positivo de distorção macro e admissão não nula no treino permitem **uma**
  comparação predefinida em desenvolvimento fresco, sem escolher tolerâncias ou
  novos ganhos pela validação. Sucesso no treino não autoriza promoção nem treino
  adicional por si. Sem ganho, encerrar essa família de restauração.

Implementação: `compression/affine_restoration.py`,
`finetuning/affine_codec_study.py` e `finetuning/run_affine_side_probe.py`.
Qualquer inferência neural ou treinamento desta rodada usa a RTX 4060 local;
Testes e decodificação independente podem usar CPU. Nenhum compute remoto.

### Resultado do mecanismo no treino e protocolo de confirmação

O probe nos 135 recortes de treino passou o critério de avanço: a distorção
mista macro normalizada caiu de **0,393630** (seleção física sem restauração
afim sobre os códigos disponíveis) para **0,385939**, ganho de **1,954%**,
com FAS1 escolhido em 92 janelas. O controle de origem tem distorção 0,395368.
Nenhum parâmetro neural foi atualizado. Isso não é ganho percentual de RMSE nem
avaliação de todos os candidatos de busca. Artefatos:
`results/experiments/affine_side_train_probe_20261102/`.

Antes de inferir validação, fica fixada uma comparação nova, seed 2087, excluindo
recortes de todos os pilotos E1 anteriores registrados. Usar 48 passos e LR
histórico de treino, mesmas tabelas/decoder/teto por janela e **uma trajetória de
propostas compartilhada**. O seletor controle aceita apenas reconstruções brutas
fisicamente admissíveis; o seletor afim também pode escolher OLS/RMS unitário
transmitidos em FAS1 para cada candidato duro, inclusive os índices iniciais.
Ambos minimizam a mesma distorção ordinária física reexpressa na normalização
original, com as mesmas sete restrições (bytes, RMSE, correlação, p90, magnitude
FFT, fase e distância RMS a 1). Não retunar regras após ver a validação.

O tempo será o **custo combinado** da trajetória e dos dois seletores, não um
benchmark isolado por braço. Mostrar frames e stream com custos comuns explícitos
(contagens/descriptor/mapas/transporte); não há novos pesos. Conferir replay
exato e envelopes de escape. As duas hipóteses têm conjuntos aninhados de
candidatos, mas isso só garante a melhor distorção selecionada, não dominância
de cada métrica por configuração. A confirmação continua sendo desenvolvimento
em arquivos conhecidos. Alvo de investimento permanece ≥5% contra controle
pareado; uma melhora menor não autoriza treinamento longo ou promoção.

### Protocolo externo de stress, antes de ler sinais novos

Escolher por metadados seis arquivos SMF: Horizontal/Vertical × rótulos de
excitação 200/1.000/10.000 Hz, primeiro bloco e trecho central de um segundo
(`[500:20500]`, a 20 kHz). Os rótulos no nome **não são sample rate**. Preservar
todos os 551 canais da Zone1 fornecida, sem seleção de amplitude ou descarte
adicional. Tiling espacial 512+39 e vinte janelas temporais adjacentes de 1.000
amostras, sem lacunas. Essa cobertura é um stress full-zone, diferente da cobertura
filtrada histórica; não atribuir diferenças de scores somente a mudança de domínio.

**Todo o grupo SMF de 36 arquivos passa a desenvolvimento externo**, inclusive
arquivos não lidos. Não usar os restantes como holdout independente. Os grupos
EDAS Polyimide/Acrylate permanecem sem avaliação de formas de onda nesta campanha.
Manter todos os registros/orientações vinculados na mesma partição. Isso não é
validação externa final das quatro famílias nem prova de independência física.

Comparar os compactos/densos transferidos **já congelados**, sem retreino,
seleção de checkpoints ou hiperparâmetros por dados externos. O compacto é a
mesma origem usada acima. O denso local é `spatial_abc_fold1_20261026/high_A/best_macro.pt`,
com hash conferido contra `full_phase24_high_A/protocol.json`; métricas históricas
reproduzem ~0,416764 de RMSE. Seu hash de arquivo inclui metadados/otimizador e
não é o hash do arquivo weights-only usado pelo relatório 78. Declarar essa
identidade explicitamente, sem misturar formatos de checkpoint.

Cada modelo usa tabelas ajustadas somente nos 135 recortes de treino existentes,
congeladas antes da avaliação externa. Reportar reconstrução direta e a mesma
restauração afim estritamente guardada de duas alternativas fixas, com bytes
completos e custos compartilhados separados. São pontos de taxa diferentes, não
um contraste causal de densidade a taxa igual. Reconstruir o segundo contínuo
para métricas pooled por canal ao lado das métricas por janela; a máscara ordinária
pooled é a união das máscaras efetivamente codificadas, não uma nova regra de
escape escolhida depois de ver o resultado.

Para verificar transferência da **busca** afim sem uma campanha longa, fixar
também o índice temporal 10 (50 ms) de cada arquivo, ambos os tiles: 12 recortes.
Comparar os dois seletores físicos da mesma trajetória E1, com a mesma regra e
48 passos/LR já fixados. Não confundir esses 12 recortes com os resultados pooled
de um segundo. Sem novo treinamento ou tuning. As leituras Range de sinal têm
orçamento de 64 MiB por arquivo; não baixar o conjunto externo inteiro.

### Falha fechada na preparação e contrato numérico antes da inferência externa

A primeira preparação parou no primeiro trecho por valores não finitos, **antes
de inferência neural externa**. Protocolo/erro foram preservados em
`external_smf_stress_samples_20261102/`. Não trocar arquivos ou selecionar outros
instantes. A repetição como auditoria numérica, sem modificar os arrays, leu os
mesmos seis trechos e revelou **120.000 NaNs por trecho**, em seis canais inteiros
de 20.000 amostras; os demais 545 canais são finitos. O padrão float32 é
`0x7fffffff`, que não deve ser silenciosamente convertido no NaN canônico.
Artefatos: `results/experiments/external_smf_stress_samples_v2_20261102/`.

O FNE1 original rejeita não finitos por definição. Antes de inferir modelos,
fica registrado um **contrato numérico separado**, sem alterar FNE1 nem remover
linhas para melhorar scores: `FNX1` preserva os bits uint32 das células não
finitas/extremas e normaliza somente células ordinárias finitas. `FNZ1` permite
o mesmo side com índices delta-encoded e corpo comprimido por zlib nível 6,
fixo e lossless; escolher o menor entre corpo bruto e comprimido por bytes,
não por qualidade. `FAX1/FAZ1` são as restaurações afins correspondentes.
Ganho/offset continuam nos mesmos campos float64. Registrar custos reais e
contrafactual de side não comprimido. Não imputar, fazer clipping ou descartar
os seis canais, e não contar NaNs como erro físico zero: reportar sua cobertura
e preservação bit-exata separadas da fidelidade ordinária.

Apenas frames com algum alvo ordinário podem entrar nesse teste; um frame
inteiramente não finito falha fechado, sem fabricar correlação/RMSE. Guardas
continuam as mesmas, sobre a região ordinária e espectro normalizado mascarado,
sem promessa de espectro físico em canais indefinidos. O decoder verifica
dimensões, índices, limite/término da descompressão, overflow da região ordinária
e bits dos escapes. Entrada finita deve gerar **exatamente** o side FNE1 legado.
É um novo formato experimental de transporte, não um ganho de pesos neurais.

Implementação: `compression/numeric_restoration.py`. Os testes incluem NaNs com
payloads diferentes, ±Inf, bit-exatidão, corrupção/truncamento, compressão dos
índices e rejeição de frames sem alvo ordinário. Todo esse tratamento é comum
aos braços compacto/denso; a avaliação externa começa somente após esses testes.

### Refinamento analítico predefinido: OLS sob a própria restrição de RMS

O stress externo é desenvolvimento, não holdout final. O resultado sem admissão
das duas propostas afins motiva uma única hipótese adicional, não sweep de ganhos:
OLS irrestrito pode exigir RMS menor que o da origem, enquanto RMS unitário pode
exigir distorção maior. Para razão RMS de origem `a`, a guarda existente define
o intervalo `[max(0,1-|a-1|), 1+|a-1|]`. Propor o ganho OLS projetado nesse
intervalo, com offset que alinha a média. Essa é uma proposta analítica para MSE
com RMS limitado, não solução ótima de todas as guardas ou da perda mista.

Registrar antes de executar: primeiro repetir o probe **só no treino** com essa
terceira proposta, comparando ao resultado das duas propostas já congelado.
Usar margem numérica interna derivada de 32 epsilons float32, limitada a 1/4
do intervalo de ganho, para evitar rejeição por arredondamento de saída; **não
alterar nenhuma desigualdade da guarda efetiva**, que continua verificada sobre
os valores restaurados. Mesmo formato, mesmo teto, sem pesos novos. Se não houver
ganho de distorção macro contra o probe anterior, encerrar o refinamento.

Somente em caso positivo, comparar os três seletores (estrito sem afim, duas
propostas e conjunto com OLS limitado) numa trajetória compartilhada em mais
63 recortes predefinidos, seed 2089, excluindo também a confirmação seed 2087.
Não voltar ao grupo externo para chamar uma adaptação desenvolvida ali de
confirmação independente. A transferência desse refinamento ficará para protocolo
separado, sem expor os grupos EDAS reservados por conveniência.

**Retomada de 03/11/2026:** o probe limitado passou: distorção macro 0,385336,
contra 0,385939 das duas propostas, **0,156%** de ganho adicional. A confirmação
seed 2089 foi então executada nos 63 recortes novos previstos. Antes de reutilizar
o SMF, fica fixado o protocolo de transferência: repetir exatamente os 12 recortes
externos consumidos, mesmas tabelas/decoder, 48 passos/LR e guardas, comparando
os três seletores numa trajetória comum. Sem ajuste de margem ou nova frequência,
sem reavaliar EDAS, e sem chamar a reutilização de confirmação independente.
O novo diretório será `external_smf_bounded_transfer_20261103/`, em
`results/experiments/`, com script `finetuning/run_external_search_transfer.py`.

### Controle adicional de cobertura: protocolo antes da re-encodificação

O diagnóstico post-hoc encontrou 39 canais finais concentrando ~74–80% da
energia centrada do stress full-zone, enquanto a regra histórica de amplitude
rejeitaria 39–44 linhas finitas, além das seis linhas NaN. Antes de atribuir
diferenças de fidelidade à generalização, executar uma ablação congelada sobre
**o mesmo alvo retido nos dois lados**: (A) saída full-zone já decodificada,
restrita às linhas retidas; (B) re-encodificação dessas mesmas linhas, em regiões
contíguas separadas, pelos mesmos modelos/tabelas. Não unir lacunas no input.

Usar exatamente as decisões `persistent-amplitude-v1` já salvas pelo diagnóstico,
sem novos limiares ou seleção por erro do modelo. A observação é o segundo
disponível, não a dose de blocos do corpus histórico inteiro. Registrar mapas,
pixels, escapes e bytes dos frames retidos; **não declarar compressão da Zone1
inteira**, pois esta ablação não entrega as linhas excluídas. Os resultados
primários full-zone e seus bytes não mudam, e a heurística não prova que as
linhas excluídas sejam fisicamente inválidas. Comparar todos os valores finitos
do mesmo alvo em métricas pooled de um segundo.

Essa é uma ablação conjunta de normalização/contexto/padding, não isolamento
de um único mecanismo. Nenhum treino, ajuste de checkpoint ou calibração externa
das tabelas. Artefatos previstos: `external_smf_fixed_coverage_20261103/`, script
`finetuning/run_external_coverage_control.py`. Não expor EDAS nem chamar a
reutilização de SMF de confirmação independente.

### Particionamento com cobertura integral: protocolo antes da execução

O controle de cobertura mostrou ganho no alvo comum, mas não entrega a Zone1
inteira. A próxima hipótese é usar as fronteiras da heurística **apenas como
fronteiras de normalização**, codificando também a faixa de alta amplitude,
sem excluir suas amostras. Os seis canais inteiramente não finitos serão
fragmentos exatos; valores finitos ordinários continuam obrigatoriamente no caminho
neural, com os escapes numéricos finitos já previstos contabilizados à parte.

Fixar os mesmos seis segundos SMF, modelos/tabelas congelados e decisões de
amplitude já salvas. A comparação é: (A) empacotar os frames originais 512+39;
(B) empacotar regiões contíguas normalizadas separadamente: região restante,
faixa de alta amplitude e fragmento não finito. Não fazer busca de novos limites,
normalização por canal, ajuste afim ou treino. Conferir **a mesma contagem de
índices neurais** antes de admitir candidatos; diferenças de padding não podem
aumentar silenciosamente a informação transmitida.

O novo container **FPB1** transmite início/altura/tipo/tamanho de cada fragmento,
shape total e CRC. Toda linha deve ser entregue exatamente uma vez, sem lacunas
ou sobreposição. Fragmentos exatos usam bits float32 little-endian + zlib-6 e
são permitidos **somente para células inteiramente não finitas**, não como bypass
gratuito do sinal finito. Fonte e candidato usam o mesmo container, cobrando
framing, partições, índices, side e payloads exatos. Incluir tabelas/descriptor/
transporte no stream e separar os pesos compartilhados.

O encoder pode comparar ambos contra o original. Escolher o novo pacote somente
se a **janela completa 551×1.000** não piorar bytes, RMSE, correlação, p90 por
canal, magnitude FFT, fase ou distância RMS a 1. Caso contrário, transmitir o
pacote original. Preservar métricas e bytes sem guarda também. A guarda aqui
tem escopo do bundle completo; não alegar que ela protege cada tile original
512/39 separadamente. Manter avaliações regionais e pooled de um segundo.

Isso testa conjuntamente normalização, contexto e padding; a heurística não
prova invalidade física das faixas, por isso nenhuma é retirada. A leitura é
**desenvolvimento externo reutilizado**, não confirmação independente ou promoção
generalista. Critério exploratório: ≥5% de RMSE com guarda a bytes não maiores,
sem ignorar regressões concorrentes por arquivo. Não ampliar treinamento se o
critério não for atingido. Artefatos: `full_coverage_partition_smf_20261103/`;
implementação `compression/partition_bundle.py` e
`finetuning/run_full_coverage_partition.py`.

### Resultados encerrados da restauração afim

Na primeira confirmação fresca (seed 2087), os 63 recortes tiveram:

| Política | RMSE macro | Redução contra origem | Frames | Stream com auxiliares comuns |
|---|---:|---:|---:|---:|
| Origem | 0,647934 | — | 96.799 | 399.440 |
| E1 fisicamente guardado, sem afim | 0,640825 | 1,097% | 96.715 | 399.356 |
| E1 + duas propostas afins | 0,628681 | 2,971% | 96.610 | 399.251 |

A melhora afim contra o controle pareado foi **1,895% de RMSE**, com 39 frames
FAS1 selecionados: 34 OLS e cinco RMS unitário; todos também trocaram índices.
Ganhos de restauração selecionados ficaram entre 1 e 1,455714. Nenhuma regressão
das sete guardas foi admitida. A correlação macro passou de 0,749373 para 0,763099,
o p90 de 1,025864 para 0,999790 e a razão RMS de 0,753662 para 0,781857.
Isso é ganho sob critérios diagnósticos, não fidelidade física universal. O custo
combinado de buscar e avaliar os dois seletores foi ~196 s; sem pesos novos.
Artefatos: `affine_e1_fresh_local_20261102/`.

A confirmação do refinamento limitado (seed 2089), em **outros** 63 recortes,
comparou todos os seletores na mesma trajetória:

| Política | RMSE macro | Redução contra origem | Frames |
|---|---:|---:|---:|
| Origem | 0,645798 | — | 96.618 |
| Estrito sem afim | 0,639578 | 0,963% | 96.543 |
| Duas propostas afins | 0,625968 | 3,071% | 96.413 |
| Acrescentando OLS limitado | 0,621873 | 3,705% | 96.390 |

O ganho causal do refinamento no contraste pareado é **0,654% contra as duas
propostas**, não a diferença entre tabelas com recortes distintos. O custo
combinado foi ~195 s, pico alocado ~1,20 GB; FAS1 foi escolhido em 52 janelas
no conjunto ampliado, contra 43 do seletor de duas propostas. Os custos comuns
somam 302.657 bytes nos três seletores; decoder/codebook/contagens são os mesmos.
Nenhum novo peso ou campo extra por frame. Artefatos:
`bounded_affine_e1_fresh_local_20261103/`. Continuam abaixo do alvo de investimento
de 5% contra o controle; não promover nem iniciar destilação longa por esse ganho.

Uma primeira chamada do runner interrompeu **antes de carregar modelos ou inferir**
por um campo histórico `selection` que era texto, não mapa de janelas. O parser
foi corrigido e testado; a execução acima começou em diretório novo ainda vazio.
Não houve reinício oculto de treino ou escolha de hiperparâmetros pela falha.

### Resultados encerrados do stress SMF com cobertura integral

Foram avaliadas **66.120.000 células por modelo**, seis segundos contínuos no
total, com 240 tiles/janelas por modelo. Os **720.000 NaNs** são protegidos como
bits exatos; as demais 65.400.000 células são ordinárias. As tabelas dos dois
modelos foram ajustadas nos mesmos 135 recortes antigos de treino, sem dados
externos. Entrada, máscaras e conteúdo transmitido são comuns entre modelos.

| Modelo congelado | RMSE pooled de 1 s por arquivo | Correlação pooled | P90 pooled por canal | Stream total | Razão sobre float32 |
|---|---:|---:|---:|---:|---:|
| Compacto | 0,8323–0,8447 | 0,5497–0,5637 | 0,9902–1,0333 | 1.327.548 bytes | 199,22× |
| Denso | 0,5196–0,5301 | 0,8506–0,8567 | 0,7178–0,7732 | 5.080.940 bytes | 52,05× |

O denso transmite mais informação; não é vitória a taxa igual. Ambos preservam
todos os 551 canais, incluindo a realização exata dos NaNs. Pesos de waveform
compartilhados ficam separados: **111.832.324 bytes em tensores** por modelo;
os arquivos de checkpoint podem ser maiores por otimizador/metadados. A taxa
inclui frames, tabelas deriváveis das contagens, descriptor, mapas e transporte.
Não transportar essas razões para outro alvo/cobertura ou chamá-las de alta
fidelidade operacional.

A média macro por janela/tile foi 0,829965 no compacto e 0,674574 no denso:
**não é a mesma agregação** do RMSE pooled de um segundo. Não escolher a que
parece mais favorável. A restauração afim sem busca não admitiu mudança em nenhum
dos 240 frames de cada modelo, sob as guardas fixadas. O subset E1 de 12 recortes
também não admitiu melhoria nos seletores estrito ou de duas propostas. A
reutilização posterior com OLS limitado admitiu ganho macro de apenas **0,186%**
(0,830174 → 0,828627), com 63.355 → 63.337 bytes de frames. Não é confirmação
independente nem justificativa para retunar contra SMF. Artefatos:
`external_smf_frozen_stress_20261102/` e
`external_smf_bounded_transfer_20261103/`.

O side numérico custou **18.720 bytes**, contra **5.769.600 bytes** se os mesmos
escapes fossem transportados sem compressão de índices/bits; esse efeito é de
serialização lossless, **não de ganho neural**. Todos os métodos pagam o mesmo
contrato numérico. O reader independente em CPU decodificou FAS1 e um frame
FAZ1 externo, preservando 6.000 NaNs no frame de 39×1.000, sem NPZ/HDF5 original.
Nenhuma versão nova substitui silenciosamente o FNE1 ou os alvos de inferência
operacionais existentes; os pacotes exigem o descriptor/reader compatível.

### Diagnóstico de cobertura e controle de entrada, sem mudar o resultado primário

O diagnóstico fixo de amplitude rejeitaria 39–44 linhas finitas, e seis linhas
inteiramente NaN ficariam fora da região modelada histórica. Restariam 501–506
linhas. Os 39 últimos canais da partição original concentram **74,05–79,51%**
da potência centrada; portanto ponderação por células e por energia respondem
a perguntas diferentes. Não interpretar full-zone versus corpus filtrado como
um contraste puro de generalização. A heurística não demonstra invalidade física.

Depois de controlar a energia das primeiras diferenças do alvo, as costuras
temporais tiveram erro relativo menor que o interior no compacto (~0,76–0,83
contra ~0,87–0,88), mas maior no denso (~0,59–0,66 contra ~0,53–0,54).
Isso é diagnóstico, não prova de causa ou de que mais contexto dará ganho global.
Arquivo: `external_smf_frozen_stress_20261102/coverage_seam_diagnostic.json`.

A ablação de entrada, **avaliando exatamente as mesmas linhas retidas nos dois
lados**, concluiu os seis casos com ambos os modelos. Re-encodificar essas linhas
separadas da faixa de alta amplitude reduziu o RMSE em **7,82–9,85% no compacto**
e **9,18–16,59% no denso** contra as saídas full-zone restritas ao mesmo alvo.
Os erros finais ficaram em 0,7330–0,7815 e 0,5088–0,5581, respectivamente.
Logo, a preparação/contexto de entrada importa neste grupo; isso não isola
normalização de padding/geometria, e não entrega as faixas excluídas.
**Não usar esses ganhos como compressão da Zone1 inteira.** Artefatos:
`external_smf_fixed_coverage_20261103/`. O teste seguinte de cobertura integral
foi registrado separadamente acima antes de executar.

### Controle denso a taxa fixa: protocolo após o particionamento

O particionamento integral concluiu sem admissões. Antes de abrir qualquer
treino ou ajuste de prior, testar um controle de codificação a taxa fixa no
**denso**: mesmos seis arquivos e janela 10 já consumidos, mesmas partições
originais 512+39, mesmas tabelas e pesos. Transferir o método E1 fixo de 48 passos
e LR histórico, sem busca de LR específica do denso. Resultado negativo não será
interpretado como ótimo do decoder ou refutação da família.

Cada fragmento conserva as sete guardas e seu teto. Para cada seletor
estrito/afim/limitado, avaliar também as combinações de fragmentos originais ou
propostos no bundle 551×1.000, escolhendo menor distorção entre as que preservam
**também** as guardas do bundle completo. Carregar a melhor solução do seletor
anterior para manter conjuntos aninhados. O container fonte/candidato é FPB1
com duas entradas nos dois lados; nem partições nem contagem de índices aumentam.
O custo de busca será reportado, sem comparação de latência com uma única
inferência. Sem treinamento, novos exemplos, EDAS ou tuning da validação.

Artefatos previstos: `dense_e1_bundle_control_smf_20261103/`; código
`finetuning/run_dense_e1_bundle_control.py`. É controle de desenvolvimento externo
reutilizado, não confirmação independente ou evidência pooled de um segundo.
Sem ganho relevante, encerrar essa rodada em vez de repetir continuações.

### Resultado do particionamento integral e do controle denso

O FPB1 entregou todas as 551 linhas sem lacunas e preservou os bits não finitos.
Os números de índices neurais foram **idênticos** entre fonte e candidato em
todas as janelas: 8.820 no compacto e 35.000 no denso. Fonte usa dois fragmentos;
candidato usa três, dos quais um exato de 46 bytes para os NaNs, além do framing.
Não houve aumento oculto de grade, pesos ou exclusão da faixa de alta amplitude.

| Modelo/política | RMSE macro da janela completa | Razão RMS | Bytes dos bundles | Stream com auxiliares |
|---|---:|---:|---:|---:|
| Compacto/origem | 0,839695 | 0,669760 | 1.270.920 | 1.312.451 |
| Compacto/particionado sem guarda | 0,853173 | 0,606392 | 1.336.739 | 1.378.270 |
| Denso/origem | 0,525283 | 0,917791 | 5.024.312 | 5.065.843 |
| Denso/particionado sem guarda | 0,496950 | 0,851424 | 5.283.031 | 5.324.562 |

Essa agregação é de janelas completas 551×1.000, não a macro de tiles 512/39
da tabela anterior. Não comparar a diferença entre agregações como ganho do método.
O particionado compacto piorou RMSE em 1,605% e aumentou bundles em 5,179%.
O denso reduziu RMSE em **5,394%**, mas aumentou bundles em **5,149%** e piorou
RMS em todas as 120 janelas; a fase piorou em 11. O p90 melhorou nos dois modelos,
mas isso não compensa automaticamente os outros critérios. Nenhuma janela foi
admitida por qualidade ou por orçamento em qualquer modelo: **0/120** em cada
teste. A política guardada conserva os resultados/bytes originais, sem promoção.

O custo extra é real mesmo com igual contagem de índices: mudaram frequências de
símbolos/entropia e framing. A mudança de representação não é uma economia só
porque o número nominal de bits de índice parece igual. Artefatos:
`full_coverage_partition_smf_20261103/`. O reader CPU também decodificou um bundle
candidato completo, com 6.000 NaNs exatos no quadro 551×1.000, sem fonte NPZ/HDF5.

O controle E1 denso a taxa fixa concluiu os seis bundles do instante central,
com as mesmas partições e dados. O RMSE foi **0,525745 → 0,524733**, ganho de
**0,1925%**, e os bundles somaram **251.400 → 251.392 bytes**. A versão sem afim
ganhou apenas 0,00018%; as duas propostas afins não acrescentaram ganho sobre ela;
o OLS limitado produziu a pequena diferença restante. Custo combinado ~222 s,
pico alocado ~2,90 GB. Não é evidência pooled de um segundo, ótimo do decoder
ou confirmação independente. Artefatos: `dense_e1_bundle_control_smf_20261103/`.

**Decisão da rodada:** preservar a restauração afim como opção experimental de
codificação, sem novos pesos, mas não promover ou ampliar destilação/treino com
ganho interno de até 3,70% e transferência SMF de ~0,19%. O benefício no alvo
retido não se transferiu para uma solução de cobertura integral sob as guardas
e o teto de bytes. Não remover a faixa dominante para salvar o número nem
relaxar RMS/fase/bytes retrospectivamente. A próxima decisão deve resolver
validade/cobertura física das faixas e contrato de fidelidade; uma futura
representação ou prior precisa de hipótese própria e confirmação multiconfiguração.

### Fechamento auditado da rodada afim/full-zone — 03/11/2026

Esta rodada não atualizou parâmetros neurais: os pesos compacto/denso e suas
identidades permaneceram intactos. Propostas E1 foram otimizadas na RTX 4060
local; OLS e restauração são operações do compressor, com coeficientes transmitidos,
não treinamento compartilhado em dados externos. Nenhum compute remoto, commit,
push ou alteração do staging do usuário foi realizado. Não há treino/busca
deixados em execução.

`finetuning/audit_affine_round.py` verificou **52 hashes de snapshots**, **2.253
frames neurais avulsos**, **1.488 frames neurais incorporados**, **744 bundles
FPB1**, **240 fragmentos exatos não finitos** e **1.359 decisões de guarda**.
Os números contam verificações/arquivos, incluindo fallback e empacotamentos,
não amostras independentes. Conferiu preservação de 120.000 NaNs por arquivo
externo e confirmou os dois grupos EDAS sem avaliação de sinal. Resultado:
`results/experiments/affine_round_audit_20261103.json`.

Os leitores CPU independentes restauraram FAS1, FNZ1/FAZ1 e FPB1, inclusive o
bundle denso completo, sem carregar o NPZ/HDF5 original. Não se presume igualdade
bit a bit de **predições neurais ordinárias** entre CPU/GPU; os escapes exatos
não dependem dessa igualdade. A suíte completa passou em **311 testes** com
CUDA desabilitada. A versão FNE1 finita original permanece inalterada, e a nova
família de formatos tem semântica/reader explícitos, sem adoção silenciosa.

**Prioridade após os resultados:** resolver o alvo/cobertura física e os limites
concorrentes antes de outra campanha longa. Manter afim limitado como referência
experimental, não destilar automaticamente o ganho interno nem buscar novos
limiares no SMF consumido. Se houver novo contraste neural, ele deve testar um
mecanismo realmente distinto — por exemplo fatoração da grade/cardinalidade a
bytes reais pareados, com inicialização/exposição controladas — e não repetir
largura, samplers ou ganhos de side. Isso é hipótese futura, não experimento já
agendado ou promessa de atingir 700× com alta fidelidade.

## 10. Triagem de grade × cardinalidade e controles a bytes reais

### 10.1 Hipótese e teste realizado

A continuidade testou se distribuir o orçamento em **mais posições com menos
bits por símbolo** melhora reconstrução/compressão, em vez de ampliar largura,
taxa ou remover faixas difíceis. Foram concluídos quatro braços de **1.024
updates** cada, em uma única RTX 4090 remota, conforme autorização atual.

Todos partiram dos mesmos pesos não quantizadores do compacto fold1. O controle
compact2048 preservou o codebook; K32/K64 foram inicializados por farthest-first
+ 12 passos Lloyd sobre vetores do encoder, somente de arquivos de treino de
ajuste (até 256 vetores/recorte). A grade densa adiciona redução temporal menor
à geometria spatial-4x; muda contexto/padding e compute, não só contagem.

Mesma seed 2090, sequência de recortes, AdamW novo LR 1e-4, wd 1e-5, batch1,
clip1, FP32/TF32 desligado, objetivo 0,5 MSE + 0,5 L1 + VQ. A seleção usou
arquivos internos de treino excluídos do novo ajuste/calibração, nos passos
0/256/512/1024. Todos selecionaram 1024. A origem já viu esses arquivos em
pré-treinamento: **não são holdout independente**. O seletor de distorção usa a
média histórica por tile/configuração/família; a avaliação física pondera células
nos grupos. Não confundir essas agregações nem ganho interno com generalização.

A avaliação pareada compreendeu os mesmos **63 recortes de desenvolvimento**,
quatro famílias/seis configurações e **4.059.000 células**. Não se declarou
validação virgem; nenhum sinal test, EDAS ou SMF foi usado nesta triagem.
Artefatos: `results/experiments/alphabet_factorization_fold1_20261104/`.

### 10.2 Resultado de fidelidade e bytes

| Braço | Posições totais | Bits nominais/símbolo | RMSE macro | Variação RMSE vs controle | Frames reais | Stream com tabelas/descriptor/mapas |
|---|---:|---:|---:|---:|---:|---:|
| Compacto K2048 | 73.332 | 11 | 0,649693 | — | 96.688 | 131.481 |
| Compacto K64 | 73.332 | 6 | 0,734676 | +13,080% | 49.968 | 68.885 |
| Denso K64 | 145.500 | 6 | 0,685714 | +5,544% | 96.604 | 115.537 |
| Denso K32 | 145.500 | 5 | 0,718390 | +10,574% | 78.676 | 97.353 |

O contraste de payload mais próximo foi **denso K64 versus compacto K2048**:
apenas 84 bytes a menos nos frames (−0,087%), porém RMSE **5,544% pior**.
A economia de stream chegou a 12,126% porque o alfabeto reduzido também diminui
as tabelas transmitidas; **não é taxa total exatamente igual**. O compacto K64
economizou 48,320% dos frames e o denso K32 18,629%, mas ambos perderam fidelidade.
Esses são pontos de trade-off, não manutenção/melhora simultânea dos dois eixos.

FAK1 é experimental: transmite índices com bit-packing ou range marginal com
fallback, shape/side e fingerprints/CRCs; contagens são ajustadas somente no
treino após selecionar os pesos. Todos usam o mesmo coder/contrato. A contagem
de símbolos e os bytes **medidos** são separados; a grade densa não tem exatamente
2× posições devido ao padding. Não extrapolar os bytes desta seleção de 63
recortes para arquivos contínuos nem misturar com taxas dos estudos anteriores.

Houve regressão de RMSE nas **seis configurações** de todos os braços reduzidos:
2,35–14,85% no denso K64, 9,12–17,90% no compacto K64 e 5,53–21,78% no denso K32.
No denso K64 o p90 melhorou em duas configurações, mas piorou nas outras quatro;
a correlação macro caiu de 0,749290 para 0,716486 e a razão RMS de 0,776029 para
0,736291. Logo, o resultado negativo não é apenas um efeito da média macro.

Nenhum dos 63 frames candidatos de cada braço reduzido passou conjuntamente
bytes e guardas físicas contra o **controle também treinado**. O fallback
preservou fidelidade, mas não trouxe ganho: os streams roteados custaram
132.812/132.828/132.572 bytes, contra 131.544 do controle com o mesmo byte de rota.
Os dois registries/tabelas são cobrados no fallback, não tratados como gratuitos.
Pesos em tensores: 111.832.324 bytes no controle e ~110,8 MB em cada reduzido;
o fallback exige os dois modelos. Nenhum candidato foi promovido.

### 10.3 Aprendizagem, custo e limites do negativo

| Braço | Distorção interna passo 0 | Passo 512 | Passo 1024 | Compute de treino sincronizado |
|---|---:|---:|---:|---:|
| Compacto K2048 | 0,409752 | 0,413982 | 0,409006 | 53,52 s |
| Compacto K64 | 0,526208 | 0,505754 | 0,503854 | 53,25 s |
| Denso K64 | 0,602728 | 0,453165 | 0,444957 | 58,16 s |
| Denso K32 | 0,623346 | 0,490023 | 0,479929 | 58,09 s |

Tempo de treino + checks foi ~58–63 s por braço; pico alocado ~2,78–3,07 GB.
O smoke separado executou oito updates adicionais. Nenhum benchmark pareado da
4060 foi feito, portanto esses tempos **não medem diretamente** aceleração entre
GPUs. Todos os 64/32 códigos apareceram na calibração dos alfabetos pequenos;
o controle usou 1.227 códigos dos 2.048 disponíveis.

O denso K64 ainda melhorava no check interno entre 512 e 1024. Isso impede
concluir que grade/cardinalidade sejam estruturalmente inviáveis ou que os
modelos tenham convergido. Também houve inicialização/legado de pré-treinamento
distintos para o quantizador, embora os pesos não quantizadores fossem iguais.
O resultado demonstrado é mais restrito: **a adaptação curta prescrita não
produziu candidato com fidelidade e bytes admissíveis**. Não atingiu o critério
pré-definido de ≥5% de ganho guardado para ampliar treino; não estender por
melhora retrospectiva de uma curva nem relaxar as guardas para salvar o braço.

### 10.4 Controle clássico no orçamento real de cada ponto

Para contextualizar os pontos baratos, o DCT esparso existente foi codificado
em CPU nos mesmos 63 alvos, com **teto por frame igual aos bytes FAK1 medidos**,
incluindo índices dos coeficientes, escala, side e framing. Precisão fixada em
oito bits, sem sweep. O método recebeu exatamente o mesmo packet robusto/máscara;
escapes são restaurados bit a bit. `compress_packet` preserva o comportamento
legado de `compress(raw, ...)` e evita renormalizar/proteger outro alvo.

| Orçamento do braço | RMSE neural | RMSE DCT | Frames neurais | Frames DCT |
|---|---:|---:|---:|---:|
| Compacto K2048 | 0,649693 | 0,825867 | 96.688 | 96.632 |
| Compacto K64 | 0,734676 | 0,855765 | 49.968 | 49.908 |
| Denso K64 | 0,685714 | 0,825884 | 96.604 | 96.540 |
| Denso K32 | 0,718390 | 0,836461 | 78.676 | 78.618 |

O controle clássico não resolveu a fidelidade média nesses tetos. Houve algumas
vitórias locais concorrentes do DCT (0/1/2/3 janelas, respectivamente), portanto
não há dominância universal da rede. O DCT usa ligeiramente menos bytes e nenhum
peso/tabela aprendido; a tabela compara frames, **não igualdade exata de custo
total de implantação**. Não é codec clássico otimizado nem limite inferior.
Artefatos: `alphabet_matched_dct_fold1_20261104/`; zero novos updates.

A auditoria remota da triagem verificou **1.186 hashes de snapshots/inputs**,
**252 frames**, **252 decisões de rota**, separação por arquivo no check e
**4.096 updates**, sem modificar a origem. O decoder independente restaurou
FAK1 K64 em CPU, sem ler o waveform original. Pesos selecionados/optimizer
permanecem no remoto; resultados, protocolos, snapshots e frames também estão
espelhados localmente. O diagnóstico congelado de quantização é registrado à
parte; sua saída contínua não é um codec nem pode ser contabilizada como ganho.

### 10.5 Diagnóstico congelado: quantização versus adaptação

Nos mesmos pesos selecionados, comparar o decoder alimentado pelos vetores
hard transmitidos com o decoder alimentado pelo encoder **sem quantização**:

| Braço | RMSE hard de desenvolvimento | RMSE contínuo diagnóstico | Distorção interna hard → contínua |
|---|---:|---:|---:|
| Compacto K2048 | 0,649693 | 0,625252 | 0,409006 → 0,383225 |
| Compacto K64 | 0,734676 | 0,647649 | 0,503854 → 0,406684 |
| Denso K64 | 0,685714 | 0,649151 | 0,444957 → 0,408613 |
| Denso K32 | 0,718390 | 0,697537 | 0,479929 → 0,464762 |

A distorção interna é a mesma mistura normalizada MSE/L1 do seletor de treino,
não RMSE físico. Há tiles constantes no check: suas razões RMSE relativas podem
explodir por denominador quase zero. Não usar essas razões para interpretar
aprendizagem ou escolher um braço. O diagnóstico v2 acrescenta a distorção
normalizada apropriada, preservando o v1 e sem alterar pesos/seleção. Os replays
hard dos 63 recortes reproduziram exatamente as métricas da triagem original.

O compacto K64 tem penalidade de quantização maior; no denso K32, até a saída
contínua continua pior. Assim, **não há evidência de um gargalo exclusivo e
uniforme de codebook**: adaptação/contexto/representação também importam.
O contínuo do denso K64 quase empata com o controle hard, mas continua pior que
o contínuo do próprio controle. Isso não demonstra vantagem de grade densa nem
justifica transmitir vetores sem cobrar seus bits. A entrada contínua pode estar
fora da distribuição vista pelo decoder; não é ótimo teórico, intervenção causal
completa ou codec de taxa equivalente.

Artefatos: `alphabet_frozen_gap_v2_fold1_20261104/` e versão inicial preservada
`alphabet_frozen_gap_fold1_20261104/`. Zero updates adicionais; inferência congelada
na mesma RTX 4090. **Decisão:** encerrar a triagem sem promoção ou extensão longa.
O DCT e o diagnóstico contínuo foram etapas adicionais para interpretar o negativo,
não formas de trocar o alvo, métrica ou orçamento até declarar sucesso.

**Próximos passos por evidência:** manter os checkpoints-controle e os pontos de
trade-off como referências. Antes de nova campanha, uma proposta de quantizador
estruturado ou de inicialização/convergência deve mostrar no treino como reduz
a penalidade de quantização **sem piorar a representação contínua**, com teto
de bytes real, exposição e controle comparáveis. Não repetir K32/densidade ou
mais updates sem esse mecanismo. Cobertura/requisitos físicos e confirmação
independente continuam necessários para promoção; nenhum destes resultados
autoriza sacrificar fidelidade para anunciar maior razão de compressão.

Fechamento: **317 testes passaram** na cópia local, incluindo bitstreams de
alfabetos variáveis, inicialização determinística em CPU e compatibilidade
legada/proteção robusta do DCT. Além da auditoria da triagem, foram verificados
os dez hashes de snapshots dos controles/diagnósticos e decodificados/reavaliados
os **252 frames DCT** contra os scores arquivados. Há 4.096 updates de produção
+ oito de smoke nesta série; os diagnósticos seguintes não atualizaram pesos.
Não há continuação de treinamento agendada por estes resultados.

## 11. Quantização por subespaços com rede congelada

### 11.1 Testes de treino: subespaços diretos e rotação PCA

O diagnóstico anterior mostrou penalidade discreta relevante. A etapa seguinte
isolou o quantizador: **o mesmo encoder/decoder do compacto K2048 selecionado**
ficou congelado; cada vetor de 128 coordenadas foi representado por dois códigos
de 64 coordenadas. K32/K64 por subespaço exigem nominalmente 10/12 bits por
posição, contra 11 do original. A capacidade combinatória não é informacionalmente
gratuita: todos os índices, codebooks e parâmetros compartilhados são cobrados.

Os centroides foram ajustados somente nos **113 recortes de treino de ajuste**,
com 12 iterações Lloyd; tabelas marginais também foram ajustadas nesses arquivos.
Os **22 recortes de arquivos de treino separados** foram usados como check,
sem novos sinais de validação. A origem já viu esses arquivos historicamente;
isso não é confirmação independente. Não acessar EDAS/SMF/test.

| Quantizador | Distorção normalizada macro no check | Frames reais | Stream candidato sem pesos compartilhados |
|---|---:|---:|---:|
| VQ original K2048 | 0,409006 | 61.527 | 84.894 |
| Produto 2×32, coordenadas originais | 0,512000 | 53.492 | 60.795 |
| Produto 2×64, coordenadas originais | 0,482519 | 64.308 | 71.867 |
| Produto 2×32, PCA balanceada | 0,448460 | 53.748 | 61.281 |
| Produto 2×64, PCA balanceada | 0,422618 | 66.456 | 74.245 |

A referência acima é **sem** os 22 bytes de rota; com rota seu stream tem
84.916 bytes. Os candidatos foram decodificados dos índices hard transmitidos,
sem usar latentes contínuos no receptor. FAK1 usa uma grade de símbolos de
largura duas vezes a grade latente, com grupos intercalados e descriptor próprio;
o reader precisa interpretar essa semântica, não inferi-la das formas dos pesos.

A rotação não foi escolhida por reconstrução. O diagnóstico nos **28.860 vetores
amostrados do treino de ajuste** encontrou 99,9850% da variância em oito
componentes, rank efetivo 5,32 e massa quadrática de covariância entre metades
de 0,4895. Isso satisfez o critério exploratório pré-fixado para um único controle
PCA balanceado (massa cruzada ≥0,10 ou variância top32 ≥0,80). Os eixos principais
foram distribuídos alternadamente entre subespaços; sem whitening, OPQ, ajuste de
rotação pela perda de waveform ou varredura de limites.

A PCA reduziu o erro latente relativo do 2×64 de **0,389118 para 0,156037**,
abaixo de **0,183450** do VQ original nos mesmos checks. Mesmo assim, sua
distorção de reconstrução permaneceu **3,328% pior**, com mais bytes de frames.
Logo, **menor distância latente não garante menor erro do sinal restaurado**.
Isso é um resultado diagnóstico, não um codec bem-sucedido nem prova de que o
erro latente seja irrelevante em toda representação.

Nenhuma das 22 janelas de qualquer candidato passou conjuntamente o teto de bytes,
as guardas físicas inalteradas e a redução de distorção. O fallback manteve os
resultados originais, sem ganho, e acrescentou custos de registro/tabela. O
critério de avanço (≥1% de ganho guardado no check e stream não maior) não foi
atingido; **nenhuma validação de sinais foi aberta para esses candidatos**.

Custos compartilhados adicionais: 16.384/32.768 bytes em centroides para K32/K64;
no PCA, mais **66.048 bytes** em matriz e centro float32, além dos cabeçalhos
dos arquivos. Eles não estão escondidos na coluna de stream e não são gratuitos:
o modelo original é retido para fallback, e a amortização depende da quantidade
de dados. Nenhuma economia de tabelas autoriza ignorar o custo do quantizador.

Artefatos: `product_vq_train_probe_20261104/` e
`product_vq_pca_train_probe_20261104/`, em `results/experiments/`. Encoder, decoder
e codebook original não foram atualizados. A auditoria inicial verificou 143
hashes, 66 frames e separação por arquivo; o reader CPU reconstruiu um frame
produto sem ler o waveform original. O diagnóstico de covariância não é uma
medida de fidelidade ou redundância já acessível ao decoder.

**Implicação:** justificar um contraste curto entre centroides ajustados por
distância latente e por perda de reconstrução através do decoder congelado,
mantendo PCA e orçamento fixos. Não justificar mais densidade, remoção de canais
ou extensão longa dos braços anteriores. O resultado dessa intervenção é
registrado separadamente; o probe PCA negativo não é reclassificado como ganho.

### 11.2 Ajuste pareado dos centroides pela perda do sinal

Foi executado o contraste seguinte na RTX 4090, **256 updates por braço**:
mesmos codebooks PCA 2×32 iniciais, mesma sequência seed 2092 e Adam novo
LR 1e-3, sem weight decay, clip1, FP32 e batch1. Somente os **4.096 valores de
centroides** foram treináveis. Encoder, decoder, codebook original, matriz PCA
e centro permaneceram congelados. As atribuições nearest hard foram recalculadas
a cada passo; o receptor continua recebendo apenas índices e assets compartilhados.

- **Controle latente:** minimiza MSE entre vetor quantizado e vetor do encoder.
- **Waveform:** minimiza a mistura normalizada 0,5 MSE + 0,5 L1 através do decoder
  congelado; não adiciona side afim, código residual ou pesos novos no decoder.
- Mesmos 113 recortes de ajuste e 22 checks, separados por arquivo;
  nenhuma leitura de sinais de validação. Escolher menor distorção de reconstrução
  no check nos passos 0/64/128/256, com passo zero elegível nos dois braços.
- Tabelas reestimadas apenas nos arquivos de ajuste após a seleção. Mesmos
  critérios de bytes/guardas físicas dos probes anteriores, sem relaxamento.

| Braço | D inicial | D no passo 64 | D no passo 128 | D selecionada, passo 256 | Frames candidatos |
|---|---:|---:|---:|---:|---:|
| Controle latente | 0,448460 | 0,448389 | 0,448450 | 0,448192 | 53.828 |
| Waveform | 0,448460 | 0,448274 | 0,448216 | 0,448172 | 53.660 |

O ganho de waveform sobre o controle pareado foi somente **0,00448% de D**;
contra a inicialização PCA, **0,06405%**. Não são percentuais de RMSE físico
ou de validação. O candidato waveform continuou **9,576% pior** em D que o VQ
original (0,409006) nos mesmos checks. Sua economia de frames não compensa essa
perda quando o objetivo exige manter/aumentar fidelidade.

O controle latente teve MSE latente relativo 0,231995; waveform, 0,235818.
O pequeno ganho de reconstrução veio apesar do erro latente um pouco maior,
coerente com o diagnóstico de que os objetivos não são equivalentes. A magnitude,
porém, ficou muito abaixo do critério pré-fixado de avanço, não justificando
prolongar esse ajuste curto ou varrer LR/iterações.

**Zero admissões em 22 janelas por braço.** O fallback conservou a reconstrução
original e custou 86.226 bytes de stream em cada braço, contra 84.916 do controle
original roteado. Os streams candidatos sem fallback custaram 61.361/61.193 bytes,
mas não mantiveram a fidelidade. Custos compartilhados adicionais continuam
**16.384 bytes** de centroides + **66.048 bytes** de transformação, mais cabeçalhos
dos arquivos, além do modelo original retido. Não ocultar esses assets em uma
razão de compressão obtida apenas dos frames.

Tempo de treino/checks foi ~3,74 s no controle e ~6,84 s no waveform, com pico
alocado ~0,54/2,24 GB. O controle latente não retropropaga pelo decoder e custa
menos compute; dose de recortes/updates, não tempo, foi pareada. A etapa de
codificação/verificação é separada. Houve quatro updates adicionais de smoke.
O orçamento curto não caracteriza ótimo ou convergência do quantizador.

A auditoria confirmou **512 updates somente de centroides**, hash do estado
completo da rede original inalterado, hashes de PCA/centro preservados, seleção
train-only, optimizer/RNG arquivados e **66 frames** fonte/candidatos com rotas
e replay dos símbolos corretos. Foram verificados 287 hashes de snapshots/inputs
na cadeia treino + avaliação; o smoke foi auditado separadamente. O reader CPU
restaurou um frame **512×1.000** do candidato waveform, com rotação e codebooks
transmitidos, sem ler o waveform original. Não se exige igualdade bit a bit das
predições ordinárias entre CPU/GPU; os índices e escapes são exatos.

Artefatos: `decoder_aware_pq_train_20261104/`, sua subpasta `evaluation/` e o smoke
`decoder_aware_pq_smoke_20261104/`, em `results/experiments/`. Nenhuma confirmação
de desenvolvimento foi aberta porque o critério de avanço falhou.

**Decisão:** encerrar estes braços sem promoção ou extensão automática. A PCA
explicou parte da falha dos subespaços diretos, mas não a resolveu sob fidelidade
e bytes concorrentes; o ajuste decoder-aware pequeno também não resolveu.
Não concluir impossibilidade da quantização por produto nem que o DAS bruto
tenha rank 5,32: esse rank é apenas da distribuição de vetores latentes amostrada.
Uma próxima proposta precisa apresentar mecanismo distinto e controle a taxa
pareada, não reciclar os negativos com mais updates, mais bits ou métrica trocada.

Fechamento desta continuidade: **322 testes locais passaram**, incluindo gradientes
restritos aos centroides e replay da rotação ortogonal. Protocolos, curvas,
checkpoint retomável dos centroides, seleção, frames e auditorias estão preservados.
O orçamento foi 512 updates de centroides + quatro de smoke; zero novos updates
de encoder/decoder e zero novos sinais de validação. Não há job ou extensão
de treinamento deixados em execução por estes resultados.

## 12. Controle preditivo com erro limitado e taxa medida

### 12.1 Mecanismo e escopo

Após os negativos do quantizador, foi testado um controle **sem rede e sem pesos
aprendidos**, diferente do DCT: quantização escalar uniforme do alvo protegido,
diferenças inteiras temporais ou espaço-temporais de Lorenzo, planos de bytes
e zlib nível 6. O formato experimental **FPI1** transmite shape, passo, preditor,
side de normalização/escapes, payload e CRC. O decoder não precisa do original,
de latentes contínuos ou de dados vizinhos não transmitidos.

Os três preditores (identidade/temporal/Lorenzo 2D) produzem **a mesma grade
quantizada**; escolher o menor frame não seleciona pela fidelidade. A inversão
por somas cumulativas é inteira, com limites de overflow e descompressão; os
escapes mantêm bits exatos. Nenhuma faixa/canal foi retirada para melhorar taxa.

Foram usados os mesmos **22 recortes de treino**, **2.909.000 células**, da etapa
anterior. As métricas hard e os frames do VQ original foram reutilizados com
checagem de shape/side/identidade. Nenhum novo sinal de validação, EDAS, SMF ou
test foi lido. Toda esta etapa foi CPU, **zero updates**; não precisou de GPU.

### 12.2 Comparação no orçamento dos frames VQ

O encoder enumerou passos fixos de precisão (epsilon de 2^4 a 2^-16) e escolheu
o menor epsilon cujo **frame medido** coubesse no teto da janela VQ. A escolha
não usa RMSE nem assume monotonicidade dos bytes. A primeira grade deixou 38,0%
do orçamento sem uso; por isso foi preservada e acrescida de um controle de
taxa refinado, com **24 propostas de bisseção geométrica por bracket elegível**,
usando somente bytes e retendo a menor precisão observada que coubesse. É uma
heurística de controle de taxa, não prova de ótimo global.

| Método | Frames candidatos | Distorção normalizada macro | RMSE macro, apenas alvos variáveis |
|---|---:|---:|---:|
| VQ de referência | 61.527 | 0,409006 | 0,665741 |
| Preditivo, grade inicial | 38.138 | 0,699117 | 0,898627 |
| Preditivo, controle refinado | 61.273 | 0,656434 | 0,866100 |

O refinamento utilizou **99,587%** do orçamento agregado dos frames, respeitando
também cada teto individual. Ainda assim, **zero das 22 janelas** passou as
guardas físicas concorrentes mais redução de distorção. O resultado negativo
não depende apenas do subaproveitamento grosseiro de taxa da primeira grade.
O fallback preservou o VQ, mas aumentou o stream de **84.916 para 85.211 bytes**
pelo descriptor adicional; não houve avanço de fidelidade/compressão.

O RMSE relativo tem denominador quase zero em um recorte constante. Por isso
essa coluna é um **diagnóstico nos 21 alvos variáveis**, selecionados somente
pela variância do alvo normalizado, igual nos dois métodos. A distorção macro,
taxas e guardas continuam usando **todos os 22 recortes**; nenhum alvo foi
excluído da codificação nem da decisão. Não comparar a macro relativa patológica
dos recortes constantes como se fosse ganho de fidelidade do novo codec.

### 12.3 Pontos de precisão maior: referência de taxa, não vitória a bytes iguais

Também foram codificados três epsilon fixos, prospectivamente definidos. O erro
limitado é absoluto nas **coordenadas normalizadas float32 protegidas**, com a
margem de arredondamento numérico medida. Não são tolerâncias físicas fornecidas
pelo usuário, nem critérios operacionais inventados. A normalização/escapes é
a mesma do VQ; as amostras escapadas são exatas e seus bytes são cobrados.

| Epsilon normalizado | RMSE nos 21 alvos variáveis | Correlação macro | Stream | Bits/célula | Razão sobre float32 |
|---|---:|---:|---:|---:|---:|
| 0,1 | 0,057758 | 0,998336 | 2.580.666 bytes | 7,097 | 4,509× |
| 0,01 | 0,005776 | 0,999983 | 3.758.979 bytes | 10,338 | 3,096× |
| 0,001 | 0,000578 | 0,99999983 | 4.682.289 bytes | 12,877 | 2,485× |

A fase FFT macro nesse mesmo subconjunto foi 0,999165/0,999992/0,99999992.
As taxas contam os **22 recortes completos**, side, framing, descriptor e mapas,
sem pesos/tabelas aprendidos. A fidelidade é maior, mas a taxa também é muito
maior que a do VQ: **não atende ao objetivo de melhorar fidelidade mantendo a
compressão**. Esses pontos são referências empíricas de alta precisão, não
limites inferiores, prova de incompressibilidade do DAS ou aceitação operacional.

Os máximos normalizados observados foram 0,100000024, 0,009999994 e 0,001000047;
os pequenos excessos sobre epsilon correspondem à margem float32 explicitamente
verificada. Não anunciar uma garantia física estrita ignorando essa margem e
o roundtrip de normalização. O preditor identidade venceu em 15–17 das 22
janelas, dependendo da precisão; isso caracteriza este coder/alvo, não prova
ausência de redundância espacial/temporal no sinal.

O VQ exige **111.832.324 bytes de tensores compartilhados**, retidos pelo híbrido;
o preditivo puro não exige pesos aprendidos. Custos de implantação/amortização
devem ser discriminados: as colunas de taxa não fingem incluir o modelo VQ nem
transformam este pequeno conjunto de recortes em uma campanha de implantação.

### 12.4 Auditoria e decisão

O controle refinado teve **966 propostas de taxa re-encodificadas**, **88 frames
FPI1 decodificados/reavaliados**, **44 decisões de rota** e **28 hashes de
snapshots/inputs** conferidos. Os erros, escapes e guardas foram verificados;
o reader CPU independente restaurou um quadro **512×1.000** sem waveform
original ou assets aprendidos. A suíte local passou em **327 testes**, incluindo
os três preditores com replay idêntico, não finitos bit-exatos, corrupção,
descompressão limitada, shapes pequenos e rejeição de side afim incompatível.

Artefatos em `results/experiments/`: `predictive_codec_train_probe_20261104/` e
`predictive_codec_refined_train_probe_20261104/`. Os resultados iniciais não
foram sobrescritos. Nenhuma confirmação de validação foi aberta, pois o critério
de avanço falhou.

**Decisão:** manter FPI1 como controle experimental de taxa/precisão, sem promoção
e sem substituir o pipeline padrão. Não ampliar treino neural ou buscar novas
tolerâncias para reinterpretar esse negativo. O resultado reforça a distinção
entre compressão agressiva com distorção alta e alta precisão com taxa maior.
Uma próxima representação precisa demonstrar mecanismo novo de redução de bits
para a informação residual, com bytes, cobertura e fidelidade concorrentes;
esses controles não justificam repetir centroides, largura ou apenas mais passos.

## 13. Fatores espaciais/temporais transmitidos a taxa limitada

### 13.1 Hipótese, codec e orçamento

Foi testada uma representação que torna a estrutura espacial **explicitamente
disponível ao receptor**: SVD da matriz protegida, com os fatores espaciais e
temporais efetivamente codificados. Isso difere de diagnosticar correlação no
residual original e supor que o decoder neural possa usá-la sem novos bits.

O formato experimental **FLR1** transmite shape, posto, ganho, normalização e
escapes, escalas float32 por componente, fatores signed8 e CRC. Os valores
singulares são incorporados aos fatores temporais. A serialização escolhe entre
bytes diretos ou diferenças modulares por linha + zlib6; a reconstrução dos
fatores é a mesma. Não há modelo/tabela aprendido ou vetor contínuo gratuito.

Nos mesmos **22 recortes de treino**, **2.909.000 células**, foram enumerados
postos 0…min(64,H,W), usando os tetos reais por janela do VQ. O controle inicial
escolhe o maior posto que cabe em bytes, sem selecionar pela qualidade. São
avaliadas duas propostas fixas: fatores crus e restauração de RMS unitário;
o ganho desta última ocupa um float64 explicitamente pago no frame. A fonte
continua fallback; nenhuma guarda foi relaxada.

A SVD é cálculo do encoder por janela, não treino de pesos compartilhados.
Toda a etapa foi CPU, sem novos updates ou sinais de validação/EDAS/SMF/test.
SVD ideal de fatores não quantizados é apenas diagnóstico; não é resultado de
compressão, ótimo com a máscara ordinária ou limite informacional do DAS.

### 13.2 Resultado inicial e busca de posto com guardas

| Política | Distorção normalizada macro | Bytes de stream híbrido | Trocas admitidas |
|---|---:|---:|---:|
| VQ de referência | 0,409006 | 84.916 | — |
| FLR1 guardado, maior posto viável | 0,408296 | 84.401 | 1/22 |
| FLR1 guardado, escolha entre todos os postos viáveis | 0,408296 | 84.401 | 1/22 |

O ganho guardado foi **0,17376% de distorção**, não de RMSE de validação; o stream
caiu 515 bytes (0,606%). Ficou abaixo do critério prospectivo de **≥1%** de ganho
guardado a stream não maior, portanto não abriu confirmação de desenvolvimento.
Não houve treinamento adicional para tentar salvar esse ponto.

A troca admitida ocorreu em **um recorte de single-mode, Configuration 8**, com
posto 8 e fatores crus. O RMSE relativo passou de **0,259217 para 0,145976**,
com **9.308 → 8.518 bytes** no frame. Correlação 0,965837 → 0,989288;
p90 por canal 0,493756 → 0,464401; razão RMS 0,962839 → 0,989655;
magnitude FFT 0,183549 → 0,118230 e fase 0,981982 → 0,996211.
É um benefício local efetivo, mas não demonstra generalização ou ganho macro
suficiente. O quadro 512×1.000 foi decodificado pelo reader independente em CPU,
sem waveform original ou pesos aprendidos.

Sem fallback, os frames do maior posto viável somaram **50.087 bytes**, contra
61.527 do VQ, mas a distorção macro piorou para **0,620768**. No diagnóstico dos
21 alvos variáveis, o RMSE macro foi **0,769111**, contra **0,665741** da fonte.
Esse subconjunto só evita o denominador relativo patológico do tile constante:
os 22 recortes continuam incluídos em bytes, guardas e distorção principal.
Não promover a economia bruta sacrificando fidelidade.

Um recorte multimode já melhorava erro global/espectro com o maior posto, mas
regredia o p90 por canal. Isso motivou um **seletor aninhado** entre os postos
viáveis, preservando a proposta inicial, a precisão dos fatores, as duas opções
de ganho e todas as guardas. O encoder pode escolher a menor distorção admissível
porque o posto/ganho são transmitidos; não foi ajuste de modelo na validação.
Mesmo após essa busca, nenhuma troca adicional foi admitida. O negativo não
decorre apenas de usar o maior posto em vez do melhor posto admissível.

### 13.3 Diagnóstico da codificação dos fatores temporais

Antes de implementar outra variante, foi verificada a energia retida por **16
coeficientes DCT por fator temporal**, em uma decomposição diagnóstica de posto
até 8. A regra pré-fixada permitia um único codec espectral de fatores somente
se pelo menos duas configurações de treino retivessem ≥80% dessa energia.

As médias por configuração ficaram em **14,96–69,52%**: nenhuma atingiu o corte.
Assim, **a variante espectral não foi implementada ou treinada**. Essa energia
contínua não seria um resultado codificado; índices, escalas, quantização e bytes
teriam de ser cobrados em um teste posterior. O diagnóstico não prova ruído
independente nem impossibilidade de outras representações de fatores.

### 13.4 Auditoria, custos e decisão

Foram re-encodificadas **1.087 propostas de posto por variante**. A auditoria do
controle inicial verificou 60 frames, 38 propostas de restauração e 44 rotas;
a busca aninhada verificou 142 frames, 120 propostas e 44 rotas. Cada variante
teve 28 hashes de snapshots/inputs conferidos. Essas contagens incluem replays
e janelas relacionadas, **não amostras independentes**.

O híbrido retém os **111.832.324 bytes de tensores VQ compartilhados**, separados
das taxas de stream. Os fatores FLR1 não têm custo de pesos aprendidos, mas
precisam ser pagos integralmente em cada frame. Nenhum modelo ou código de
baixo posto é gratuito no cálculo de implantação.

Artefatos em `results/experiments/`: `lowrank_factors_train_probe_20261104/`, seu
`temporal_factor_diagnostic.json` e `lowrank_guarded_rank_search_train_20261104/`.
A suíte completa passou em **331 testes**, incluindo fatores quantizados,
escapes NaN/extremos bit-exatos, ganho transmitido, posto zero, corrupção,
descompressão limitada e rejeição de side afim incompatível.

**Decisão:** preservar FLR1 como controle experimental e seu ganho local, sem
promoção ou extensão automática. A representação não manteve a fidelidade na
maior parte dos recortes; seleção de postos e ganho não resolveram isso. Não
abrir validação ou sacrificar canais fracos para transformar o ganho local em
alegação generalista. A próxima hipótese precisa explicar como codifica melhor
a informação restante, e não apenas aumentar posto, bits ou orçamento de busca.
