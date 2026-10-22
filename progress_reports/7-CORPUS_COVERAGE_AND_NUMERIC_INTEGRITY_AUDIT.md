# Auditoria FEBUS: cobertura, redundância e integridade numérica

> Refinamento da análise espacial: o [relatório 12](12-SPATIAL_CHANNEL_MAPS_AND_GEOMETRY_INTEGRITY.md)
> recuperou os índices dos recortes preparados. As 138 lacunas internas contadas
> abaixo pertencem ao sinal bruto, mas ficam fora do crop central selecionado:
> os 1.194 blocos preparados são espacialmente contíguos. A necessidade de
> rastreabilidade permanece; não atribuir as falhas do modelo a gaps nesses recortes.

Data: 2026-10-07. Auditoria sem treino/download, HDF5 abertos somente para leitura.
Artefatos: `results/experiments/corpus_audit_20261007/` e
`results/experiments/corpus_extremes_20261007/`.

## Decisão: qualidade numérica antes do próximo treino

**Há um problema concreto de representação numérica em parte do corpus:** valores
finitos extremos em 40 janelas multimode dominam a normalização global e apagam a
variação de quase todos os canais não extremos ao converter para float32.
Isso deve ser tratado/instrumentado antes de novos treinos comparativos. Não é
necessário atribuir a esses valores origem física inválida para demonstrar a perda
numérica; sua origem ainda exige investigação do export/interrogador.

Não remover, mascarar ou fazer clipping silencioso dos HDF5. Uma solução futura
deve preservar o alvo original, por exemplo com um mecanismo reversível de escape
dos valores extremos e métricas separadas por região, ou outro tratamento validado.
Não apresentar RMSE global artificialmente favorável como melhora no sinal normal.

Outros achados: pouco tempo físico/diversidade por família apesar de 12 GB brutos;
~95% de cobertura temporal já aproveitada; cobertura espacial parcial em algumas
configurações; quatro cópias dos mesmos recortes por fold consumindo ~27,6 GB extras.
Mais janelas sobrepostas ou mais cópias não equivalem a mais dados independentes.

## 1. Escopo e método

- Inventário local: 207 HDF5 brutos, 23 grupos família/configuração, 12.049.766.184 bytes
  (~12,05 GB decimal). Não foi atualizado o inventário remoto nesta etapa.
- Leitura em chunks de todos os datasets de sinal escolhidos pela preparação:
  checksum completo do payload, finitude por canal/bloco e alertas de magnitude.
  Sem carregar cada HDF5 inteiro em RAM nem reescrever arquivos.
- Manifestos dos quatro folds deduplicados por coordenadas: ID, bloco, início,
  tamanho temporal e canais. Compatibilidade de geometria/origem verificada.
- Conteúdo espectral/heterogeneidade: duas janelas por arquivo **somente do treino
  do fold 1**, seed 2026: 210 janelas de 105 arquivos, 11 configurações. Descritores
  não usam resultados de teste nem escolhem hiperparâmetros de treino.
- Investigação de qualidade dos três arquivos com alertas: todas as suas 280 janelas
  preparadas, usando uma cópia canônica de cada coordenada. Isso inclui os diferentes
  papéis de CV, mas é inspeção numérica, não avaliação/seleção de modelos.

## 2. Volume efetivo, famílias e tempo

| Família | Arquivos | Grupos/configurações | GB brutos decimais | Janelas únicas | Tempo armazenado em blocos (s) | União de clock por grupo (s) |
|---|---:|---:|---:|---:|---:|---:|
| Doped | 27 | 3 | 1,04 | 3.940 | 206,85 | 198,35 |
| Multicore | 90 | 5 | 4,17 | 13.060 | 685,65 | 349,25 |
| Multimode | 18 | 3 | 4,10 | 1.680 | 88,20 | 84,90 |
| Single-mode | 72 | 12 | 2,74 | 5.200 | 273,00 | 263,60 |
| **Total** | **207** | **23** | **12,05** | **23.880** | **1.253,70** | **896,10** |

Há apenas ~14,94 minutos na soma das uniões de clock de cada grupo. **Não é uma
estimativa exata de aquisições independentes**: núcleos simultâneos e blocos com
overlap não devem ser contados como observações temporais novas. Mais canais não
significam mais tempo. Não comparar diretamente GB FEBUS com GB OOI sem contabilizar
sample rate, canais, diversidade e seleção posterior ao processamento.

As uniões usam `Source/time` como epoch em segundos e `Origin` temporal em ms,
coerente com Spacing. Todos os 207 arquivos tinham timestamps utilizáveis. Trata-se
de reconstrução dos intervalos a partir de metadados, não de confirmação externa de
continuidade física. Os blocos têm 21.000 amostras a 20 kHz (~1,05 s); timestamps
avançam ~1 s, portanto há overlap. Duas amostras desses blocos podem diferir devido
ao processamento de bordas: overlap temporal não prova igualdade exata do sinal.

### Cobertura temporal

20 janelas de 1.000 amostras por bloco cobrem união de 20.000/21.000 amostras:
**95,24% do tempo armazenado nos blocos**, igual nas famílias. O tempo somado das
janelas é 1.194 s; removendo overlap de clock entre arquivos/blocos/núcleos por grupo,
a união preparada é ~851,70 s, versus 896,10 s brutos (~95,05%).

Dobrar `windows-per-block` produziria sobretudo recortes sobrepostos, não dobraria
a duração disponível. Há aproximadamente 5% de cobertura temporal adicional possível,
além de mudar duração/contexto da janela. As janelas atuais duram apenas 0,05 s e
resolvem FFT a 20 Hz: não representam ciclos completos de excitações nominais de
1 ou 10 Hz. Janelas de duração física maior precisam de estratégia de memória/tiling;
não reduzir sample rate descartando bandas do alvo apenas para imitar OOI.

### Cobertura espacial e máscara de canais

- Seis arquivos multimode (Configuration 5) têm ~1.012 canais brutos, ~1.006–1.012
  finitos, mas a preparação mantém 512 centrais. Isso retém ~51% dos canais finitos
  nesses blocos, ~48% do espaço×tempo finito ao incluir cobertura temporal.
- Doze arquivos single-mode excedem 512 canais finitos; corte menor, com cobertura
  espaço×tempo mínima ~89,47%. Demais arquivos não são cortados por esse limite.
- 5.355 instâncias canal/bloco foram removidas por não finitude em algum ponto;
  138 são lacunas internas, todas em seis arquivos single-mode Configuration 8.
  Remover linhas internas muda a vizinhança física. Metadados atuais registram
  contagem/dx, mas não o mapa completo dos índices mantidos: corrigir rastreabilidade
  antes de interpretar geometria como uniformemente espaçada após filtragem.
- `dx` varia de 0,2 a 5 m; GaugeLength registrado de 1 a 20 m. Mesmo tamanho em pixels
  não equivale a mesmo contexto espacial físico. MoS somente por nome da fibra pode
  não acomodar todas essas diferenças.

Tiling espacial pode ampliar conteúdo usando os mesmos HDF5, preservando splits por
configuração e NaNs/mapeamento físico; não cria novas aquisições independentes.

## 3. Redundância e armazenamento

23.880 coordenadas únicas têm **quatro cópias** nos folds: 95.520 NPZ, 36,77 GB decimais.
Uma cópia por coordenada ocupa aproximadamente 9,19 GB; excesso estimado ~27,58 GB
(diferenças de metadados/split também influenciam a compressão do NPZ).

Igualdade numérica confirmada por hash dos arrays em uma janela por arquivo:
207 coordenadas, 828 cópias, nenhuma divergência. Não foi feito hash de todos os
95.520 arrays. Os quatro manifestos concordam nas coordenadas/geometrias.

Não houve payload de sinal bruto exatamente duplicado entre os 207 HDF5 pelo hash
completo. Isso não implica independência: proximidade temporal, núcleos simultâneos
e conteúdo correlacionado permanecem. Copies por fold também não causam vazamento
por si só: seus papéis são definidos separadamente por configuração.

**Recomendação de infraestrutura:** store canônico das amostras e splits em manifestos/
referências, evitando recriar bytes por fold. Não migrar/deletar automaticamente:
planejar compatibilidade com loaders, metadata_json e mapas de canal, validar equivalência
e obter autorização antes de remover arquivos locais antigos. Nenhum arquivo foi apagado.

## 4. Conteúdo do treino: descritores exploratórios

Medianas por janela, duas janelas por arquivo do treino do fold 1. Não são estimativas
globais de todas as configurações; doped/multimode aqui têm somente uma configuração.
Espectros calculados após retirar média temporal por canal e aplicar Hann.

| Família | Janelas / arquivos / grupos | Energia em top 10% canais | Fração efetiva de canais | Entropia espectral normalizada | Flatness espectral | Energia temporal 5–10 kHz |
|---|---:|---:|---:|---:|---:|---:|
| Doped | 18 / 9 / 1 | 0,735 | 0,124 | 0,971 | 0,840 | 0,439 |
| Multicore | 108 / 54 / 3 | 0,300 | 0,471 | 0,978 | 0,909 | 0,456 |
| Multimode | 12 / 6 / 1 | 0,254 | 0,711 | 0,964 | 0,883 | 0,640 |
| Single-mode | 72 / 36 / 6 | 0,665 | 0,180 | 0,694 | 0,465 | 0,410 |

A fração efetiva é `1/sum(p_canal²)/C`, indicador de concentração, não contagem
de canais válidos. Em doped, seis linhas de borda concentram mediana de ~74,4% da
energia. Em multimode central, energia é mais distribuída e fortemente de banda alta.
Média global/desvio global podem privilegiar canais dominantes e suprimir escalas
locais. Testar normalização por canal exige guardar seus parâmetros e contabilizar
overhead; não descartar canais/ruído para reduzir artificialmente a dificuldade.

Entropia/flatness do espectro médio **não medem entropia de codificação**, não provam
ruído independente e não sustentam um limite Gaussian rate-distortion. Correlação
entre linhas adjacentes preparadas é significativa (mediana ~0,71 em multimode).
O alvo pode ter estrutura espacial mesmo com espectro temporal largo.

## 5. Valores finitos extremos: achado prioritário

88.770 células brutas com `abs(valor)>1e20`, somente em três arquivos multimode.
Nenhuma está próxima do limite máximo float32: não há evidência de que sejam apenas
o sentinel `finfo.max`. O threshold é um alerta conservador, **não regra de validade física**.

| Configuração / arquivo | Células brutas extremas | Janelas preparadas afetadas / verificadas | Máximo nas afetadas |
|---|---:|---:|---:|
| 5 / _2025-12-08_09-15-58_UTC.h5 | 27.989 | 0 / 100 | Não entram nos recortes atuais |
| 6 / _2025-12-08_09-21-52_UTC.h5 | 46.448 | 20 / 80 | 1,343×10^36 |
| 6 / _2025-12-08_09-22-11_UTC.h5 | 14.333 | 20 / 100 | 7,610×10^24 |

Nos arquivos Configuration 6, o bloco afetado é índice 1. Extremos aparecem nas
últimas quatro linhas preparadas no primeiro arquivo e última linha no segundo.
São finitos, por isso passam pelo filtro `np.isfinite` da preparação.

Reprodução exata da normalização usada no treino:
`((data_float64 - mean_global)/(std_global + 1e-6)).astype(float32)`.

- Primeiro arquivo: 10.160/10.160 instâncias de canais originalmente variáveis e
  sem células acima do threshold viraram constantes dentro da janela.
- Segundo: 10.180/10.220 dessas instâncias perderam toda a variação temporal.
- Contagem é canal×janela, não canais físicos únicos. É uma perda numérica, não
  limitação do decoder: o modelo recebe sinal já colapsado nesses canais.
- 40/23.880 janelas (~0,17% global), 40/1.680 multimode (~2,38%), 40/600 da
  Configuration 6 (~6,67%). Isso **não explica todo o problema**. O probe dos
  relatórios 6/7 usou treino fold 1, sem essa configuração, e permanece válido
  como evidência de ajuste/overfit no seu protocolo.

### Como tratar sem falsear a fidelidade

1. Inspecionar aquisição/export e semântica dos valores com documentação FEBUS;
   manter uma classificação de qualidade rastreável, não assumir invalidade só pelo tamanho.
2. Implementar proteção reversível de dynamic range, por exemplo escape dos valores
   extremos em side stream exato e normalização sobre região restante. Restaurá-los
   após decodificação, sem destruir o original. Não usar clipping oculto como solução.
3. Medir separadamente fidelidade da região modelada, preservação bit-exata do escape,
   taxa/bytes de side stream e cobertura. RMSE global após restaurar extremos pode
   ficar artificialmente pequeno porque eles dominam o denominador: isso não prova
   reconstrução boa dos canais de interesse.
4. Reavaliar somente após manter o mesmo tratamento nos splits, usando validação
   para decidir. Os testes já foram consultados em etapas anteriores; auditoria externa
   continua necessária. Se houver exclusão científica por invalidade confirmada,
   documentar explicitamente a mudança de alvo e suas métricas, não comparar cegamente
   aos relatórios históricos.

## 6. Sequência revisada

1. **Proteção numérica e avaliação por regiões sobem para a primeira posição.**
   Não treinar variantes arquiteturais tentando aprender variação que desapareceu
   antes do forward. Preservar valores/arquivos originais e contabilizar side stream.
2. Store canônico/rastreabilidade e planejamento de espaço antes de expandir corpus.
   Mapa de índices espaciais após máscara deve ser incluído nas próximas preparações.
3. Normalização e geometria: auditar/testar escala por canal, tiling espacial e
   duração física, com métricas comparáveis e overhead explícito. Há pouca margem
   temporal não usada, mas alguma margem espacial/diversidade de recortes.
4. Adaptação conservadora/fine-tuning parcial e MoS em ablações por configuração
   continuam alternativas; não promover memorizadores nem inferir generalização
   de média favorável somente em uma fibra.
5. **Mais aquisições FEBUS-like faz sentido como etapa condicional:** três grupos
   doped e três multimode, pouco tempo físico, limitam diversidade. A auditoria torna
   a hipótese de escassez de cobertura mais plausível, mas não demonstra que ela
   seja a única causa. Melhorar código/qualidade/representação precede grandes downloads.
6. Arquitetura/taxa e treino do zero permanecem possíveis se dados modelados e
   otimização estiverem adequados. Modelo final continua adiado.

## 7. Reprodução e validações

```bash
.venv/bin/python finetuning/audit_febus_corpus.py \
  --output-dir results/experiments/corpus_audit_new
.venv/bin/python finetuning/investigate_corpus_extremes.py \
  --audit-dir results/experiments/corpus_audit_new \
  --output-dir results/experiments/corpus_extremes_new
```

Saídas devem ser novas. O primeiro script lê o payload bruto completo; pode demorar
alguns minutos, mas não copia dados nem usa GPU para treino. Artefatos: `files.csv`,
`families.csv`, `train_content.csv`, resumos e investigação por janela.

**48 testes passaram**, incluindo union de intervalos, layout, hash independente
do chunk, detecção de canais não finitos, concentração de energia, espectro e
reprodução da perda de variação por extremos finitos. Auditoria terminada;
**sem treino, download, processo em background ou alteração de dados/checkpoints**.
