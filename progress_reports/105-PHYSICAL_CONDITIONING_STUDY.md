# H3 — metadados versus identificação de configuração

**Protocolo de 15/11/2026, registrado antes do novo treinamento.**
Protocolos, resultados e decisões desta etapa permanecem neste documento.

## Questão identificável e revisão do adiamento

O relatório 98 adiou FiLM porque hardware é constante por configuração e sessões
não têm independência certificada. Isso permanece verdadeiro. O novo desenho
avalia uma questão mais limitada: condicionamento por atributos numéricos
transfere melhor para **combinações excluídas do ajuste** que um rótulo
categórico ou vetor constante? Não chamar o resultado de causalidade física,
generalização entre sessões ou confirmação independente.

O censo já disponível contém nove vetores estendidos de hardware distintos.
Os dois grupos multicore compartilham o mesmo vetor: excluí-los um a um não
criaria uma combinação nova. Reservar os grupos single-mode lexicograficamente
primeiro e último com vetor único: **Configuration 11 e Configuration 8**.
Os oito grupos restantes preservam as quatro famílias no treino. O preflight
deve verificar essas condições antes de abrir sinais para o novo ajuste.
Estas combinações foram consumidas em estudos anteriores: a exclusão vale
para este contraste, não cria um holdout científico virgem.

## Entradas, braços e arquitetura

- Fonte OOI original anterior à adaptação FEBUS, SHA-256
  `7d2da0d1cbd1d3dae579d7fd900850afc9ef15f645ea844168c0f78f18b0a27c`.
  Não usar a fonte madura que já aprendeu as combinações excluídas.
- Mesma CNN/VQ densa K2048×128 em todos os braços. Todos os pesos treináveis.
- Substituir cada GroupNorm por GroupNorm com FiLM residual limitado:
  `GN(x) × (1 + 0,1 tanh(gamma)) + 0,1 tanh(beta)`.
  Gerador `11→16→2C`, ReLU, última camada zero, mantendo exatamente a função
  inicial. Mesma capacidade e inicialização nos três braços.
- **constant:** dez zeros e flag zero; adapta apenas uma função comum.
- **hardware:** dx, dt, GaugeLength, PulseWidth, PulseRateFreq, SamplingRate,
  SamplingRes, Oversampling, AmpliPower e DerivationTime. Transformação log2
  e padronização ajustadas apenas nos oito grupos do treino. Campos de pulso
  mantêm semântica/unidades nativas; não inferir unidades pela magnitude.
- **category:** one-hot dos oito grupos vistos; grupo desconhecido recebe
  vetor zero. Não criar embedding treinado usando o nome de um grupo reservado.
- Para hardware/category, substituir a condição por dez zeros + flag ausente
  a cada décimo update **de cada família**, evitando correlacionar ausência com
  a ordem cíclica das famílias. Constant ignora ausência. Essa regra é comum, fixa e
  não altera a sequência dos recortes.

O receptor recebe a condição realmente usada. Frames têm wrapper com CRC:
constant não transmite vetor derivável; category paga ID conhecido/desconhecido/
ausente; hardware paga dez float32 padronizados + flag. Cobrar wrapper, base,
registros, mapas e side; pesos/shared config discriminados.
Nada é inferido de estatísticas do alvo no receptor.

## Dados, budget e integridade

- Os 43 arquivos check históricos nunca entram nos updates ou calibração.
  Excluir também **todos** os arquivos dos dois grupos reservados do ajuste.
- Calibrar probabilidades apenas nos recortes fit dos oito grupos vistos.
  Reutilizar escalas de métricas prospectivas, sem fornecê-las à rede.
- Mesma sequência de família → grupo → arquivo → recorte nos três braços;
  crops contíguos de oito canais × 1.000 amostras. Seed 2117.
- **16.384 updates por braço**, 131.072.000 apresentações de células cada,
  AdamW fresco LR=1e-4, wd=1e-5, clip1, FP32, seed 2117.
- Smoke de dois updates por braço; verificar função inicial, índices, frames,
  gradientes de FiLM e ausência de arquivos/grupos reservados antes da fase
  científica. Smokes não entram nos totais e não escolhem parâmetros.
- GPUs0/1: constant/hardware em paralelo; category em seguida. Nenhuma extensão
  automática, seleção de checkpoint ou varredura de campos/GL/seed.
- Avaliar fonte e endpoint nas mesmas 136 janelas train-only; publicar
  separadamente os oito grupos vistos e os dois excluídos. Receptor separado
  deve reproduzir os bits do caminho direto condicionado.
- No endpoint, repetir com metadados ausentes, usando os mesmos pesos e
  probabilidades já calibradas. Ausência não é nova oportunidade de seleção.
- Sem sinais val/test/EDAS nem alteração do particionamento original.

## Decisão prospectiva

Investimento exige no conjunto dos dois grupos excluídos: hardware ≥5% melhor
em NRMSE* que **ambos** os controles, bytes completos não maiores que ambos e
nenhuma macro desses grupos pior; também nenhuma regressão de macro nos oito
grupos vistos versus constant. Publicar guardas espectrais, amplitude, p90 e
cobertura; média favorável com regressão física não promove candidato.
Para o custo de cada subconjunto visto/excluído, somar seus frames e cobrar o
setup completo uma vez, igualmente nos três braços; não amortizar setup apenas
no braço favorecido. Não comparar os dois subconjuntos como a mesma população.
Falha de integridade interrompe; não relaxar gate nem escolher outra combinação.
Este é gate de oportunidade, não P3. Comparar ainda à referência prática madura,
declarando seu prévio acesso FEBUS, sem usá-la como controle causal.

Se hardware só igualar category, o resultado não demonstrará benefício físico
além de identificar configuração. Se falhar, encerrar este desenho; não afirmar
que metadados ou novas sessões são inúteis. A questão de independência de sessão
continua exigindo documentação/aquisição nova.

Destino: `results/experiments/physical_conditioning_20261115/`.

## Preflight e engenharia

O censo confirmou os dois vetores estendidos únicos e a preservação das quatro
famílias nos oito grupos de ajuste. A escolha não usou os waveforms.
**492 testes locais passaram**, incluindo função inicial bit-exata, gradientes
de FiLM, exclusão dos grupos na normalização, custos/CRC de contexto e receptor
independente sem acesso ao estado do encoder. A execução remota só recebe a
fase científica após smoke completo dos três braços, conferindo a igualdade
dos índices iniciais e da sequência de treino. Sem resultados novos ainda.

O supervisor foi lançado no remoto. Budget científico total: **49.152 updates /
393.216.000 apresentações de células**; inicialização e smokes discriminados.
A análise de resultados foi implementada sem consultar saídas desse treino.
Os testes do gate conferem ambos os controles, ausência de regressão nos grupos
vistos, cobrança integral de setup por subconjunto e não arredondamento do
limiar. A suíte ampliada passou **495 testes**; 288 links locais válidos.

Após a conclusão dos três braços:

```bash
.venv/bin/python -m finetuning.summarize_physical_conditioning \
  --study results/experiments/physical_conditioning_20261115 \
  --output results/experiments/physical_conditioning_20261115/ANALYSIS
```

A análise não treina, não abre waveforms e não promove automaticamente.

## Resultados concluídos — 16/11/2026

O reinício cancelou apenas o acompanhamento local. Os três processos científicos
terminaram com código zero; **nenhum treino foi repetido**. Cada braço completou
16.384 updates / 131.072.000 apresentações de células, 4.096 updates por família
e 1.636 updates com condição ausente. Sequência, crops, fonte e capacidade
nominal foram conferidos; as 28 vias FiLM tiveram a função inicial preservada.

### Todas as configurações — mesmo conjunto de desenvolvimento

136 janelas / 43 arquivos / dez grupos / quatro famílias / 12.799.000 células:

| Condição | NRMSE* macro famílias ↓ | Stream bpc ↓ |
|---|---:|---:|
| Constante | 0,999957 | 0,595118 |
| Hardware | 1,002881 | 0,614276 |
| Categoria | **0,985253** | 0,602642 |

Nenhum ponto se aproxima da referência prática madura ~0,490336 a ~0,679244 bpc
nessa coorte. A referência tem outro histórico de adaptação e não é controle
causal deste estudo nem taxa pareada para alegar dominância geral.

### Combinações excluídas do ajuste

21 janelas / seis arquivos / duas configurações single-mode / 4.658.000 células.
Setup completo cobrado uma vez em cada subconjunto, como registrado:

| Condição | NRMSE* médio das duas configurações ↓ | Bytes completos ↓ |
|---|---:|---:|
| Constante | 1,234624 | 309.708 |
| Hardware | 1,225943 | 320.271 |
| Categoria, ID desconhecido | **1,152121** | 312.047 |

Hardware reduziu erro em somente **0,70314% contra constante**, com
**3,41063% mais bytes**. Contra categoria, teve **6,40743% mais erro /
2,63550% mais bytes**. As quatro condições do gate de oportunidade falharam.

| Grupo excluído | Constante | Hardware | Categoria desconhecida |
|---|---:|---:|---:|
| Configuration 11 | 1,768451 | 1,783748 | 1,640796 |
| Configuration 8 | 0,700797 | 0,668137 | 0,663447 |

Hardware piorou Configuration11. Contra constante, teve NRMSE* e p90 maiores
em dois dos seis arquivos e erro espectral maior em 3/6 pares grupo×banda.
Contra categoria: cinco dos seis arquivos com NRMSE*/p90 maiores e 5/6 pares
de banda com erro maior. Nos grupos vistos, houve nove arquivos com NRMSE*
maior, 12 com p90 maior e 11/24 regressões de erro espectral versus constante.

O controle categórico **não recebeu rótulos distintos dos dois grupos
excluídos**: ambos usam vetor zero/ID desconhecido pago. Seu melhor resultado
não demonstra memorização desses grupos; os pesos do backbone foram adaptados
de modo diferente nos oito grupos vistos. Tampouco demonstra utilidade física
dos atributos numéricos além da categorização.

### Metadados ausentes

Mesmos pesos/probabilidades, sem recalibração:

| Condição | NRMSE* global sem metadados | NRMSE* dos dois grupos excluídos |
|---|---:|---:|
| Constante | 0,999957 | 1,234624 |
| Hardware | 1,003304 | 1,235653 |
| Categoria | 0,987688 | 1,156649 |

Constante reproduziu exatamente seus frames normais, como exigido.
Hardware sem metadados teve NRMSE* maior em 29/43 arquivos, p90 maior em 33 e
erro espectral maior em 10/30 pares que hardware com metadados.
Ausência não resgatou o candidato nem foi usada para seleção.
Fase continua descritiva, com suportes publicados, não certificação física.

## Decisão e integridade

**Encerrar este desenho sem extensão, troca de grupos/campos/seed ou promoção.**
O contraste limitado de H3 agora foi executado; não foi demonstrada vantagem
de fidelidade por bit do vetor de hardware. Isso não refuta toda forma de FiLM,
metadados calibrados, novos regimes ou generalização entre sessões.
Proveniência independente permanece pendente; não abrir reservas para procurar
um resultado favorável.

Recuperados **1.838 arquivos com SHA-256 conferido**; 12 checkpoints/receptores
preservados remotamente com hashes. Inventário remoto: 2.749.969.419 bytes.
Auditados **1.656 frames / 10.476.000 posições de índices / 24 literais**,
com CRCs, custos, exclusões e snapshots conferidos sem ler waveforms.
O vetor pago coincide bit a bit com o plano congelado e recebido.

Recalcular localmente o desvio logarítmico de um atributo nativo diferiu do
registro remoto em 1,665×10⁻¹⁶ em float64. A auditoria usa os parâmetros
**congelados e transmitidos**, sem recalibrá-los; não houve alteração de
contextos, métricas, gates ou resultados por essa diferença de recomputação.

Evidências: `RECOVERY_AUDIT.json`, `POSTFLIGHT_AUDIT.json` e `ANALYSIS/`,
incluindo comparações físicas dos conjuntos completo, vistos e excluídos.
O reinício não consumiu updates extras. Teste fold0 e EDAS intactos.
A continuação autorizada trata outra pergunta de H5 no mesmo documento
[99](99-CLOCK_ALIGNED_MULTICORE_OPPORTUNITY.md): magnitude condicional e
codificação lossless dos coeficientes, não extensão deste FiLM reprovado.
