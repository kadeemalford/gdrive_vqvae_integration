# PoroTomo: transferência útil com dose alvo igual, mas inferior ao investimento direto em FEBUS

**Data: 10/11/2026.** Campanha `porotomo_bridge_fold0_20261110` concluída,
com referência F reutilizada após auditoria e sem promoção. Continuidade dos
[relatórios 80](80-LEARNING_RATE_AND_EXTERNAL_PRETRAINING_REVIEW.md), seção 10,
e [82](82-QUANTX_PRETRAINING_BRIDGE_RESULTS.md).

## 1. Duas conclusões diferentes, ambas importantes

1. **A fonte pública ajudou a inicialização para uma dose FEBUS final curta:**
   contra offshore → 8.192 updates FEBUS, o pré-treino PoroTomo seguido dos
   mesmos 8.192 updates reduziu RMSE em **4,537%**, nas seis configurações.
   Porém, consumiu mais treino e **4,093% mais bytes**; não é avanço gratuito
   de fidelidade à mesma taxa ou ao mesmo compute.
2. **Não foi a melhor aplicação do orçamento total:** frente aos 24.576 updates
   diretos em FEBUS, a ponte produziu **5,494% mais RMSE e 1,728% mais bytes**.
   F domina este ponto P em taxa–distorção sob o contrato medido. A meta de
   investimento não foi alcançada; não ampliar o ramo automaticamente.

Portanto, o resultado não é “outro dataset nunca ajuda”. A fonte correta pode
facilitar adaptação, mas isso não demonstrou um codec superior ao controle
disponível nem à referência FEBUS mais longa já preservada.

## 2. Fonte, integridade e controle

- **60 HDF5 / 1.798 janelas**, quatro estratos de diretório em março de 2016,
  poço PoroTomo, Silixa iDAS. Não são quatro famílias de fibra independentes.
- Traços nativos declarados proporcionais a **strain-rate**; uma conversão
  HDF5/SEG-Y foi verificada integralmente bit a bit, com SHA3 nativo coincidente.
  Não foi feita conversão de calibração absoluta para a unidade FEBUS.
- Dados fonte a **1 kHz**, 384 canais e 1 s por janela; alvo FEBUS a 20 kHz,
  50 ms e geometria retida nativa. Não houve reamostragem, filtro, integração,
  denoising, remoção de canais ou concatenação de arquivos.
- A preparação versão 2 tratou contêineres HDF5 sem waveform como inelegíveis
  estruturalmente antes da seleção por quantis. Nenhuma seleção por desempenho
  neural, amplitude ou SNR. Os dados e a seleção anterior interrompida foram
  preservados; nenhum modelo usou uma preparação incompleta.
- Mesmos pesos offshore **originais**, CNN nativa/grade compacta/K2048, loss
  `0,5 MSE + 0,5 L1 + VQ`, LR `1e-4`, AdamW novo no início e na fronteira
  de etapa. Os mesmos 8.192 batches FEBUS finais e seeds foram usados.
- **F:** 16.384 + 8.192 updates FEBUS. **P:** 16.384 PoroTomo + 8.192 FEBUS.
  **G:** somente os 8.192 FEBUS finais, sem nova etapa de pré-treino.
- F não foi retreinado: foram verificados inicialização, sequências, hashes de
  dados/código/vendors, momentos/contadores retidos, pesos e artefatos de stream.
  A avaliação de F/OOI inicial também foi reutilizada apenas após verificar
  inputs, calibração, implementação de replay, grupos, modelos e bundles exatos.

É uma comparação de **updates**, não de FLOPs ou de células observadas por
update. PoroTomo usa imagens fonte maiores; ~2.469 s de updates externos +365 s
FEBUS, contra ~730+365 s em F. Os ~47,2 min de updates de P são ~2,59× o tempo
de updates de F nesta execução, além de validação/I/O. Usou-se uma única RTX
4090; nenhum treino concorrente desta pesquisa ocupou outra GPU.

## 3. Resultado completo em 7.220 janelas

| Pipeline | Novos updates externos / FEBUS | RMSE relativo macro ↓ | Correlação ↑ | Bytes completos | Razão completa |
|---|---:|---:|---:|---:|---:|
| G — offshore → FEBUS curto | 0 / 8.192 | 0,802886 | 0,577210 | **14.810.782** | ~171,71× |
| P — offshore → PoroTomo → FEBUS | 16.384 / 8.192 | 0,766458 | 0,625097 | 15.416.987 | 164,966× |
| **F — offshore → FEBUS direto** | 0 / 24.576 | **0,726543** | **0,669522** | 15.155.168 | 167,816× |

Codec comum FAK1/FSB1, probabilidades ajustadas nas mesmas 32 coordenadas de
treino, replay discreto antes dos scores e reader independente. Mapas,
manifesto, tabelas, registro e framing estão cobrados no transporte. Os modelos
de implantação têm **111.869.967 bytes cada**, discriminados como compartilhados;
incluindo o modelo uma vez nesta coorte curta, P fica em ~**19,981×**.

Esta campanha parte do offshore sem adaptação, não do melhor checkpoint FEBUS
avançado. Os scores ~0,73–0,80 não são uma regressão silenciosa da referência
A já treinada por 98.304 updates, cujo RMSE nesta coorte era **0,667586**.
Não combinar esses orçamentos como se fossem a trajetória de um único modelo.

## 4. Distribuição da melhora condicional

| Família / configuração | G: RMSE, 8.192 FEBUS | P: RMSE, mesma dose FEBUS | F: RMSE, 24.576 FEBUS |
|---|---:|---:|---:|
| Doped / 5m LSF | 0,787650 | 0,735207 | **0,704751** |
| Multicore / C1 and C3 | 0,859605 | 0,829375 | **0,814367** |
| Multimode / Configuration 6 | 0,896801 | 0,867506 | **0,822881** |
| Single-mode / Configuration 10 | 0,682889 | 0,653422 | **0,552099** |
| Single-mode / Configuration 2 | 0,711510 | 0,679435 | **0,645029** |
| Single-mode / Configuration 6 | 0,608066 | 0,568376 | **0,495396** |

P melhora G em ~3,27–6,66% de RMSE por configuração. Também melhora o RMS ratio
macro de G, `0,605806 → 0,641888`, mas isso não demonstra preservação física
completa e compra bytes adicionais. Contra F, RMSE/correlação pioram nas seis
configurações, com **33 vetos** por métrica/configuração. Magnitude FFT macro
de P: **0,620324**, fase **0,776497**, RMS ratio **0,641888**, todos piores que F.
As guardas não foram relaxadas e nenhum canal foi retirado após avaliar resultados.

Na triagem, P tinha RMSE **0,922936** ao terminar a fase pública e
`0,826401 / 0,805169 / 0,781928 / 0,768600` nos quatro blocos FEBUS finais.
São scores da triagem histórica, não da avaliação completa da tabela.
Selecionado e endpoint coincidiram. O AdamW terminou a etapa FEBUS em **8.192**,
coerente com o reset comum. As curvas ainda melhoravam ao teto: não afirmar
ótimo de convergência ou que fine-tuning mais longo seria inútil.

## 5. Interpretação e decisão

A verificação do produto físico e o benefício condicional de P são mais
informativos que escolher dados apenas por fabricante ou sample rate. Porém,
o contraste não isolou qual atributo — readout, geometria, conteúdo ou quantidade
de células — causou a diferença contra outras fontes. Não prova superioridade
intrínseca de Silixa sobre QuantX ou generalização externa das quatro famílias.

O pré-treino público poderia interessar se houvesse restrição de exposição FEBUS
ou muitas adaptações posteriores; essa amortização é **hipotética**, não economia
medida nem justificativa de promoção. Aqui o objetivo continua fidelidade por
bit no modelo geral; F ofereceu menor erro com menos bytes e menos tempo.

**Fechar este contraste sem aumento automático de pré-treino ou fine-tuning.**
Preservar os checkpoints públicos como evidência de transferência condicional,
não substituir as referências compacta/densa já existentes. Uma investigação
posterior de curvas longas precisaria de protocolo pareado próprio, sem apagar
o negativo do orçamento atual.

## 6. Continuidade distinta: síntese dos mesmos símbolos

Foi lançado um probe **somente em arquivos de treino** para isolar um mecanismo
diferente: um dicionário de síntese separado, mantendo encoder, dicionário de
análise, índices originais e decoder congelados. Otimizar síntese por erro latente
serve de controle à otimização por waveform. Não se alteram K, grade, PCA ou a
regra de atribuição dos símbolos.

Isso difere do PQ decoder-aware anterior: aquele usou 2×32 centroides e
recalculou atribuições nearest durante os updates. Agora se usam os **2.048
endereços originais fixos** de uma referência transferida madura, para testar
se a reconstrução pode interpretar melhor informação já transmitida.

O budget inicial é 256 updates por braço, com uma única extensão pareada e
predefinida até 1.024; só haverá abertura de avaliação de desenvolvimento se
o ganho normalizado de triagem superar 1% contra origem e controle. Esses
percentuais de loss **não são RMSE físico**. Um novo lookup global adiciona
**1.048.576 bytes de tensor**, que precisam ser cobrados como modelo compartilhado
em qualquer confirmação de codec. Índices fixos por si só não provam igualdade
do stream completo. Sem promoção ou ganho novo estabelecido nesta atualização.
