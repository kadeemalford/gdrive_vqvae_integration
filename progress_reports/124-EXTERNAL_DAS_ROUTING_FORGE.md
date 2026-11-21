# DAS externo: replicação no strain-rate de dois poços do Utah FORGE

## Protocolo prospectivo — 19/11/2026

Segunda população escolhida pelos metadados, antes do resultado do [EGS](123-EXTERNAL_DAS_ROUTING_EGS.md).
Fonte: J. Ajo-Franklin et al., [GDR 1824](https://gdr.openei.org/submissions/1824),
CC BY 4.0, objetos públicos S3 `v1.0.0` do experimento de abril de 2024. O
catálogo está em curadoria. O primeiro cabeçalho de cada poço declara amplitudes
**proporcionais** a strain-rate (`×11600 para nm/m/s`); verificar nos oito.
Preservar a representação IEEE float32 armazenada e converter apenas as métricas
para unidades físicas. Dados disparados por evento, não fluxo contínuo.

- Quatro estágios distintos: 16A_4 e 16A_5 para ajuste; 16A_8 e 16A_9 para
  avaliação. Em cada estágio, o primeiro gatilho por chave do poço 16B
  (Carina) e do 78B (iDAS) deve ter o mesmo timestamp GPS. São dois pares
  de eventos de avaliação, não quatro réplicas. Verificar pais distintos,
  datas, versão, ETag, dimensões e timestamps antes de ler amplitudes.
- Cada objeto contribuirá com os 3.000 tempos e os canais com índices nativos
  `[500,756)`, selecionados sem valores. Conferir SEG-Y big-endian, amostras
  de 1 ms pelo header binário e por traço, canais consecutivos e distância
  de fibra crescente. Não reamostrar nem alinhar espacialmente poços distintos.
  Manter cabeçalho global e todos os cabeçalhos dos traços selecionados no custo.
- Reutilizar o contraste marginal/espaço-temporal, a calibração por canal no
  fit, duas precisões e escolha pelos dois folds de ajuste dos estudos anteriores.
  Dois ajustes separados por poço; cada evento de check paga a instalação,
  transporte, geometria, fonte e rota. Congelar todas as tabelas antes de abrir
  os checks. Publicar economia por estágio, poço e precisão, inclusive braço
  contextual não escolhido; sem combinar com EGS nem alterar a regra.
- Só considerar economia replicada em ambos os poços e precisões se positiva
  nos dois eventos em todas as combinações. A aquisição pareada não atribui
  causalmente diferenças a instrumento ou a fibra: mudam poço, instalação
  e iluminador simultaneamente. Não alegar família óptica sem identificação
  própria. Sem F4-A ou acesso ao FEBUS; CPU.

Execução: `.venv/bin/python -m finetuning.run_external_forge_routing freeze`,
depois `execute`. [Artefatos](../results/experiments/external_forge_routing_20261119/).

## Resultado

Cada estágio de avaliação contribuiu com um evento de três segundos, observado
pelos dois instrumentos. Os estágios 8 e 9 ocorreram em 5 e 6 de abril,
depois dos estágios 4 e 5 de ajuste (3 e 4 de abril). Cada linha abaixo
representa 768.000 células. Taxas calculadas a partir dos bytes completos:

| Estágio | Poço | ε/σ | Marginal (B) | Contextual (B) | Escolha do fit | Economia da rota |
|---|---|---:|---:|---:|---|---:|
| 8 | 16B | 0,03 | 645.945 | 707.518 | marginal | 0% |
| 9 | 16B | 0,03 | 644.165 | 713.293 | marginal | 0% |
| 8 | 16B | 0,1 | 480.296 | 427.665 | contextual | 10,96% |
| 9 | 16B | 0,1 | 478.626 | 433.709 | contextual | 9,38% |
| 8 | 78B | 0,03 | 627.653 | 577.781 | contextual | 7,95% |
| 9 | 78B | 0,03 | 640.837 | 579.086 | contextual | 9,64% |
| 8 | 78B | 0,1 | 461.498 | 366.828 | contextual | 20,51% |
| 9 | 78B | 0,1 | 474.606 | 366.906 | contextual | 22,69% |

A vantagem da política se repetiu nos dois eventos em **três das quatro
combinações** de poço e precisão. Em 16B a 0,03 o retorno ao marginal evitou
aumentos de 9,53% e 10,73%. Em 78B houve economia nas duas precisões.
No conjunto dos dois poços/estágios, a economia foi **4,36% / 0,291 bpc**
e **15,83% / 0,781 bpc**, respectivamente. Não houve economia estrita em
todas as combinações, nem o mínimo agregado de 10% na precisão estrita.

Todos os métodos recuperaram os mesmos q e waveform quantizado; zero
violações. Na escala nominal declarada no cabeçalho, os limites de erro
para 16B foram 8.799 / 29.330 nm/m/s e para 78B, 16.878 / 56.261 nm/m/s.
Esses valores são conversões do fator escrito pelo exportador, não calibração
independente nem tolerância operacional. O alvo aqui é o **SEG-Y exportado**.
O cabeçalho textual C11 rotula o intervalo ambiguamente como `1000MS`;
os campos binário e de todos os traços selecionados concordam em 1.000 µs.

177 s em CPU. A [auditoria](../results/experiments/external_forge_routing_20261119/delivery_audit.json)
reconciliou 128 arquivos de entrega e redecodificou 32 streams usando somente
os objetos. [Resultados e decomposição](../results/experiments/external_forge_routing_20261119/results.json);
[protocolo original e planos com hashes](archive/external_das_20261119/).

**Decisão conjunta com 123:** evidência externa de ganho **seletivo** em
strain-rate; no Terra15 a política não obteve ganho. Preservar a regra sem
reajuste. Não misturar m/s com strain-rate, extrapolar os dois eventos
para produção contínua, ou iniciar F4-A para FEBUS. O [relatório de instalação
16B](https://gdr.openei.org/submissions/1603) descreve inclusive uma fibra
engenheirada Constellation; a presença de vários tipos no cabo por si só não
estabelece um contraste controlado entre fibras. A próxima evidência de
maior valor é DAS nativo de duas fibras sob a mesma excitação com sessões
repetidas e vínculo verificável entre bruto, fibra e interrogador. O
[artigo de Aquistore](https://cseg.ca/simultaneous-acquisition-of-distributed-acoustic-sensing-vsp-with-multi-mode-and-single-mode-fiber/)
documenta esse pareamento SM/MM; não foi identificado download dos brutos
pareados nas fontes consultadas. O acervo [BIGORRE](https://doi.org/10.1093/gji/ggae382)
documenta eventos de uma aquisição contínua FEBUS de três semanas, não sessões
repetidas equivalentes à Configuration 8. Nenhum desses acervos foi avaliado aqui.
