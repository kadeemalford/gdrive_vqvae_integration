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
