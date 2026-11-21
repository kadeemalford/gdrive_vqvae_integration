# DAS externo: o roteamento se repete entre gravações de um Terra15?

## Protocolo prospectivo — 19/11/2026

Primeira replicação externa do **método** do [120](120-FEBUS_FIT_ONLY_INSTALLATION_ROUTING.md).
Dados: V. Rodriguez Tribaldos e C. Hopp, [EGS Collab Experiment 2, GDR 1606](https://gdr.openei.org/submissions/1606),
CC BY 4.0. O cabo contém fibras de mais de um tipo, mas os três interrogadores
DAS do depósito estavam nas fibras monomodo. Nesta medição: apenas Terra15
T216125, um sítio e uma montagem. O produto dos cabeçalhos é **velocity, m/s**.
Codificar o float32 nativo, sem conversão para strain-rate, suavização, seleção
de canais por amplitude ou supressão de ruído.

- Sessão operacional = início de gravação identificado pelo instrumento
  (`saving_start_computer_time`, `seq_00000000000`). Seleção por cabeçalho, não por
  energia. Uma por dia, com os mesmos 268 canais, dx de 2,4503 m, GL de 4,9 m
  e 36 kHz. Fit A/B: 17 de março (série `taps`) e 18 de março; avaliação C/D:
  22 de março e 17 de maio de 2022. Não inferir remontagem, independência
  estatística ou excitação presente em uma janela pelo nome da série.
- Em cada gravação: todos os canais nos intervalos `[57600,59600)` e
  `[576000,578000)`. Dois trechos descontínuos, 2.144.000 células no check
  e outras tantas no fit. Sem extrapolar o horizonte de instalação.
  Fixar chaves, ETags S3, cabeçalhos, clocks e fontes antes dos valores;
  calcular hashes das fatias efetivamente usadas. Sem acesso a reservas.
- Em cada fold, `σ` = mediana dos desvios temporais positivos dos canais/trechos
  do **treino daquele fold**. No ajuste final usar A+B. Comparar ε/σ = 0,03
  e 0,1 na mesma sequência discreta. Usar quantização, seleção de suporte
  marginal, CDFs, rANS, resets e envelope dos estudos anteriores. Exigir
  economia ≥3% de bytes completos nos dois folds de fit para rotear ao
  espaço-temporal. Congelar as decisões **e todas as tabelas finais** antes
  de ler valores de C ou D. Medir marginal, espaço-temporal e rota escolhida;
  publicar ambos os resultados por gravação mesmo quando a rota for marginal.
- Unidade de implantação: cada gravação avaliada isoladamente, com uma instalação
  por braço/precisão; dois trechos com bindings e metadados, envelope, fonte de
  leitura/runner e identificador de rota pagos. Python/stdlib instalados.
  Requerer q e reconstrução quantizada idênticos entre todos os braços e
  zero violações do limite absoluto. Reportar totais e decomposição; só há
  economia replicada nas duas precisões se for positiva em C **e** D nas duas.
- Contraste secundário de diversidade, viabilizado pelos dois trechos por
  gravação: mesmo coder espaço-temporal, mesmo K e escala calibrados em A+B,
  mesmos 1.072.000 exemplos de ajuste de **CDF** por braço. Comparar a média
  dos bytes de treinar as contagens nos dois trechos de A ou nos dois de B
  à média das duas combinações balanceadas de um trecho por gravação
  (A0+B1 e B0+A1). Publicar as quatro medições, sem escolher a melhor. Isso
  mede diversidade **nas contagens**, condicional à calibração comum; não é
  uma curva de treino neural, de convergência ou de causalidade entre fibras.
- Sem seleção por resultados, extensão de duração, redivisão ou F4-A. Se faltar
  compatibilidade de cabeçalho, acesso ou integridade, registrar bloqueio.
  Nenhuma inferência sobre repetibilidade da Configuration 8, outras fibras
  ou o desempenho relativo à VQ madura em FEBUS. CPU; zero GPU.

Execução: `.venv/bin/python -m finetuning.run_external_egs_routing freeze`,
depois `execute`. [Artefatos](../results/experiments/external_egs_routing_20261119/).

## Resultado

A política escolheu **marginal nas duas precisões**. As economias contextuais
nos folds de ajuste foram −0,008% e −0,009% a 0,03; 9,29% e 1,31% a 0,1.
O segundo fold não chegou aos 3% exigidos. A replicação de economia da **rota
selecionada** falhou: empate exato com o marginal nos quatro pontos de check.

| Gravação | ε/σ | Marginal = roteado (B) | Contextual (B) | Economia contextual |
|---|---:|---:|---:|---:|
| C, 22/03 | 0,03 | 1.007.102 | 1.007.059 | 0,004% |
| D, 17/05 | 0,03 | 1.343.827 | 1.343.692 | 0,010% |
| C, 22/03 | 0,1 | 773.589 | 746.062 | 3,56% |
| D, 17/05 | 0,1 | 1.137.420 | 1.036.861 | 8,84% |

Existe oportunidade **não capturada** a 0,1. Não atribuir essa economia à
política nem recalibrar o limiar nos checks. As tolerâncias foram 1,6894×10⁻⁶
e 5,6313×10⁻⁶ m/s; q e reconstrução idênticos entre os métodos, sem violações.

**Diversidade a número de células constante:** média dos dois ajustes mistos
contra média dos dois ajustes de uma gravação. Economia em C/D de
**0,008% / 0,189%** a 0,03 e **0,008% / 0,276%** a 0,1. Direção favorável
nas quatro comparações, pequena neste orçamento e condicionada ao K/σ comuns.
Não houve benefício grande demonstrado; dois inícios de gravação no ajuste
não estabelecem uma lei de escala. As combinações e bytes individuais estão
no [resultado](../results/experiments/external_egs_routing_20261119/results.json).

208 s em CPU. A [auditoria dos objetos](../results/experiments/external_egs_routing_20261119/delivery_audit.json)
conferiu 96 arquivos, redecodificou 64 streams e reconciliou os bytes de cada
contraste. O [protocolo original](../results/experiments/external_egs_routing_20261119/protocol.original.md)
foi preservado pelo hash do plano. Cada gravação foi representada por apenas
0,111 s no total, em duas regiões temporais; resultados não medem todo o ensaio.
