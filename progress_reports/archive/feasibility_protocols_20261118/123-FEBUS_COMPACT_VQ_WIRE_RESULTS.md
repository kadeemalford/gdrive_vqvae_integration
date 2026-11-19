# FEBUS: reempacotamento VQ economiza 2,9944% no painel congelado

Resultado do [protocolo 122](122-FEBUS_COMPACT_VQ_WIRE_PROTOCOL.md).
O transporte da VQ madura passou de **1.119.421 para 1.085.901 B** para
os mesmos **136 tiles e 12.799.000 células**: **33.520 B** a menos, de
**0,69969 para 0,67874 bpc**, com receiver e registry instalados.
O limiar prospectivo de 3% **não passou**: faltaram 63 bytes. Não arredondar
para transformar o resultado em aprovação.

O objeto medido inclui geometria, cabeçalho, offsets e integridade global.
A geometria foi preservada byte a byte e o leitor verificou o acesso aos tiles
em ordem reversa. Para os 134 frames FAK1, a identidade do frame interno e
a ausência de overlay efetivo comprovam equivalência da entrada no decoder
instalado; **não houve nova inferência nem roundtrip neural de waveform**.
Para os dois literais, verificou-se a igualdade dos bits restaurados.

A decomposição impede extrapolar os 2,9944% como ganho geral:
os dois tiles literais respondem por **28.236 B**, cerca de 84% da economia
líquida. Os outros 134 tiles poupam **5.360 B**, **0,49%** dos seus bytes,
já incluindo entradas no índice e antes dos 76 B de cabeçalho e hash comuns.
O resultado não altera pesos, índices, tolerância ou fidelidade; é ganho de
formato em uma população com tiles constantes. A instalação comum não foi
carregada nem incluída no denominador de transporte.

**Decisão:** preservar a implementação como melhoria pontual de empacotamento,
sem promover a VQ, substituir a referência histórica ou ajustar o formato
para recuperar os 63 bytes. Qualquer adoção exigiria um teste prospectivo
com frequência representativa de tiles constantes. **Oito testes passaram.**
[Resultados e hashes](../results/experiments/feasibility_compact_bundle_20261118/results.json).
Sem treino, GPU, rede ou leitura das reservas.
