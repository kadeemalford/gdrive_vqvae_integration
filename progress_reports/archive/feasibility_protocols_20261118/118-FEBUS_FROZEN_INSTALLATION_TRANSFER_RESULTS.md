# FEBUS: a economia transfere, mas não atinge o critério nas duas precisões

Resultado do [protocolo 116](116-FEBUS_FROZEN_INSTALLATION_TRANSFER_PROTOCOL.md)
em **380 janelas de 19 arquivos**, oito configurações e três famílias sem
multicore. Tabelas do estudo 114 reutilizadas sem ajuste; instalação cobrada
uma vez por configuração. Os arquivos são adicionais para este contraste,
não sessões independentes.

| ε/σ de calibração | Economia líquida contra marginal | Economia em bpc | Critério de taxa |
|---|---:|---:|---|
| 0,03 | 616.320 B (5,20%) | 0,358 | Não: abaixo de 10% |
| 0,1 | 1.150.903 B (14,10%) | 0,668 | Sim |

Nenhuma configuração perdeu mais de 1%. Os resultados são da execução
[concluída](../results/experiments/feasibility_npz_transfer_20261118/retry_01/results.json)
com o plano congelado original; a tentativa interrompida foi preservada.
Amortização e transferência somaram **3.240 roundtrips**, zero violações
do limite de erro e 53 testes aprovados. Apenas CPU e NPZ de treino.

**Decisão: o critério conjunto não passou.** A oportunidade a 0,1 se manteve
em outros arquivos com instalação cobrada; a 0,03 é positiva, mas abaixo
do mínimo fixado. Não relaxar a precisão para declarar sucesso, não repetir
o painel com outra seleção e não iniciar F4-A. Este estudo sustenta
reutilização da instalação nesse recorte, mas não define uma tolerância
científica ou uma taxa aceitável de produto. O próximo passo com maior
valor é obter o contrato de erro e o horizonte operacional do uso, além da
documentação temporal e de novas sessões para as perguntas ainda bloqueadas.
