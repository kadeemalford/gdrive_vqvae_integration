# FEBUS: as tabelas congeladas se pagam nos arquivos de check

Resultado do [protocolo 115](115-FEBUS_FROZEN_INSTALLATION_AMORTIZATION.md).
Foram processadas 160 janelas adicionais nos mesmos oito arquivos de check.
Nenhuma tabela foi reajustada. Os valores abaixo são economias agregadas
contra o marginal, cobrando transporte e instalação uma vez por configuração.

| ε/σ de calibração | Economia líquida a 20 janelas por arquivo | Primeiro ponto de amortização observado |
|---|---:|---:|
| 0,03 | 79.796 B (1,41%) | 16 janelas por arquivo |
| 0,1 | 440.330 B (11,61%) | 4 janelas por arquivo |

Foram verificados 960 roundtrips e 192 artefatos cumulativos, sem violação
do limite de erro. [Artefatos](../results/experiments/feasibility_npz_amortization_20261118/).

O resultado demonstra amortização neste recorte e horizonte, não transferência
entre arquivos nem viabilidade generalista. Na precisão mais estrita, a
economia é pequena; o gatilho de 115 exigia economia positiva nas duas
precisões, não 10%. O [teste de transferência 116](116-FEBUS_FROZEN_INSTALLATION_TRANSFER_PROTOCOL.md),
com critério mais forte, foi iniciado em CPU em 19 arquivos compatíveis,
sem reajustar tabelas. Nenhuma GPU, reserva ou sessão nova foi usada.
