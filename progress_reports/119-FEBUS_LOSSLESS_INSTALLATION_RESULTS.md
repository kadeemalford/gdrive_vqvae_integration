# FEBUS: instalação reversível melhora a taxa, mas não uniformemente

Resultado do [protocolo 118](118-FEBUS_LOSSLESS_INSTALLATION_ENVELOPE.md).
Foram reconstituídas **243 instalações byte a byte** e conferidos os 729
ZIPs de origem. O transporte e as tabelas efetivas não mudaram. Todos os
envelopes escolheram XZ; seu overhead foi de 85 B por instalação, mais
2.190 B da fonte incremental do decoder, cobrada uma vez no contrato abaixo.

| ε/σ_cal | Total contextual original | Total contextual novo | Economia contra marginal novo |
|---|---:|---:|---:|
| 0,03 | 4.278.651 B | 3.827.992 B | 10,19% / 0,589 bpc |
| 0,1 | 2.195.529 B | 2.011.709 B | 32,36% / 1,304 bpc |

Os dois agregados passam os mínimos de taxa, **mas as guardas de cobertura
não passam**. A 0,03, só single-mode economiza no agregado da família;
doped, multicore C2/C3 e multimode ficam mais caros. A 0,1, doped regride,
assim como single-mode/Configuration 11 individualmente. C1/C4 segue fora.
Não interpretar a média como economia universal nem selecionar retrospectivamente
em quais configurações usar a instalação contextual.

**Decisão:** preservar a redução reversível como melhoria de implantação.
Ela remove parte real do custo sem relaxar ε, mas não autoriza F4-A nem uma
nova varredura de envelopes. O próximo mecanismo, se avaliado, teria de
demonstrar transferência ou controlar a escolha marginal/contextual antes
do check, não repetir a escolha retrospectiva nos mesmos dados.

Execução em CPU: 8,02 s; 16 testes com regressões, oito deles novos.
Sem sinais, GPU, reservas ou treino.
[Resultados](../results/experiments/feasibility_installation_envelope_20261118/results.json)
e [verificações](../results/experiments/feasibility_installation_envelope_20261118/completion_evidence.json).
