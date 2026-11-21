# FEBUS: roteamento fit-only evita regressões no painel de desenvolvimento

Resultado do [protocolo 120](120-FEBUS_FIT_ONLY_INSTALLATION_ROUTING.md).
Os 40 folds de arquivos fit produziram 18 decisões congeladas antes do
ledger check. Nenhum sinal check foi relido ou recodificado.

| ε/σ_cal | Economia com instalação compartilhada | Economia em bpc | Contextual selecionado |
|---|---:|---:|---|
| 0,03 | **21,75%** | 1,263 | single-mode/Configuration 8 |
| 0,1 | **31,82%** | 1,293 | multimode/Configuration 7; single-mode/Configurations 3 e 8 |

As demais configurações ficaram no marginal. O efeito é relevante para
implantação seletiva: houve economia agregada e nenhuma configuração
ultrapassou a guarda de regressão. **O critério generalista do 107 não
passou:** famílias que ficaram integralmente no marginal não economizaram,
e C1/C4 continua ausente. Não alterar a definição de sucesso retrospectivamente.

**Direção:** a evidência favorece o sistema seletivo com quantização explícita,
não instalar contexto em todas as fibras. Confirmar em sessões ou arquivos
compatíveis adicionais com a regra e as instalações congeladas, pagando
seus custos completos. Os checks já foram consumidos; sem uma população
nova, repetir avaliações não fornece confirmação. F3 ainda depende de novas
sessões documentadas e este resultado não inicia F4-A.

Execução em CPU: 228 s, 66 testes aprovados. Auditoria dos 729 ZIPs D2c
e 243 envelopes sem divergências. A ausência de hashes históricos dos
slices fit é uma limitação explícita. Sem treino neural, GPU ou reserva.
[Resultados](../results/experiments/feasibility_fit_routing_20261118/results.json)
e [auditoria](../results/experiments/feasibility_fit_routing_20261118/post_execution_audit.json).
