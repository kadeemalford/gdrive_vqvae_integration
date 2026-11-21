# FEBUS: confirmação do roteamento em novas aquisições

Preparação prospectiva para confirmar o [roteamento decidido no fit](121-FEBUS_FIT_ONLY_INSTALLATION_ROUTING_RESULTS.md).
O [contrato congelado](../results/experiments/feasibility_routing_confirmation_preflight_20261118/contract.json)
identifica por hash as 18 decisões e os 22 envelopes necessários ao
contraste entre roteamento e marginal. Zero sessões admitidas e zero sinais lidos.

## Admissão

Fornecer HDF5 brutos de novas aquisições e preencher o
[manifesto de aquisição](../config/new_febus_acquisitions.template.csv),
indicando a configuração histórica correspondente. Confirmar sessão, data,
dispositivo, montagem/excitação e relações entre bruto, derivados e canais
simultâneos. Repetições em novas ocasiões com a mesma montagem medem
repetibilidade; montagens ou excitações diferentes medem transferência.
Não separar exportações nem núcleos simultâneos como sessões independentes.

Comparar primeiro cabeçalhos e mapa físico com o contrato instalado.
Uma configuração ou geometria não suportada não deve ser forçada a caber:
registrar cobertura ausente. **C1/C4 não tem rota treinada**; sua coleta é
útil para cobrir a lacuna, mas não entra como confirmação de uma rota que
não existe. Para testar a rota contextual na precisão estrita, priorizar uma
repetição de **single-mode/Configuration 8** com a geometria compatível.

## Avaliação após admissão

Congelar seleção e horizonte antes de ler valores. Nas precisões 0,03 e 0,1,
usar as escalas, suporte, CDFs e decisões já congelados, sem adaptação.
Comparar marginal e roteamento sobre os mesmos valores quantizados e cobrar
instalação, geometria e roteamento. Verificar roundtrip e limite de erro;
reportar taxa por sessão/configuração e separar ausência de regressão de
economia. Não promover automaticamente por um agregado favorável.

**Estado:** aguardando novos HDF5 com proveniência. Sem usar validação
consumida, teste fold0 ou EDAS como substitutos.
