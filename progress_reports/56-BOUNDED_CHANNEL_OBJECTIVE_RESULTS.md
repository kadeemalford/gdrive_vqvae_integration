# Objetivo por canal: admissão exploratória positiva, ainda com margens pequenas

## Resultado completo

Fonte: `results/experiments/channel_tail_pilot_20261019/`, `COMPLETE`,
`full_validation/COMPLETE`, `selection.json`, oito snapshots e históricos.
Novo braço concluiu 32.768 updates; controle full do relatório 54 foi reutilizado,
sem repetir treinamento. Mesmas 5.920 janelas completas e referência robusta.

| Candidato 4x | RMSE macro | Correlação | p90 macro por canal | RMS macro | Guarda contra baseline |
|---|---:|---:|---:|---:|---|
| Baseline/source epoch 24 | 0,479352 | 0,869126 | 0,717070 | 0,901455 | Referência |
| Full sem auxiliar, fase 8 | **0,476036** | **0,870449** | 0,715337 | 0,896813 | Rejeitado |
| Full + auxiliar 0,05, fase 8 | 0,476394 | 0,870237 | **0,712982** | 0,896346 | **Elegível** |

Peso/cap permaneceram fixos em 0,05/4. O selecionado macro e o guardado coincidem
na fase 8. **Fases 4 e 8 foram elegíveis; as outras seis não passaram**. Não é
resultado robusto em toda a trajetória, nem autorização para escolher pesos de
loss adicionais nessa validação.

Frente ao baseline, redução relativa de RMSE ~**0,62%** e de p90 macro ~**0,57%**.
Frente ao controle full, RMSE macro é **pior em ~0,000358** (~0,075%), mas p90
melhora em ~0,002355 (~0,33%). É uma troca de critérios, não dominância universal.
Taxa nominal pooled permanece **42,811x**, sem novo parâmetro/side no codec.

## Seis configurações

| Família/configuração | RMSE auxiliar | Correlação auxiliar | p90 por canal auxiliar |
|---|---:|---:|---:|
| Doped / 5m LSF | 0,430827 | 0,904034 | 0,794347 |
| Multicore / C1 and C3 | 0,524780 | 0,849597 | 0,642965 |
| Multimode / Configuration 6 | 0,564381 | 0,824930 | 0,671968 |
| Single-mode / Configuration 10 | 0,205247 | 0,975211 | 0,668214 |
| Single-mode / Configuration 2 | 0,602482 | 0,795266 | 0,844115 |
| Single-mode / Configuration 6 | 0,349035 | 0,936679 | 0,715617 |

RMSE, correlação e p90 melhoram nas seis configurações **frente ao baseline**.
Frente ao controle full, p90 melhora nas seis, mas RMSE e correlação são
ligeiramente piores nas seis. Configuration 2 segue limitante e a melhora de
p90 contra baseline é só ~0,000291. RMS macro cai, embora fase FFT melhore
ligeiramente (0,929120→0,929908). A guarda não cobre toda métrica física possível.

## Auditorias pareadas, sem novas inferências

Runner: `finetuning/audit_physical_pair.py`, reutilizando funções de pareamento,
estatística e bootstrap do auditor anterior. Exige protocolos físicos de validação
compatíveis, mesma referência, coordenadas, custos e diagnósticos do alvo.
Cada comparação contém **5.920 janelas pareadas**, grupos e arquivos:

- `results/experiments/channel_tail_vs_baseline_20261019/`.
- `results/experiments/channel_tail_vs_control_20261019/`.

Melhora das médias de grupo não significa melhora de todas as janelas. Contra
baseline, RMSE piora em **28,3% das janelas multimode**, 24,2% de single-mode
Configuration 6 e 18,7% de Configuration 2. Isso não viola a guarda pré-definida,
que opera nas médias por configuração; não omitir essas exceções.

Bootstrap descritivo de arquivos (5.000 reamostragens) para delta RMSE contra
baseline cruza zero em Configuration 2 e Configuration 6. Os arquivos podem ser
cópias/correlacionados; não interpretar como inferência confirmatória ou evidência
de independência entre aquisições. As pequenas margens exigem confirmação.

## Decisão

Preservar o candidato e os endpoints; **não promover como modelo final nem
ativar a loss por default**. O ganho formal é de admissão exploratória sob os
três critérios deste protocolo, não de tolerância física universal atendida.

Próximo passo: confirmar a mesma regra **sem retunar peso/cap** em outro fold e
seed, com controle novo pareado e source exclusivo desse fold. Não transferir
pesos do fold 0 para o fold 1. Protocolo no
[relatório 57](57-CHANNEL_OBJECTIVE_FOLD_CONFIRMATION_PROTOCOL.md).

O objetivo foi motivado por validação reutilizada. Mesmo a confirmação entre
folds não substitui aquisições externas, tolerâncias de aplicação ou bitrate
aceitável. Nenhuma nova continuação do braço já concluído foi iniciada.
