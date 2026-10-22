# Especialistas e controle balanceado: ganho pequeno, sem promoção de MoE

## 1. Resultado

Treinos, avaliação completa e join de roteamento foram concluídos.
Fonte: `results/experiments/domain_specialization_pilot_20261012/`, summaries
por braço, `full_validation/`, `routed.summary.csv`, `routed.groups.csv`,
`routing.json` e marcador `COMPLETE`.

| Sistema | Correlação macro | RMSE relativo macro | Pior família RMSE | p90 canal macro |
|---|---:|---:|---:|---:|
| Generalista congelado | 0,682337 | 0,719900 | **0,817242** | 1,228475 |
| Shared uniforme | 0,688106 | 0,716104 | 0,818694 | 1,238731 |
| Balanced compartilhado | **0,690911** | 0,714201 | 0,831837 | 1,232500 |
| Roteado por família | 0,688383 | **0,710335** | **0,817242** | **1,216491** |

São as mesmas 5.820 janelas de desenvolvimento, não teste independente. Roteado
tem ganho de RMSE ~0,54% relativo frente ao Balanced, mas correlação menor. Nem
diferença estatística/robustez entre seeds nem fidelidade suficiente foram demonstradas.

## 2. Onde o sistema ganha e perde

| Família/configuração | Frozen RMSE | Shared | Balanced | Roteado |
|---|---:|---:|---:|---:|
| Doped / 5m SMF | 0,673827 | 0,675631 | 0,676965 | 0,673827 |
| Multicore / C1 and C4 | 0,793277 | 0,788318 | 0,788175 | 0,783728 |
| Multimode / Configuration 7 | 0,817242 | 0,818694 | 0,831837 | 0,817242 |
| Single-mode / Configuration 11 | 0,505409 | 0,451566 | 0,380990 | **0,370322** |
| Single-mode / Configuration 3 | **0,705322** | 0,718731 | 0,732546 | **0,758505** |
| Single-mode / Configuration 7 | 0,575028 | 0,575023 | 0,565944 | 0,570801 |

Expert doped e multimode selecionaram a inicialização: preservar o checkpoint
inicial evitou promover adaptações piores. Multimode terminou com RMSE do piloto
~0,968, muito pior que o ponto inicial; isso reforça a dificuldade de transferência
entre suas configurações, sem provar uma causa única.

O ganho do roteado está especialmente em single-mode Configuration 11. Porém,
Configuration 3 piora de 0,705322 para 0,758505 e p90 por canal de 1,382117 para
1,611098. Melhor p90 macro não significa melhora de toda configuração.

## 3. Custos e interpretação

- Rota era fixa pela família antes do treino, sem oracle de erro.
- Balanced recebeu 65.536 updates, exatamente a dose por família dos quatro
  experts somados; o roteado retém também Shared/fallback, com custo adicional.
- Balanced altera dose, não pesos relativos de configurações dentro de cada família.
  Dose igual não implica mesma ordem, optimizer ou wall-clock.
- Índices+side+rota nominal pooled: **165,828x** roteado, versus **165,914x** nos
  generalistas. Diferença pequena no payload, não no armazenamento de modelos.
- Registry roteado com cinco modelos: ~**559 MB de parâmetros float32**, contra
  ~112 MB de um generalista, sem amortização de implantação definida.
- Fallback funcional para metadados ausentes/desconhecidos não é auditoria de
  fidelidade em uma família nova.

**Não escalar para gating aprendido/MoE por ora.** A especialização testada não
resolveu doped/multimode e seu ganho macro não justifica automaticamente cinco
modelos nem as regressões single-mode. Preservar candidates como referências,
sem eliminar a linha para futuros grupos/seeds ou representação diferente.

## 4. Próximo gargalo: alocação espaço-temporal

Os candidatos atuais investem mais índices no tempo, mantendo compressão espacial
por 16. Testar se parte do orçamento deveria preservar canais em vez de apenas
aumentar contexto/adaptação/uso de códigos. Comparar geometria temporal atual,
equilíbrio temporal/espacial de mesma taxa e geometria predominantemente espacial.

Isso retorna às etapas 4/8, sem novo dataset ou mudança de objetivo/quantizador.
Protocolo no [relatório 29](29-SPATIOTEMPORAL_RATE_ALLOCATION_PROTOCOL.md).
Se a alocação não ajudar, reconsiderar taxa total/capacidade/diversidade com
controles explícitos, não simplesmente prolongar especialistas ou adicionar gating.
