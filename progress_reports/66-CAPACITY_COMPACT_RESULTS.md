# Resultado da expansão nativa a taxa compacta fixa

## Conclusão

O estudo remoto `native_capacity_compact_fold1_20261029` terminou em
**29/10/2026 às 23:32 UTC (20:32 de Brasília)**. Ambos os braços receberam a
extensão predefinida e completaram **24 épocas/98.304 updates**. As avaliações
completas e guardas terminaram; nenhum modelo foi promovido.

| Braço | Parâmetros | Época selecionada | RMSE macro | Correlação | p90 canal macro | Razão nominal |
|---|---:|---:|---:|---:|---:|---:|
| Maior | 96.517.033 | 23 | **0,674562** | **0,724291** | 1,079955 | 157,357× |
| Controle | 27.958.081 | 22 | 0,684932 | 0,716521 | 1,082335 | 157,357× |

A rede maior reduziu o RMSE macro em **1,51%**, mantendo exatamente o ponto
compacto. Na fase 16, a direção já era favorável: **0,688197 / 0,712518** contra
**0,695175 / 0,706438** em RMSE/correlação. Ganho positivo, mas modesto; não
demonstrou uma mudança de patamar de fidelidade ou aproximação de 700×.

## Limites por configuração/canal

As guardas identificaram **cinco regressões** da rede maior contra o controle:

- Doped 5m SMF: RMSE +0,000048 e correlação −0,000371.
- Multicore C1 and C4: p90 +0,000735.
- Single-mode Configuration 11: p90 **+0,198380**, a regressão mais importante.
- Single-mode Configuration 3: p90 +0,004594.

São deltas descritivos, não testes de significância nem tolerâncias físicas.
Uma melhora minúscula de p90 macro não permite ignorar o prejuízo localizado.
Codebook utilizado na última época: **292/2.048** na maior e **243/2.048** no
controle; esse diagnóstico não prova causalidade nem justifica refresh automático.

### Curvas e hipótese de overfitting

A reconstrução temporal de treino caiu de **0,4896 para 0,4725** na rede maior
entre as épocas 16/24; na validação de triagem, de **0,4601 para 0,4542**.
No controle, as mesmas perdas caíram de **0,4931 para 0,4766** e de **0,4663 para
0,4605**, respectivamente. Portanto não há nessa fase o padrão simples de treino
melhorando enquanto validação piora que explicaria o ganho pequeno por overfitting
dominante. As perdas de treino e validação têm amostragem/ponderação diferentes;
comparar seus níveis diretamente não estima um gap de generalização. Não há
evidência suficiente para atribuir a limitação somente ao número de parâmetros,
somente à falta de dados ou somente à quantização.

## Custo e interpretação

Tempo somado das épocas: **2,44 h** na maior e **0,94 h** no controle, sem
avaliações completas ou inicialização; aproximadamente **2,59×** de custo temporal
para 3,45× de parâmetros. Ambas usaram somente a RTX 4090 física 0; a GPU 1
permaneceu fora da execução. A estimativa inicial foi conservadora porque o smoke
media exclusivamente a janela máxima e não a distribuição real de tiles.

Este é um contraste de **largura da CNN nativa, scratch, uma seed e um fold**,
não uma refutação de redes grandes, contexto multiescala ou atenção. Também não
permite concluir que mais dados resolveriam o restante ou que todos os detalhes
foram perdidos irrecuperavelmente na quantização. Não comparar seus agregados
diretamente com a taxa densa ou o fold anterior como se fossem a mesma coorte.

## Próxima decisão

Preservar os dois checkpoints e encerrar o orçamento, sem prolongamento automático
ou salto para redes ainda maiores. O aumento simples de largura não ofereceu o
ganho generalista forte pretendido. Antes de outro treino, revisar as caudas,
especialmente single-mode 11, e o uso do gargalo; investigar codificação entrópica
congelada e desenhar o controle ROI com piso global conforme o roadmap. Uma
confirmação de capacidade deve ser predefinida se esse ganho modesto justificar
seu custo; não é promoção ou autorização para combinar várias intervenções.

As sínteses JSON, CSVs de métricas/grupos e protocolo foram copiadas para
`results/experiments/native_capacity_compact_fold1_20261029/` na máquina local.
Pesos, logs e snapshots completos permanecem no workspace remoto. O teste
reservado não foi inferido. Nenhum novo treino foi enfileirado após a conclusão.
