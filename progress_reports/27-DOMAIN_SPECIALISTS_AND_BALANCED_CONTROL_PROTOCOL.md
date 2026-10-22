# Especialistas por família e controle compartilhado de exposição

> Experimento concluído; comparação e decisão no
> [relatório 28](28-DOMAIN_SPECIALIZATION_RESULTS.md).

## 1. Hipótese e fonte

Investigar se especialização transfere melhor para configurações reservadas,
em vez de tratar uso do codebook ou mais updates como solução única. Todos os
braços começam dos mesmos pesos **native transferido 4x epoch 13**, melhor referência
macro atual, com **AdamW novo/LR 3e-5**. Sem mudança de arquitetura, quantizador,
alvo bruto, normalização escape-v1, corpus ou splits; refresh fica desligado.

Runner: `finetuning/run_domain_specialization_experiment.py`.
Saída: `results/experiments/domain_specialization_pilot_20261012/`.

## 2. Dados, doses e controles

| Família | Janelas treino | Janelas val | Canais treino |
|---|---:|---:|---:|
| Doped | 1.300 | 1.280 | 45 |
| Multicore | 7.820 | 2.600 | 70 |
| Multimode | 520 | 560 | 512 |
| Single-mode | 2.500 | 1.380 | 23–512 |

Corpus tem desequilíbrio importante em janelas e custo por janela. Especializar
aumenta a exposição de famílias minoritárias; isso não prova sozinho interferência
entre famílias. Por isso a comparação tem:

| Braço | Treino | Orçamento adicional |
|---|---|---:|
| Shared | uniforme por janela no corpus inteiro | 16 x 1.024 = **16.384** |
| Balanced | famílias em dose igual, pesos por janela dentro da família | 16 x 4.096 = **65.536**, exatamente 16.384/família |
| Expert doped | somente treino doped | 16 x 1.024 = **16.384** |
| Expert multicore | somente treino multicore | 16 x 1.024 = **16.384** |
| Expert multimode | somente treino multimode | 16 x 1.024 = **16.384** |
| Expert single-mode | somente treino single-mode | 16 x 1.024 = **16.384** |

Total: **147.456 updates**, sequenciais na GPU. Balanced iguala contagem conjunta
dos quatro especialistas e dose por família, **não wall-clock, sequência exata de
janelas ou estados de optimizer**. O sistema roteado também retém Shared como
fallback, aumentando seu compute total e armazenamento. Não declarar custo total igual.

Sampler `family-window` intercala famílias com dose exata e preserva ponderação
por janela dentro de cada família. Não é o `family-group` histórico, que também
repondera configurações. As configurações não são movidas entre train/val/test.

Expert multimode recebe ~31,5 passagens equivalentes pelo tamanho de seu pool,
contra ~2,1 no multicore. Isso pede curvas de overfitting, não promoção por perda
de treino. Patience 17 garante o orçamento de 16 épocas; inicialização é candidata.

## 3. Seleção, roteamento e avaliação

- Shared/Balanced selecionam por RMSE macro em piloto quatro-famílias.
- Cada expert treina e seleciona **somente na própria família**, com o mesmo split
  de configurações e cap de duas janelas/arquivo. Os subsets por família não
  precisam coincidir janela a janela com o piloto compartilhado; hashes são salvos.
- Depois, avaliar frozen generalist, Shared, Balanced e os quatro experts nas
  **mesmas 5.820 janelas completas**, sem updates.
- Rota fixada antes do treino: família conhecida → expert correspondente;
  família desconhecida/ausente → Shared. Sem seleção por erro do original,
  gating aprendido ou média de reconstruções.
- O join usa apenas metadados, preserva todos os scores e contabiliza **um byte
  de rota por janela** além de índices+side. Não altera/checkpointa modelo durante avaliação.
- Não trocar automaticamente para Shared numa configuração que tenha score ruim;
  isso seria outra decisão de desenvolvimento, não confirmação desse protocolo.
- Comparar sistema roteado com frozen, Shared e Balanced, por configuração e
  cauda de canais, não apenas média macro.

Roteamento/fallback implementados em `finetuning/domain_routing.py` e join em
`finetuning/summarize_domain_routing.py`. Fallback é testado funcionalmente,
mas **fidelidade em família realmente desconhecida não foi medida**. Nenhum
acesso ao teste ou auditoria externa; scores são exploratórios.

## 4. Custos e limites de sistema

Cada modelo tem 27.958.081 parâmetros e mesma geometria/K; índice nominal por
janela continua igual. Sistema roteado mantém **quatro experts + fallback**,
~**559 MB decimais de parâmetros float32**, versus ~112 MB para um generalista.
O relatório automático registra bytes de parâmetros e de checkpoints de treino;
estes últimos incluem optimizer/metadados e **não são tamanho de decoder implantado**.

Taxa pooled índices+side+rota permanece parcial: não inclui container/registry
serializado/mapas físicos. Custos fixos de modelos exigem horizonte de amortização
explícito antes de alegar compressão operacional competitiva.

Se Balanced igualar/superar roteado, há menos motivação para manter vários modelos.
Se experts ganharem à dose familiar pareada, confirmar em nova seed/configurações
antes de escolher um sistema. Se ambos falharem nas famílias difíceis, investigar
taxa/capacidade/representação e diversidade, não escalar downloads automaticamente.

## 5. Testes, operação e recuperação

**139 testes passaram**, incluindo filtro por manifest/split, rejeição de mudança
de domínio no resume, fallback sem metadados e dose exata do controle balanceado.
Topologia/GPU máxima não muda frente aos smokes anteriores; nenhum update de
smoke foi reutilizado como inicialização. Dataset e fontes são verificados.

Execução lançada como serviço de usuário **`febus-domain-specialists-20261012`**.
**Resultados pendentes nesta atualização**. Estimativa **8–12 h**, principalmente
pela exposição maior a janelas 512-canais; atualizar ETA com tempos medidos.

Log: `results/experiments/domain_specialization_pilot_20261012.pipeline.log`.
Logs por braço, full_validation e tabelas routed ficam dentro da saída.

```bash
systemctl --user status febus-domain-specialists-20261012

# Somente sem cópia ativa:
.venv/bin/python finetuning/run_domain_specialization_experiment.py \
  --output-dir results/experiments/domain_specialization_pilot_20261012 --resume
```

Protocolo/source/manifest/código fixados por hashes; saves atômicos, lock e resume
por época completa. Serviço independe da sessão interativa, não garante sobrevivência
a logout/reboot. Não sobrescrever resultados/fontes nem executar cópias concorrentes.
