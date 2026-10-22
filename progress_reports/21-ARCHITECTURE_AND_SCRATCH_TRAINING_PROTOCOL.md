# Arquitetura e treino do zero: comparação controlada FEBUS-only

> Treino e avaliação completa concluídos; resultados e decisões no
> [relatório 22](22-ARCHITECTURE_INITIALIZATION_RESULTS.md).

## 1. Mudança de prioridade

Autorizada a investigação de arquitetura e inicialização em vez de continuar
indefinidamente a mesma adaptação. O candidato longo 4x permanece referência:
corr macro ~0,560 e RMSE relativo ~0,810 não representam fidelidade suficiente.

Não trocar dataset, arquitetura e inicialização simultaneamente. Esta comparação
usa **o mesmo corpus FEBUS, split fold 1, alvo bruto, normalização global escape-v1,
quantizador e geometria temporal-4x**. Novas aquisições ficam em uma linha separada,
condicionada à identificação de lacunas e a provenance/licença/unidades.

## 2. Hipóteses e braços

| Braço | Arquitetura | Inicialização | Pergunta |
|---|---|---|---|
| transferred_4x | native-v1 | best_macro do longo 4x, optimizer novo | Referência com mesma exposição adicional/LR do piloto |
| scratch_4x | native-v1 | pesos aleatórios + codebook inicializado somente no treino | A herança dos pesos limita a adaptação FEBUS? |
| scratch_temporal | temporal-dilated-v1 | pesos aleatórios + codebook inicializado somente no treino | Contexto residual temporal anisotrópico ajuda à mesma taxa/capacidade? |

O contraste scratch_4x/scratch_temporal isola a variante de arquitetura dentro
do mesmo protocolo de treino do zero. O contraste transferred/scratch mantém
orçamento **adicional**, não orçamento histórico total: a referência já teve
pré-treino e adaptações. Não apresentar isso como comparação com compute total igual.

### Variante arquitetural local

Em cada bloco residual do encoder/decoder, a segunda convolução 3x3 é substituída
por **1x9**, com dilatações temporais **1/2/4** cíclicas, separadamente em cada lado.
A primeira 3x3 mantém mistura espacial. Normalizações, número de features,
strides, grid, quantizador e decoder de saída permanecem comparáveis.

- **27.958.081 parâmetros em cada braço**, exatamente pareados.
- Para 512x1.000, **8.000 índices** em cada braço; codebook 2.048, 11 bits/índice.
- Nenhum skip transmite o original contornando o quantizador.
- Maior campo temporal altera custo/viés, não os bits nominais por janela.
- Hipótese específica de arquitetura, não busca exaustiva; orientação do kernel
  e dilatação mudam juntas nesta variante. Uma ablação posterior sem dilatação
  seria necessária para atribuir separadamente seus efeitos.
- Implementação em `finetuning/model_factory.py`, sem modificar o repositório
  externo `smartgrids`. Metadata arquitetural e geometry são carregados
  explicitamente; pesos não bastam para restaurar os strides.

### Inicialização do zero e prevenção de artefato de codebook

As duas redes do zero usam construtor aleatório, seed 2026, sem carregar tensor
pré-treinado ou optimizer antigo. Somente dimensões arquiteturais são reutilizadas.
O codebook recebe vetores dos encoders aleatórios em **32 janelas de treino**,
oito por família, até 256 vetores por janela, reservoir determinístico.

Isso evita usar embeddings quase nulos como único teste de viabilidade do treino
do zero. Não é pré-treino supervisionado nem calibração de validação: **zero
updates**, encoder intacto, apenas inicialização de embeddings com latentes de
treino. Essa exposição adicional sem updates é declarada e idêntica em caminhos
de amostra dos dois braços scratch, embora os vetores dependam da arquitetura.

## 3. Orçamento da triagem

| Item | Protocolo |
|---|---|
| Corpus disponível de treino | 12.140 janelas, quatro famílias; nenhuma alteração do dataset |
| Sampling | window, seed 2026, mesma ordem entre braços |
| Updates | **16 épocas x 4.096 = 65.536 por braço**, 196.608 totais |
| Equivalência aproximada | 5,4 passagens pelo tamanho do corpus, não 16 passagens completas |
| Objetivo | 0,5 MSE + 0,5 L1 temporal + VQ; células escapadas fora da loss |
| Otimizador | AdamW novo em todos, LR **1e-4**, weight decay 1e-5, grad clip 1 |
| Validação por época | 102 janelas, duas por arquivo, quatro famílias/seis configurações |
| Seleção | menor RMSE macro restaurado, incluindo epoch 0 |
| Parada precoce | patience 17: **não corta** o orçamento fixo de 16 épocas |
| Auditoria após treino | candidatos selecionados nas **5.820 janelas completas** |
| Teste | não consultado |

LR é pareada e maior que a continuação anterior de 3e-6; a referência atual
fica intacta e como candidata inicial. Uma LR comum não implica LR ótima para
cada inicialização. Este piloto não estabelece convergência nem inviabilidade
de scratch: se continuar melhorando, orçamento posterior deve ser definido
antes de concluir que a arquitetura/inicialização falhou.

Registrar separadamente perda temporal/VQ, perplexidade média por janela e códigos
usados na época. Uso agregado de códigos pode mascarar colapso por janela, por
isso não substitui perplexidade/curva de reconstrução. Diagnósticos robustos por
canal, amplitude/fase e piores grupos acompanham o seletor escalar.

## 4. Testes e integridade antes do lançamento

- **127 testes passaram** na suíte ampliada, além do smoke GPU descrito abaixo.
- Testes unitários: parâmetros/grids iguais, gradientes finitos, roundtrip de
  arquitetura/stride, rejeição de arquitetura desconhecida/divergente.
- Reservoir determinístico, sem updates no encoder; resume preserva política
  de diagnóstico e rejeita arquitetura divergente.
- Smoke real em GPU na maior janela de treino **512x1.000**:
  três forwards/backwards e updates descartáveis, **nenhum peso salvo/reutilizado**.
  Pico alocado ~**3,96 GB** em cada braço; shapes e contagens reais conferidos.
  Evidência: `results/experiments/architecture_initialization_smoke_20261010/`.
- Corpus train/val íntegro e presença das quatro famílias verificados.
- Protocolos fixam source, manifest, código e initializations por hashes; source
  e checkpoints históricos não são sobrescritos. Resume usa orçamento restante.

## 5. Execução e recuperação

Runner: `finetuning/run_architecture_initialization_experiment.py`.
Saída: `results/experiments/architecture_initialization_pilot_20261010/`.
Log de sequência: `results/experiments/architecture_initialization_pilot_20261010.pipeline.log`.
Logs/metrics por braço e `full_validation/` ficam dentro da saída.

Execução lançada em serviço de usuário **`febus-architecture-pilot-20261010`**,
sequencial na GPU. **Resultados de treino ainda pendentes nesta atualização**.
Estimativa: algumas horas; atualizar duração a partir das épocas medidas.

```bash
systemctl --user status febus-architecture-pilot-20261010

# Somente se não houver cópia ativa:
.venv/bin/python finetuning/run_architecture_initialization_experiment.py \
  --output-dir results/experiments/architecture_initialization_pilot_20261010 --resume
```

O serviço é independente do OpenCode interativo, não garantia de sobreviver a
logout/reboot. Resume repete a parte do epoch interrompido, restaura optimizer/RNG
e não acrescenta epochs ao orçamento. Initializations e fontes possuem ledger;
mudança de código/protocolo exige nova saída ou análise explícita, não resume silencioso.

## 6. Decisões após a triagem

1. Comparar inicial/selecionado/final e curvas; checar colapso/instabilidade antes
   de atribuir resultado à arquitetura ou falta de dados.
2. Confirmar os scores dos selecionados na validação completa, incluindo cauda
   multimode, bandas low e taxa pooled. Não promover por média isolada.
3. Scratch ainda em melhora: ensaio de orçamento/convergência e/ou LR controlado,
   preservando controle transferido; resultado fraco inicial não elimina a linha.
4. Arquitetura alternativa promissora: confirmar em outra seed/grupos e ampliar
   ablação de contexto sem alterar corpus e objetivo conjuntamente.
5. Nenhum ganho estrutural: investigar gargalos de quantização/residual e diversidade
   de aquisições com controles DCT. Não concluir automaticamente que novo dataset
   externo, MoS ou arquitetura maior resolverão a fidelidade.

Requisitos de aceitação física e novo holdout externo continuam necessários para
seleção final. Não foi realizado download de corpus novo nesta etapa.
