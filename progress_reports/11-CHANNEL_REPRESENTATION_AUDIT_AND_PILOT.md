# Representação por canal: auditoria reversível e piloto controlado

> Os resultados do treinamento pareado e o protocolo de validação ampliada estão no
> [relatório 13](13-CHANNEL_PILOT_RESULTS_AND_EXPANDED_VALIDATION.md).

## 1. Motivação e escopo

O [relatório 10](10-LONG_PRESERVATION_RESULTS.md) mostrou ganho pequeno da adaptação
longa, forte atenuação e multimode ainda difícil. Investigar se o z-score global
favorece canais dominantes e permite erro elevado em canais de pequena amplitude.
Não remover canais nem declarar componentes como ruído inválido para facilitar o alvo.

Primeiro foi executada auditoria **sem updates** com o full epoch 6 congelado:
duas janelas por arquivo de treino e validação do fold 1, seed 2026,
**210 treino + 102 validação**, quatro famílias. Nenhum forward de teste.
Artefatos: `results/experiments/channel_representation_audit_20261008/`.

## 2. Normalização experimental e custo explícito

`channel-escape-v1`, formato de side stream `FNC1`:

- Escapes acima de abs>1e20 preservados exatamente, mesma região ordinária do baseline.
- Média/desvio por canal sobre células ordinárias, parâmetros transmitidos em float32.
- Normalização usa os próprios parâmetros transmitidos; constante/all-escaped tratado
  explicitamente. Nenhum clipping ou mutação dos dados originais.
- Header de 24 bytes + **8 bytes por canal** + 8 bytes por escape.
- Apenas escapes são lossless; ordinary roundtrip tem arredondamento. Erro próximo
  de zero deve ser medido em relação à escala do canal, não relativo ao valor de
  cada célula perto de zero. Isso foi explicitado nos testes.

Na auditoria, máximo RMSE relativo comum do roundtrip: **3,048×10⁻⁸**.
Sem escapes nas janelas selecionadas; testes sintéticos incluem extremos, canais
constantes e totalmente escapados. A preservação física dos valores não é prova
de que a rede treinada reconstrói bem a nova representação.

O overhead não é desprezível: ~**0,064–0,070 bit/célula** por canal versus
~0,001–0,010 do baseline global nos grupos de validação. Com índices VQ fixos e
esses side bytes, razões nominais médias de ~**543–718× global** caem para
~**268–296× por canal**. Não são taxas de codec completo: container, mapas espaciais,
codificação entrópica e demais custos ficam fora dessa aproximação.

## 3. Inferência congelada: não basta trocar o loader

RMSE relativo na **mesma região ordinária**, depois de restaurar unidades originais,
com denominador comum RMS do original menos sua média global:

| Família, validação | Global | Por canal, sem retreino |
|---|---:|---:|
| Doped | 0,835126 | 0,915385 |
| Multicore | 0,924954 | 0,948904 |
| Multimode | 0,965741 | 1,000417 |
| Single-mode | 0,751229 | 0,865907 |
| **Macro** | **0,869262** | **0,932653** |

Troca apenas na inferência piorou todas as médias raw. A rede aprendeu o input
global: esta é mudança de domínio da entrada, **não comparação justa de dois modelos
treinados com normalizações distintas**. Não promover essa troca no loader antigo.

### Canais de baixa variância

O erro relativo médio por canal no multimode global foi muito alto (~57.777),
sensível a denominadores pequenos. Não confundir esse número com RMSE global
~0,966 ou interpretar diretamente como dificuldade uniforme de todos os canais.

Exemplo auditado de Configuration 7 com 256 linhas preparadas:
std temporal mediano ~**1.774**, mas últimas duas linhas têm std ~**0,00096/0,00167**.
Predições de pequena magnitude na escala global podem ser enormes relativas ao
alvo nessas linhas. A normalização por canal reduz esse efeito em inferência,
mas piora o erro global e não melhora universalmente as correlações.

Não houve exclusão de canais de baixa variância no score global. A métrica média
relativa por canal usa std>1e-6 como proteção diagnóstica, exige interpretação de
escala, distribuição e erro absoluto. Sua sensibilidade é motivo para definir
limiares físicos de aceitação, não para otimizar só esse agregado.
Exemplo registrado em `low_variance_example.json`; a origem instrumental/física
dessas linhas não foi estabelecida.

## 4. Próximo teste iniciado em background

Runner: `finetuning/run_channel_experiment.py`.
Saída: `results/experiments/channel_representation_pilot_20261008/`.
Log externo: `results/experiments/channel_representation_pilot_20261008.log`.
Resultados de treino pendentes nesta atualização; não há promoção automática.

| Item | Protocolo |
|---|---|
| Fonte comum | full best_macro epoch 6 do estudo de preservação |
| Braços | global escape-v1 / channel-escape-v1 |
| Treino | mesmo fold 1, seed 2026, sampler uniforme, modelo inteiro ajustável |
| AdamW | novo em ambos, LR 1e-5, weight decay 1e-5 |
| Orçamento | **6×2.048 updates por braço**, máximo total 24.576 |
| Validação | duas janelas por arquivo, 102 janelas, quatro famílias |
| Seleção | RMSE macro em alvo comum após restauração, inicialização candidata |
| Parada | patience 7, sem parar antes do orçamento pareado de seis epochs |
| Taxa | side bits/célula e razão nominal índices+side por grupo/epoch |
| Teste | não consultado |

Essa LR difere do estudo anterior para permitir adaptação à representação; o novo
braço global controla LR/optimizer/orçamento. Não atribuir diferenças versus o treino
longo exclusivamente à normalização. A política por canal muda **input e ponderação
efetiva dos erros no treino**; o experimento compara o conjunto dessa representação,
não isola somente centralização ou somente o objetivo.

### Evitar seletores incompatíveis

`--physical-validation` restaura a predição de cada braço e compara com o **mesmo
alvo global-normalizado**, excluindo só os mesmos escapes do score ordinário.
Assim `macro_relative_rmse` pode selecionar/comparar braços sem trocar o denominador
pela escala por canal. Métricas espectrais registradas também usam esse alvo comum;
loss temporal/VQ/objective continuam sendo da representação de treino e não são
substitutos para o seletor comum.

Normalização por canal exige essa validação; resume rejeita mudança de política ou
de modo de validação. As inicializações têm os mesmos pesos, mas diferentes entradas,
portanto seus scores **não precisam ser iguais**. O comparador mantém igualdade da
seleção de janelas, mas admite a diferença inicial declarada no protocolo.

## 5. Operação, análise e limites

Piloto deve durar dezenas de minutos, sujeito a I/O e validação adicional; estimativa,
não ETA medida. Space preflight exige 5 GiB; saves/locks/resume preservam arquivos
anteriores. Background OpenCode depende do serviço; retomada em terminal independente:

```bash
.venv/bin/python finetuning/run_channel_experiment.py \
  --output-dir results/experiments/channel_representation_pilot_20261008 --resume
```

Ao concluir:

```bash
.venv/bin/python finetuning/summarize_numeric_adaptation.py \
  --experiment-dir results/experiments/channel_representation_pilot_20261008
```

Avaliar melhoria raw contra o braço global e não apenas contra a inicialização
channel mais fraca; verificar famílias/configurações com regressão, amplitude/fase,
erros por canal e overhead. Exigir ganho que justifique custo e confirmar com
validação completa antes de escalar. Resultado negativo justifica investigar
geometria/contexto ou arquitetura/taxa, não declarar a normalização inútil em todo cenário.

São comparações exploratórias em splits já usados para decisões. Nova aquisição/
configuração holdout permanece necessária para auditoria final. **79 testes passaram**
antes do treino: formatos, roundtrip, máscara, alvo comum, cobrança de side bytes,
integração sintética de treino e controles históricos. Dados/checkpoints originais
preservados; nenhuma mudança no roadmap para registrar estado momentâneo.
