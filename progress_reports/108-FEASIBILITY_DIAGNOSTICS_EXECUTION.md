# Execução dos diagnósticos de viabilidade F1/F2

**17/11/2026.** Continuação autorizada dos
[relatórios 106](106-FEBUS_FEASIBILITY_AND_RESEARCH_REDIRECTION.md) e
[107](107-FEASIBILITY_CONTRACT_AND_EXECUTION_HANDOFF.md).
Este documento separa preparação, execução e interpretação. Não reabre H1–H5,
não libera reservas e não autoriza automaticamente treino F4.

## 1. Escopo congelado

- D1: painel F1 de 20 janelas/arquivos, três modelos congelados, mesmo núcleo
  físico, contextos 8/16/32/full possíveis, normalização nativa ou afim full fixo.
  Até 414 forwards no painel; teto científico 480. Zero updates.
- D2a: manifesto temporal por componentes do pool de 32 arquivos, coordenadas
  do bruto e continuidade, antes de qualquer leitura de valores FEBUS.
- D2b: coder marginal/contextual na mesma quantização e controles sintéticos.
  CDF inteira, roundtrip, bytes de instalação, causalidade e resets auditados.
- D2c: somente após preflights válidos; a impossibilidade de fit/check em C1/C4
  continua uma limitação explícita, não resolvida por dividir arquivos relacionados.
- Uma única GPU remota, braços sequenciais. F3 depende de sessões novas; seleção
  e treino F4 dependem da revisão dos resultados.

As tarefas de implementação foram delegadas ao GPT‑6.1 Sol em arquivos
separados. Revisão científica, verificação de pesos e lançamento remoto ficam
com o agente principal.

## 2. Preflight dos pesos F1

Inspeção remota em CPU, sem inferência:

| Modelo | Update | SHA-256 |
|---|---:|---|
| VQ madura | Export da época 23 | `b943db9f5cf6418e61ffca1814d788ac23fd6111468a425a918ba7d7cb5bc593` |
| Diversidade, dez configurações | 8.192 | `58b93e3f08163ffa1806358f1b9caf935da336be61d5802b76fb3d6785824a2d` |
| FiLM constante | 16.384 | `be74292a3a3072dfc05a904067c226ab3f61e4cce5e3a967124445e32225b2a7` |

Evidência: [configurações, paths e hashes inspecionados](../results/experiments/feasibility_context_preflight_20261117/weights_inspection.json).
As arquiteturas registradas são respectivamente `native-v1`,
`dense-full-tile-product-v1`/flat e `physical-film-v1`/constant.
Os modelos não serão substituídos por checkpoints escolhidos no painel.

O launcher [`run_feasibility_context_remote.py`](../tools/run_feasibility_context_remote.py)
preserva fontes remotas divergentes, verifica hashes, recusa uma GPU 0 com
processo compute ativo e executa com `CUDA_VISIBLE_DEVICES=0`. Não transfere
waveforms; traz somente os novos resultados do diagnóstico.

## 3. D2a — Manifesto temporal candidato, leitura bloqueada

Implementação: [`prepare_feasibility_temporal.py`](../tools/prepare_feasibility_temporal.py).
Evidência: [preflight temporal](../results/experiments/feasibility_temporal_preflight_20261117/temporal_preflight.json)
e [validação](../results/experiments/feasibility_temporal_preflight_20261117/validation.json).

- Plano candidato determinístico: 32 arquivos, um bloco/arquivo, primeiros
  5.000 samples por região retida contígua; **18.275.000 células**.
- Papéis candidatos: 20 arquivos fit, dez check, dois bloqueados em C1/C4.
  Nenhum papel é certificado como generalização entre sessões.
- Auditoria condicional sobre todos os 87 arquivos/486 blocos de treino,
  incluindo os 43 checks históricos: 345 interseções entre arquivos. Sob a
  hipótese histórica de conversão de relógio, nenhum arquivo adicional sai do pool.
- **Plano executável vazio:** não há evidência suficiente para certificar
  unidade/origem/referência do `Source/time`, relação com `Origin/Spacing` e
  colocação de `BlockOverlap=5`. Nenhum sinal foi aberto por esse preflight.

Consulta documental adicional em 17/11/2026:
[Xdas — formatos de dados](https://xdas.readthedocs.io/en/stable/user-guide/io/data-formats.html)
adverte explicitamente que a documentação das versões FEBUS é insuficiente
e recomenda fornecer trimming e posição do timestamp dentro de cada bloco.
O exemplo de offset no centro de um bloco é **um exemplo**, não evidência da
posição dos timestamps deste corpus. Não importamos os defaults de um loader
como certificado do nosso interrogador/exportação.

A documentação a1das da versão 2.3.5, localizada na busca pública, descreve
clocks absolutos por bloco e overlap percentual, mas não certifica por si só
o mapeamento temporal dos nossos arquivos de versão 2.3.24. O fetch direto
dessa página falhou; ela não é uma nova confirmação instrumental.

**Decisão:** D2b sintético pode avançar independentemente. D2c continua bloqueado
até confirmação da semântica pelo responsável pela aquisição, documentação da
versão/exportador ou evidência equivalente auditável. Não buscar alinhamento
pelo waveform de check para contornar a lacuna. Não promover o plano candidato
a executável nem usar as reservas como substituto.

## 4. D1 — Implementação e lançamento

Implementação: [`run_feasibility_context.py`](../finetuning/run_feasibility_context.py).
O runner valida identidades e os hashes históricos do painel, congela fontes
antes dos forwards, publica registros longos e compara replay full/native
contra full/afim fixo por bits. A taxa registrada é nominal/diagnóstica;
não é uma medição de codec completo.

A revisão corrigiu um detalhe de preparação: a preservação exata de zeros por
escape não pode zerar sua entrada normalizada. Um zero original geralmente
normaliza para `−média/escala`; conservar esse valor mantém a função de entrada
histórica. Há teste específico dessa propriedade, além de escapes e afim fixo.

**46 testes prospectivos aprovados** após a revisão: runner, planner, métricas,
congelamento e manifesto temporal. Antes dessas implementações, a suíte local
completa passou com 516 testes. Nenhum desses testes usa waveform FEBUS.

O lançamento remoto usa:

```bash
.venv/bin/python -m tools.run_feasibility_context_remote \
  --output results/experiments/feasibility_context_replay_20261117 \
  --weights-manifest results/experiments/feasibility_context_preflight_20261117/weights_inspection.json
```

**Execução concluída:** 414 forwards, 414 registros, zero updates, pesos
inalterados e 60 verificações full/native versus full/afim fixo bit-idênticas.
Todos os 20 inputs conferem com seus hashes históricos. Nenhuma leitura de
validação, teste ou EDAS. Resultado:
[replay F1](../results/experiments/feasibility_context_replay_20261117/results.json).

### 4.1 Resultado e interpretação científica

O sinal descritivo de sensibilidade predefinido passou para **os três modelos**,
inclusive com afim fixo. No par explícito **8→16 canais**, em normalização fixa,
os dois arquivos de oito configurações passam o limiar de 5% em diversidade
e FiLM; cinco configurações passam na VQ madura. Isso é sensibilidade de
contexto, não um gate de promoção.

A tabela abaixo resume, de modo **secundário/descritivo**, a comparação entre
o menor contexto viável e full. São 19 núcleos com contraste; a janela de dois
canais fica fora desta razão, mas continua nos resultados. Larguras full variam
entre arquivos. Percentuais negativos significam menos RMSE no contexto estreito.

| Modelo | Norma | Mediana de `RMSE_estreito/RMSE_full − 1` | Arquivos com menos/mais erro |
|---|---|---:|---:|
| VQ madura | Nativa por contexto | −6,25% | 11 / 8 |
| VQ madura | Afim full fixo | +85,53% | 4 / 15 |
| Diversidade 10 | Nativa por contexto | −59,87% | 19 / 0 |
| Diversidade 10 | Afim full fixo | −37,66% | 16 / 3 |
| FiLM constante | Nativa por contexto | −58,03% | 19 / 0 |
| FiLM constante | Afim full fixo | −38,64% | 15 / 4 |

Análise reproduzível:
[`summarize_feasibility_context.py`](../finetuning/summarize_feasibility_context.py)
e [resumo com comparações por arquivo e hashes](../results/experiments/feasibility_context_analysis_20261117/summary.json).
O script confere os snapshots das fontes e os invariantes da execução, sem
reler waveform. Os pares completos, incluindo regressões, estão em
`context_scores.json` no diretório de replay.

**O que muda na interpretação:**

1. A hipótese de incompatibilidade treino–inferência ganha evidência direta:
   nos dois modelos treinados em oito canais, o menor contexto melhora todos
   os 19 núcleos sob preparação nativa. Não era adequado usar apenas seu erro
   full-tile para concluir que faltava diversidade ou utilidade dos metadados.
2. **Não é só a escala afim:** a sensibilidade persiste sob afim fixo. Mistura
   espacial, estatísticas internas, bordas/padding e decisões de quantização
   continuam inseparáveis nesse diagnóstico. Não chamar isso de bug comprovado
   de GroupNorm, nem somar percentuais das duas condições como efeitos causais.
3. A VQ madura responde de outra forma: estreitar com afim full piora 15 dos
   19 núcleos e a mediana é muito desfavorável. “Usar oito canais” não é solução
   universal; o contrato aprendido e a preparação interagem.
4. Isso **não reverte os negativos históricos** de diversidade/FiLM. Seus
   desenhos e gates permanecem encerrados; a nova evidência limita a extrapolação
   deles para outros contratos de treino/inferência.
5. As reduções são no **mesmo núcleo central**, não no tile completo e não a
   bytes pareados. Em geometria densa, crops 8×1.000 têm 1,375 bits nominais de
   índices/célula, contra 0,6875 em tiles grandes divisíveis por 16. Overlaps,
   halos, normalização e framing ainda precisariam ser pagos. Não anunciar
   “60% de ganho do codec” nem “mesma fidelidade com metade dos dados”.

**Decisão atual:** antes de outra curva de dados ou backbone, a oportunidade
mais bem demonstrada é **compatibilidade do contrato de contexto/preparação**.
Ela justifica desenhar um único contraste controlado de compatibilidade,
mas não escolher automaticamente um novo modelo ou iniciar treino F4.
F2/F3 continuam limitados pelas pendências documentadas; a combinação de
F1 positivo com F2 ainda inconclusivo não resolve a seleção definitiva.

## 5. D2b — Coder e sintéticos

Implementados `compression/feasibility_tables.py`, `feasibility_stream.py` e
`feasibility_portable.py`, com runner
[`run_feasibility_synthetic.py`](../finetuning/run_feasibility_synthetic.py).
O backend é rANS próprio com CDF inteira de 20 bits, sem requantização float
oculta. Há decoder causal independente e objeto `.pyz` que inclui fonte do
decoder, instalação e stream. Python 3/biblioteca padrão são a plataforma
compartilhada declarada; o interpretador e o sistema operacional não são
empacotados.

[Preflight binário e fixtures](../results/experiments/feasibility_coder_preflight_20261117/protocol.json)
congelados antes da campanha. Foram aprovados 33 testes específicos de coder,
incluindo roundtrip de q/waveform por bits, corrupção, causalidade, resets e
especiais. A revisão do principal conferiu as regras de fit, suporte, CDF,
escape, contexto decodificado e contabilidade no código.

### 5.1 Resultado dos controles

**Campanha completa concluída em CPU:** cinco seeds, checks independentes,
duas precisões primárias, **50 casos, 150 streams e 52.428.800 células**
fit/check. Duração 604,6 s, abaixo do teto de duas horas; pico RSS de
237.105.152 bytes (226,1 MiB).

[Resultados completos](../results/experiments/feasibility_synthetic_20261117/results.json)
incluem incerteza Monte Carlo entre as cinco seeds, tempos, NLL float, CDF
inteira, bytes de transporte, instalação e objeto `.pyz` efetivamente produzido.
O principal conferiu os hashes dos **450 artefatos** de transporte, instalação
e objeto executável. Todos os roundtrips q/waveform foram bit-exatos;
**zero violações do bound físico**.

Economias médias do contextual temporal contra marginal:

| Controle | ε / sigma_cal | Economia CDF (bpc) | Economia transporte (bytes) | Economia objeto Python/stdlib (bytes) |
|---|---:|---:|---:|---:|
| Gaussiano i.i.d. | 0,03 | −0,0043 | 0 | −4.805,4 |
| Gaussiano i.i.d. | 0,1 | −0,0009 | 0 | −1.533,0 |
| AR(1), ρ=0,95 | 0,03 | 1,4953 | 195.989,0 | 192.751,0 |
| AR(1), ρ=0,95 | 0,1 | 1,4672 | 192.315,0 | 191.121,6 |
| Variância condicional | 0,03 | 0,4680 | 61.340,8 | 54.860,4 |
| Variância condicional | 0,1 | 0,4343 | 56.931,8 | 55.001,8 |

Valores positivos são economia; negativos, custo adicional. Cada objeto check
contém 1.048.576 células. O fallback eliminou regressão de transporte no i.i.d.,
mas não tornou gratuita a instalação contextual. Constantes e especiais também
usaram transporte sem economia e pagaram overhead contextual.

**Caso instrutivo:** na variância condicional a ε=0,03, o braço espaço-temporal
economizou 0,4125 bpc em CDF e 54.062 bytes de transporte, mas **aumentou o
objeto completo em 19.252,6 bytes**, em média, após instalação. Os canais
sintéticos são independentes entre si; acrescentar contexto espacial não é
dependência nova e pode diluir contagens/elevar o custo das tabelas.

**Decisão metodológica:** o probe distingue ausência de dependência contextual
no controle i.i.d. e dependência aproveitável em AR/variância com média zero.
Assim, os testes não estão restritos à previsibilidade da média ou lag1.
Eles não estabelecem sensibilidade a toda dependência FEBUS, e não medem a
degradação em diferentes comprimentos de reset; somente o reset fixo de 1.000
samples foi executado. Não inferir capacidade ou throughput de uma futura rede
a partir deste rANS diagnóstico em Python.

**D2b encerrado com validação sintética positiva. D2c permanece bloqueado**
pelo contrato temporal, independentemente desse sucesso. Não houve score F2
FEBUS, liberação de reservas, seleção definitiva de candidata ou treino F4.

## 6. Verificações consolidadas

- **559 testes locais aprovados** na suíte completa após as implementações.
  Avisos NumPy de correlação em constantes permanecem avisos dos testes
  históricos; não houve falhas.
- `git diff --check` limpo.
- Relatórios 1–106 e inputs congelados conferidos por SHA-256, sem alterações.
  Sequência documental 1–108; 194 links locais acessíveis nos documentos
  atualizados. Permanecem indisponíveis somente as duas referências históricas
  ao arquivo da consolidação já ausente nesta cópia local.
- Sem commit, push ou descarte de trabalho anterior. As fontes necessárias ao
  F1 foram sincronizadas com backup de divergências e conferência de hash;
  o resultado remoto concluído foi recuperado localmente.

## 7. Próxima decisão

F1 fornece evidência direta de sensibilidade de contexto/preparação nos modelos
testados; F2 tem diagnóstico sintético funcional, mas não evidência FEBUS de
economia condicional. Portanto, a prioridade continua compatibilidade de
contrato, **sem declarar superioridade de F4-B sobre uma oportunidade F4-A
ainda não medida**.

Para avançar D2c, obter documentação ou confirmação auditável da posição/unidade
dos clocks e overlap da versão/exportador deste acervo, revalidando o manifesto
antes de ler sinais. F3 continua dependendo de novas sessões. Um eventual
contraste de treino de compatibilidade precisa de protocolo próprio com
controle, geometria, dose e bytes completos; o resultado central deste replay
não o executa nem garante seu sucesso.
