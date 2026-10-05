# Reconstrução generalista FEBUS: protocolo e próximos passos

## Objetivo e premissa

O objetivo é reconstruir fielmente **os dados FEBUS do Drive em todas as famílias**,
não apenas maximizar uma média dominada por multicore ou selecionar exemplos favoráveis.
Reconstrução não é denoising: remover ruído presente no original não significa aumentar
a fidelidade do codec. Não converter Strain em Strain Rate para misturar alvos.

O checkpoint VQ-VAE original de `smartgrids` foi treinado com dados de origem OOI.
O artigo `smartgrids/bibliography_copy/artigo_principal/artigo_principal_extraido.txt`
descreve aquisição offshore em 200 Hz, espaçamento de aproximadamente 10,2 m e
gauge length de aproximadamente 40,9 m, em condições físicas diferentes das FEBUS.
O artigo **não é a especificação da arquitetura VQ-VAE deste projeto**; suas
conclusões de denoising/detecção não demonstram fidelidade de compressão FEBUS.
Transferência de pesos é uma hipótese econômica, não um requisito do modelo final.
Treinar do zero e alterar a arquitetura são opções legítimas, posteriores por custo.

## Experimento de balanceamento implementado — 2026-10-05

- `window`: cada janela uma vez por época, embaralhada (baseline).
- `group`: família/configuração uniforme, depois janela uniforme, com reposição.
- `family-group`: família uniforme, configuração uniforme dentro da família,
  janela uniforme dentro da configuração, com reposição. Esta é a variante testada.
- Época completa: o mesmo número de atualizações do treino uniforme, não mais
  atualizações artificialmente para o balanceado. Janelas podem repetir ou não ser
  sorteadas na época balanceada. Isso muda exposição e pode aumentar overfitting
  das configurações pequenas; acompanhar validação por grupo.
- Somente janelas de treino entram no sampler. Splits por configuração preservados.
- Ambos os braços partem do checkpoint-base, LR `1e-5`, AdamW, seed 2026,
  mesma perda, orçamento de épocas e early stopping. Saídas separadas.
- Seleção ainda por menor **val_loss ponderada por janela**, igual para ambos
  os braços. Não mudar simultaneamente amostrador e critério de seleção.

### Testes concluídos

Seis testes unitários passaram: cobertura uniforme, equilíbrio de famílias/grupos,
distinção entre modos, determinismo, rejeição de janelas de teste no sampler e
agregação macro sem dar peso extra a configurações com mais janelas.
Compilação Python e expansão dos novos parâmetros Make também verificadas.

Piloto pareado: 4 folds × 2 braços × 2 épocas × 128 atualizações. Cada braço usa
o mesmo subconjunto determinístico de 128 janelas de validação para selecionar
checkpoint. Avaliação posterior: 16 janelas por configuração **da validação**,
sem consultar testes para decidir ajustes no piloto.

Artefatos: `results/experiments/sampling_pilot_20261005/` (concluído).

| Fold | Corr. macro uniforme | Balanceado | RMSE macro uniforme | Balanceado |
|---|---:|---:|---:|---:|
| 0 | 0,1932 | 0,2017 | 0,9778 | 0,9760 |
| 1 | 0,1849 | 0,1969 | 0,9788 | 0,9756 |
| 2 | 0,1950 | 0,1932 | 0,9760 | 0,9762 |
| 3 | 0,1763 | 0,1763 | 0,9827 | 0,9826 |

Macro significa média de configurações dentro de cada família e depois média das
famílias presentes. Não confundir com média por janela dos resultados históricos.

| Família | Corr. uniforme | Balanceado | RMSE uniforme | Balanceado |
|---|---:|---:|---:|---:|
| Doped | 0,1633 | 0,1744 | 0,9860 | 0,9843 |
| Multicore | 0,2177 | 0,2152 | 0,9737 | 0,9742 |
| Multimode | 0,1500 | 0,1596 | 0,9886 | 0,9870 |
| Single-mode | 0,2003 | 0,2072 | 0,9722 | 0,9693 |

**Conclusão preliminar:** ganho macro em 2/4 folds e pequena regressão em multicore.
Não é consistente o suficiente para alterar o padrão ou treinar o modelo final.
O orçamento curto é smoke test de implementação, não comparação conclusiva de
generalização. Não comparar seus números diretamente ao fine-tuning de 10 épocas.

### Comparação completa iniciada (ainda sem conclusão)

Comando executado em segundo plano:

```bash
.venv/bin/python finetuning/run_sampling_experiment.py \
  --output-dir results/experiments/sampling_full_20261005
```
OBS: adicione --resume .../last.pt para continuar a partir de um checkpoint específico;

Oito treinos sequenciais na GPU, até 10 épocas por braço/fold, todas as janelas de
validação e teste. Checkpoints anteriores em `results/fine_tune_results/` intactos.
Baseline novo evita o critério antigo de salvamento de `best.pt` do fold 0.

Logs: `results/experiments/sampling_full_20261005.launch.log` e
`results/experiments/sampling_full_20261005/fold_N/{window,family-group,evaluation}.log`.
Após término bem-sucedido: `REPORT.md`, `comparison.json` e marcador `COMPLETE`.
Ausência de `COMPLETE` significa execução em andamento ou falha; verificar logs.
O runner recusa diretório preexistente para evitar sobrescrita acidental.

Pode levar da ordem de um dia nesta RTX 4060 (estimativa, não garantia).
O relatório automático resume métricas; **não promove automaticamente** um modelo.

## Critérios de decisão generalista

1. Reportar macro por família/configuração e média por janela lado a lado.
2. Verificar ganho pareado entre folds, famílias com regressão e piores configurações.
   Fold 3 não contém doped/multimode no teste: não inferir melhoria dessas famílias
   a partir dele. Há poucos grupos em algumas famílias, limitando conclusões.
3. Uma semente não demonstra robustez; repetir candidatos promissores com outras
   sementes e reportar dispersão. Janelas do mesmo arquivo não são observações
   independentes para intervalos de confiança: usar configurações/arquivos como clusters.
4. Selecionar hiperparâmetros na validação. Os testes dos folds já foram consultados:
   próximos resultados são exploratórios, não avaliação externa intocada.
5. Antes de implantação, reservar nova aquisição/arquivos/configurações não usados
   na seleção como auditoria final; não fazer split aleatório entre janelas do mesmo HDF5.
6. Definir tolerâncias de aceitação para correlação, RMSE, erro espectral e pior
   família conforme uso pretendido. Por enquanto **não há limiar de qualidade suficiente
   acordado**; melhoria relativa não prova que o modelo é suficientemente bom.

## Sequência até o treinamento final (condicional, não automática)

1. **Terminar comparação de amostragem.** Se houver ganhos sem regressões importantes,
   manter balanceamento; se o trade-off persistir, testar mistura parcial entre pesos
   uniformes por janela e hierárquicos, escolhendo a mistura só na validação.
2. **Diagnóstico do gargalo e métrica generalista.** Tabelas por família, configuração,
   frequência nominal, núcleo/canal, variável, `dt_s`, `dx_m`; verificar amplitudes,
   fase, RMS espacial e erro de potência/espectro. Frequência nominal do experimento
   não é sample rate: usar `dt_s`. Comparar seleção por val_loss atual com RMSE macro
   por família/configuração, sem escolher o checkpoint no teste.
3. **Adaptação leve e objetivo espectral.** Comparar decoder-only, encoder congelado
   e modelo inteiro; LR após plateau; perda espectral multirresolução junto a MSE/L1.
   Medir ocupação/perplexidade do codebook e erro antes/depois da quantização para
   separar gargalos de encoder, codebook e decoder. Alterar um componente por vez.
   Atenção: atualmente `--resume` restaura LR do estado AdamW; mudar apenas
   `FINETUNE_LR` não reduz efetivamente a LR retomada. Corrigir isso antes desse teste.
4. **Geometria física.** Testar estratégias multiescala para duração temporal,
   espaçamento e número de canais; não destruir bandas FEBUS relevantes para forçar
   dados ao domínio OOI. Usar a mesma representação para comparação justa.
5. **Ensemble / MoS (mixture of specialists).** Se persistirem trade-offs entre
   famílias, comparar um generalista com especialistas por fibra e fallback generalista.
   Primeiro roteamento por metadados conhecidos (família, variável e geometria),
   nunca seleção oracle pelo erro contra o original no teste. Especialistas de cada
   fold treinam somente nos grupos de treino daquele fold e são selecionados só na
   validação. Algumas famílias têm poucas configurações: maior risco de overfitting.
   Se usar gating aprendido, treiná-lo apenas no treino/validação; testar recortes
   desconhecidos, metadados ausentes e robustez do fallback. Média simples entre
   reconstruções é baseline de ensemble, não garantia de ganho: pode suavizar eventos.
   Comparar latência, memória, custo de modelos e overhead do roteamento/codec,
   além de fidelidade. O sistema completo, não cada especialista isoladamente, deve
   cobrir todas as famílias. Especialização não implica transferência para fibra inédita.
6. **Diagnosticar cobertura de dados e buscar datasets próximos de FEBUS.** Se os
   resultados continuarem insuficientes, executar o protocolo da seção abaixo.
   Havendo evidência de limitação de diversidade/volume efetivo, priorizar ampliar
   o corpus antes do treinamento longo de um novo modelo. A busca pode ocorrer em
   paralelo ao diagnóstico; não exige esgotar todas as variantes arquiteturais.
7. **Treino do zero / arquitetura FEBUS.** Se os melhores candidatos ainda falharem
   na pior família, espectro ou critérios de aceitação, comparar inicialização aleatória
   da arquitetura atual contra transferência com orçamento suficiente. Depois testar
   strides menos agressivos, capacidade/codebook, quantização residual ou multiescala
   e condicionamento por geometria. Ganho de capacidade tem custo: avaliar fidelidade
   na mesma taxa de compressão ou mostrar curva rate-distortion. Não insistir em
   pesos OOI como restrição se isso impedir qualidade FEBUS.
8. **Modelo final em todos os dados de desenvolvimento.** Só após escolher protocolo
   e orçamento pela validação cruzada. Usar época/orçamento definido antecipadamente,
   ou manter validação por configuração; não há early stopping honesto sem validação.
   Auditoria externa permanece fora do treino. Se optar por todos os dados disponíveis
   do Drive, esse checkpoint é de produção e a qualidade nesses mesmos dados não é
   evidência de generalização. Registrar normalização, geometria, roteamento,
   taxa de compressão e limites conhecidos com os pesos finais.

## Hipótese de escassez de dados e ampliação do corpus

### Por que é plausível, mas não inevitável

Segundo o volume informado pelo usuário, o modelo original utilizou aproximadamente
500 GB de HDF5 crus antes do processamento, bem mais que o corpus FEBUS atual.
Essa diferença torna plausível uma limitação de cobertura, mas **não diagnostica
sozinha falta de dados**. O volume original não foi auditado nesta etapa.

Bytes dependem de precisão, compressão HDF5, quantidade de canais e sample rate.
Horas contíguas ou janelas sobrepostas podem ser muito redundantes. Os indicadores
mais relevantes são duração útil, janelas efetivamente usadas após processamento,
aquisições/configurações independentes e diversidade por família, variável, geometria,
regime espectral, amplitudes e condições de ruído. Mais dados OOI não necessariamente
ensinam a reconstruir FEBUS: proximidade ao domínio-alvo também importa.

Outras causas possíveis de resultados insuficientes são otimização incompleta,
normalização, perda inadequada, stride/capacidade, gargalo de quantização ou uma taxa
de compressão incompatível com a fidelidade desejada. Poucos dados também podem
ser suficientes para adaptar uma boa representação; isso precisa ser medido.

### Diagnóstico antes de atribuir a falha ao tamanho do corpus

1. **Auditar cobertura e exposição.** Quantificar arquivos, duração física, grupos
   independentes e janelas únicas realmente usadas no treino, por família/configuração.
   Auditar seleção temporal/espacial: o preparo atual recorta até 512 canais centrais
   e usa janelas de 1.000 amostras. Verificar se aproveitar melhor os próprios HDF5
   amplia cobertura útil antes de procurar novos dados. Janelas adicionais do mesmo
   arquivo não equivalem a novas aquisições independentes.
2. **Curvas de aprendizagem por grupos.** Com validação fixa, treinar com subconjuntos
   crescentes de arquivos/configurações exclusivamente do treino, mantendo cobertura
   de famílias quando possível. Não reduzir dados sorteando apenas janelas altamente
   correlacionadas. Comparar orçamento de atualizações controlado e convergência,
   com múltiplas sementes; caso contrário confunde-se quantidade de dados com treino
   mais longo. Reportar curvas por família e pior configuração, não só agregado.
3. **Separar generalização de capacidade.** Bom ajuste no treino com validação ruim
   sugere overfitting/cobertura ou mudança de domínio, mas não prova escassez.
   Erro alto tanto no treino quanto na validação pede verificar otimização e gargalos.
   Um teste de overfit em poucas janelas pode revelar incapacidade de representação
   ou bugs; seu sucesso não demonstra generalização. Inspecionar também quantização
   e fidelidade na taxa de compressão escolhida.
4. **Evidência de ampliação útil.** Ganhos recorrentes na validação com mais grupos
   independentes, curvas ainda ascendentes e lacunas claras nas famílias fracas
   justificam expansão do corpus. Platô não prova que dados extras são inúteis:
   pode ser necessário obter condições diferentes, não mais do mesmo sinal.

### Etapa eventual de busca e avaliação de datasets DAS

- Procurar artigos e repositórios de dados DAS com aquisições **FEBUS ou fisicamente
  próximas**: experimentos de laboratório/terrestres/acústicos, diferentes tipos de
  fibra e condições que preencham lacunas do Drive. Priorizar proximidade medida,
  não apenas uso da sigla DAS nem tamanho bruto. Nenhum dataset externo foi ainda
  selecionado ou baixado nesta etapa.
- Criar inventário dos candidatos com DOI/URL, licença, acesso, volume, duração,
  fabricante/interrogador, fibra, variável e unidade, sample rate, `dx`, gauge length,
  geometria, bandas, ruído e natureza dos sinais. Registrar metadados desconhecidos;
  não assumir compatibilidade. Gauge length e resposta instrumental podem diferir
  mesmo quando sample rate e espaçamento parecem semelhantes.
- Preferir dados crus com rastreabilidade. Distinguir strain de strain rate, original
  de filtrado/denoised e amostras da mesma aquisição. Não remover componentes do
  alvo para tornar a reconstrução artificialmente fácil; denoising não é o objetivo.
- Testar primeiro um subconjunto pequeno e representativo: processamento correto,
  finitude, espectros e compatibilidade geométrica. Resampling exige antialiasing
  e preservação das bandas relevantes; uma harmonização que destrua informação
  FEBUS não é aceitável. Manter normalização e metadados rastreáveis por domínio.
- Fazer ablação pareada: **FEBUS somente** versus **externos + FEBUS**, com o mesmo
  protocolo de validação FEBUS, arquitetura e orçamento explícito. Comparar mistura
  balanceada com pré-treino externo seguido de adaptação FEBUS. Evitar que datasets
  externos maiores dominem atualizações; medir transferência negativa em cada família.
  Usar pesos transferidos inicialmente se for econômico, e comparar treino do zero
  do novo corpus quando justificado, sem mudar tudo simultaneamente.
- Splits externos por aquisição/local/configuração, nunca janelas aleatórias do mesmo
  arquivo. Auditar duplicatas entre fontes. Testes/auditoria FEBUS continuam fora do
  treino e da escolha de datasets/pesos de mistura. Avaliação externa é secundária:
  o critério principal permanece fidelidade **FEBUS**, inclusive nas piores famílias.
- Só escalar downloads e treinamento quando o piloto mostrar ganho útil na validação
  FEBUS, ou quando houver justificativa explícita para investigar uma lacuna ainda
  não coberta. Registrar custos de armazenamento/treino, resultados negativos e quais
  famílias beneficiaram. Mais dados podem ajudar, mas não garantem generalização.

**Consequência para a sequência:** se escassez de cobertura for sustentada pelo
diagnóstico, a ampliação do corpus passa a anteceder o treinamento longo do zero e
o modelo final. Se o gargalo for capacidade/quantização, investigar arquitetura em
vez de assumir que centenas de GB adicionais resolverão o problema. As duas
limitações podem coexistir; nesse caso comparar dados e arquitetura por ablações.

**Nenhum treino final, especialista ou treino do zero foi iniciado nesta etapa.**
