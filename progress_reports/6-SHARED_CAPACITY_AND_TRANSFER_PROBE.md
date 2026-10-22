# Capacidade compartilhada e transferência entre configurações FEBUS

Data: 2026-10-07. Artefatos: `results/experiments/shared_capacity_probe_20261007/`.

## Decisão e conclusão

**Não promover os modelos adaptados deste teste.** Os dois braços reduziram bastante
o erro no pool de oito janelas, mas pioraram em controles não usados na ramificação
e na validação por configuração. O melhor checkpoint de validação foi o inicial
(passo 0), tanto no VQ quanto no contínuo.

O teste evidencia overfitting e perda de desempenho em dados fora do pool para
este protocolo deliberadamente pequeno/agressivo. **Não prova que o corpus completo
FEBUS seja insuficiente**, nem que remover quantização ou simplesmente aumentar
tempo de treino resolveria generalização. A reconstrução multimode permanece difícil
quando a capacidade é compartilhada, mesmo no pool.

Prioridade revisada: auditar cobertura e exposição úteis, preservar representação
na adaptação (LR/regularização/fine-tuning parcial) e medir interferência entre
famílias antes de grandes downloads, treino do zero ou MoS de produção. Limites
de taxa/arquitetura continuam hipóteses em aberto.

## 1. Protocolo e integridade

- Fonte comum: best uniforme do fold 1, `sampling_full_20261005/fold_1/window/best.pt`.
- Um modelo compartilhado por braço: VQ versus AE contínuo realmente treinado.
  Encoder/decoder começam nos mesmos pesos, quantizador não participa do contínuo.
  Mesma geometria/padding/crop e normalização global por janela.
- Pool: oito janelas de oito arquivos do treino do fold 1, duas por família.
  Preserva as quatro âncoras do relatório 6; acrescenta outra configuração/arquivo
  quando disponível, escolhidos somente por metadados, seed 2026.
- Controles: uma janela adicional por arquivo do pool, preferindo outro bloco,
  fora do pool. São **controles da ramificação**, não dados nunca vistos pelo
  checkpoint de origem: pertencem ao treino original e podem ter sido usados nele.
  Mesma aquisição implica correlação/dependência, não auditoria externa.
- Validação: duas janelas por arquivo do fold de validação 2, total 102 janelas de
  51 arquivos, cobrindo quatro famílias e seis configurações. Não há grupo nem
  arquivo compartilhado com pool/controles. Teste fold 1 não foi avaliado.
- AdamW novo, LR `1e-4`, weight decay 0, grad clip 1; objetivo temporal
  `0,5*MSE + 0,5*L1`, somando VQ somente no braço quantizado.
- 4.000 atualizações por modelo, 8.000 totais. Embaralhamento determinístico de ciclos
  completos das oito janelas: exatamente 500 exposições por janela, sem augmentation.
  Mesma ordem e orçamento nos braços. Um update não corresponde a uma época completa
  do corpus; é uma apresentação de uma janela do pool.
- Medições no início e a cada 500 passos: macro por configuração dentro da família,
  seguido de média das famílias. Checkpoints separados por melhor treino e validação.
- Uma seed e um fold. Não é validação cruzada completa, seleção final de arquitetura
  ou teste de significância. LR alta, oito janelas e ausência de weight decay foram
  escolhidas para diagnóstico de ajuste, não como recomendação de produção.

### Pool utilizado

| Família | Configuração | Arquivo | Canais |
|---|---|---|---:|
| Doped | 5m i29 | Copy of 1000Hz.h5 | 45 |
| Doped | 5m i29 | Copy of 100Hz.h5 | 45 |
| Multicore | C2 and C4 | Copy of 100Hz_CH1.h5 | 70 |
| Multicore | C1 and C2 | Copy of 50Hz_CH1.h5 | 70 |
| Multimode | Configuration 5 | Copy of _2025-12-08_09-15-11_UTC.h5 | 512 |
| Multimode | Configuration 5 | Copy of _2025-12-08_09-15-41_UTC.h5 | 512 |
| Single-mode | Configuration 9 | Copy of 5000Hz.h5 | 23 |
| Single-mode | Configuration 12 | Copy of 10000Hz.h5 | 36 |

IDs, blocos e offsets constam nos CSVs de seleção/protocolo. As janelas têm 1.000
amostras a 20 kHz, duração 0,05 s; títulos representam frequência nominal, não sample rate.

## 2. Resultado global e seleção de checkpoint

| Modelo | Estado | Passo | RMSE macro pool | Controles | Validação |
|---|---|---:|---:|---:|---:|
| VQ | Inicial | 0 | 0,8817 | 0,8779 | 0,8714 |
| VQ | Melhor treino / final | 4000 | 0,4975 | 1,0746 | 1,0781 |
| VQ | Melhor validação | 0 | 0,8817 | 0,8779 | 0,8714 |
| Contínuo | Inicial | 0 | 0,8783 | 0,8784 | 0,8809 |
| Contínuo | Melhor treino / final | 4000 | 0,4935 | 1,0790 | 1,1703 |
| Contínuo | Melhor validação | 0 | 0,8783 | 0,8784 | 0,8809 |

Pool, controles e validação têm distribuições diferentes; comparar principalmente
**antes/depois dentro de cada split**, não apenas subtrair valores absolutos entre
eles. Piora em dados fora do pool surge já na primeira medição pós-treino (500).
O passo 0 foi preservado como candidato e evita substituir pesos bons por adaptações
piores. O melhor treino ainda foi o último ponto: capacidade/convergência no pool
não ficou completamente caracterizada, especialmente em multimode.

## 3. Famílias: treino versus validação

| Família | RMSE pool VQ inicial → final | Contínuo inicial → final | RMSE validação VQ inicial → final | Contínuo inicial → final |
|---|---:|---:|---:|---:|
| Doped | 0,7502 → 0,4067 | 0,7377 → 0,4217 | 0,8365 → 1,0535 | 0,8292 → 1,1608 |
| Multicore | 0,9311 → 0,4287 | 0,9264 → 0,4143 | 0,9260 → 1,1079 | 0,9217 → 1,2154 |
| Multimode | 0,9896 → 0,9398 | 0,9869 → 0,9295 | 0,9685 → 1,0874 | 1,0221 → 1,2441 |
| Single-mode | 0,8561 → 0,2147 | 0,8621 → 0,2086 | 0,7545 → 1,0635 | 0,7503 → 1,0607 |

Todas as famílias pioraram na validação. Remover o quantizador não foi suficiente:
o comportamento de memorização/transferência persiste no controle contínuo, que tem
outro orçamento de informação e não inclui regularização VQ.

Nos controles, o VQ piorou RMSE em doped, multicore e single-mode; multimode melhorou
de 0,9892 para 0,9550. No contínuo, padrão semelhante (multimode 0,9867 → 0,9452).
Isso mostra efeitos diferentes por família, mas não constitui transferência robusta:
os controles são dos mesmos arquivos e a validação multimode piorou.

### Mesmas âncoras do teste individual

Ambas as experiências dão 500 apresentações do alvo, mas no compartilhado estas
alternam com sete outros alvos. Igual número de exposições não significa mesma
trajetória de gradiente nem convergência.

| Âncora | RMSE VQ individual / compartilhado | Contínuo individual / compartilhado |
|---|---:|---:|
| Doped | 0,2588 / 0,3867 | 0,2689 / 0,3989 |
| Multicore | 0,2298 / 0,3395 | 0,2272 / 0,3214 |
| Multimode | 0,6701 / 0,9648 | 0,5527 / 0,9550 |
| Single-mode | 0,2148 / 0,2134 | 0,2003 / 0,2024 |

O caso multimode que era melhor memorizado isoladamente passa a ser mal ajustado
no modelo compartilhado, inclusive no contínuo. É compatível com interferência/
capacidade compartilhada/otimização, **não uma ablação causal que prove competição
entre famílias**: também acrescentamos uma segunda janela da mesma família e
alteramos a sequência de updates. Comparar futuramente treino somente multimode,
com as mesmas duas janelas, ao treino misto controlará melhor essa hipótese.

## 4. Interpretação e próximo passo

1. **Não aumentar cegamente as repetições deste pool nem promover o checkpoint final.**
   Treino melhora enquanto transferência degrada. A inicialização permanece referência.
2. **Auditar cobertura/exposição e conteúdo dos dados atuais** antes de buscar dezenas
   ou centenas de GB. Quantificar grupos/arquivos independentes, duração útil e
   redundância das janelas; examinar energia espectral e heterogeneidade por canal,
   particularmente em multimode. Mais janelas do mesmo sinal não equivalem a
   diversidade. Não chamar de ruído descartável componentes do alvo de reconstrução.
3. **Teste controlado de compartilhamento/fine-tuning parcial:** mesmas janelas
   multimode isoladas versus modelo misto, orçamento por alvo igual; depois decoder
   ajustável com encoder/codebook congelados ou LR conservadora/replay mais amplo.
   Medir validação e pior família, mantendo passo 0 e sem escolher pelo teste.
   LR/regularização do diagnóstico não provaram que a adaptação generalista fracassa.
4. **MoS merece ablação, não promoção:** comparar especialistas por família com o
   generalista em protocolo por configuração, dados e orçamento controlados, incluindo
   roteamento por metadados e fallback. Modelos por janela do relatório 6 não são MoS.
5. **Dados externos e arquitetura seguem condicionais.** Se treino compartilhado em
   corpus mais amplo ajustar bem mas validação continuar ruim, curvas por grupos e
   expansão FEBUS-like ganham prioridade. Se não ajustar, investigar otimização,
   geometria e rate-distortion. A taxa VQ continua muito agressiva; o contínuo não
   é comparação de codec na mesma taxa, e códigos maiores não são solução demonstrada.
6. Modelo final e treino do zero permanecem adiados. Uma seed, um fold e um pool de
   oito janelas não resolvem o diagnóstico de quantidade de dados do corpus completo.

## 5. Operação, recursos e reprodução

```bash
.venv/bin/python finetuning/run_shared_capacity_probe.py \
  --output-dir results/experiments/shared_capacity_probe_20261007
```

O comando acima foi executado e terminou. `--dry-run` verifica plano; `--resume`
retoma com os mesmos parâmetros e pula braços concluídos. Trava impede duas
instâncias no mesmo diretório. Progresso no terminal; métricas a cada 500 updates.

Checkpoints atômicos por braço: `probe_last.pt`, `probe_best_train.pt`,
`probe_best_val.pt`, com otimizador e passo. Métricas ficam em JSON por passo e CSV
resumido; recuperação usa o estado completo mais avançado e recupera o registro
se a gravação de JSON tiver sido interrompida. Uma interrupção refaz no máximo o
trecho posterior ao checkpoint. Pesos marcados `probe_only=true`, não de produção.

Retenção limitada porque havia ~15 GB livres: checkpoints novos somam ~1,56 GB
(decimal), sem apagar nada anterior. Verificação exige pelo menos 4 GiB livres para
escritas atômicas. Antes de novos experimentos grandes/downloads, planejar espaço;
nenhum dado ou checkpoint antigo foi removido automaticamente.

Artefatos: `protocol.json`, seleções de pool/controles/validação, históricos por braço,
`comparison.json`, `REPORT.md`, `learning_curves.png`, marcadores `COMPLETE`.
A curva gerada foi inspecionada: erro no pool cai enquanto controles/validação sobem.

**40 testes passaram**, cobrindo seleção determinística, separação de configurações/
arquivos, controles fora do pool, preservação das âncoras, agenda de 500 exposições
por janela e avaliação macro/per-sample. Experimento concluído; **nada permanece
rodando em background e nenhum modelo final foi treinado**.
