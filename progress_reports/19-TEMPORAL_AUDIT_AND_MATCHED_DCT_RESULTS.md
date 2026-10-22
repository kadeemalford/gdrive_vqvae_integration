# Auditoria temporal completa e controle DCT com orçamento pareado

> O diagnóstico adicional de fase foi concluído. O
> [relatório 20](20-TARGET_WEIGHTED_PHASE_DIAGNOSIS.md) mostra melhora de alinhamento
> com pesos do alvo, sem eliminar regressões localizadas e de cauda por canal.

## 1. Resultado e decisão

A auditoria completa confirma o ganho global do candidato longo 4x, mas revela
trade-offs de canal. O controle DCT não supera 4x no macro na amostra pareada,
porém **vence nas três configurações single-mode**. Não promover uma solução
única nem estender automaticamente o treino: investigar a cauda de erros e o
mecanismo da mudança de fase antes da próxima alteração do modelo.

Fontes concluídas:

- `results/experiments/temporal_density_long_audit_20261010/`: 5.820 janelas por
  candidato, standard/long_2x/long_4x, mesmo fold/normalização/target físico;
  checkpoints selecionados previamente, sem updates.
- `results/experiments/temporal_density_matched_dct_20261010/`: 102 janelas iguais
  para todos os pares, duas por arquivo, seed 2026, quatro famílias/seis configurações.
- Protocolos, hashes, métricas por janela/grupo e marcadores `COMPLETE` preservados.
- Serviço de avaliação+controle terminou com sucesso em **9min51s**.

São grupos de desenvolvimento já utilizados; nenhuma avaliação do teste ou
auditoria independente. Os resultados não estabelecem aceitação física ou produção.

## 2. Avaliação completa: erro global, taxa e cauda

| Candidato | RMSE macro | Correlação | RMS ratio | Fase FFT | Compressão nominal pooled |
|---|---:|---:|---:|---:|---:|
| Standard | 0,867647 | 0,450308 | 0,454018 | 0,793429 | 620,23x |
| Longo 2x | 0,848946 | 0,490283 | 0,489055 | 0,801028 | 325,09x |
| Longo 4x | **0,809797** | **0,559971** | **0,575644** | 0,796833 | 165,91x |

Pooled usa soma de bits de todas as janelas: 465.920.000 células e 232.800 bytes
side por candidato; índices fixos 22.176.000 / 44.000.000 / 88.000.000 bits.
Não é taxa de um codec neural completo: faltam container, mapas, identificação
de modelo e eventual codificação entrópica.

| Diagnóstico macro por canal | Standard | Longo 2x | Longo 4x |
|---|---:|---:|---:|
| Mediana do erro relativo | 0,917906 | 0,902127 | **0,880425** |
| p90 do erro relativo | **1,112015** | 1,136542 | 1,211698 |
| Erro médio dos canais não fracos | 0,944029 | 0,937380 | **0,934080** |

Esses números são médias macro de estatísticas calculadas por janela, não quantis
pooled de todos os canais. A mediana melhora e a cauda piora: **mais densidade não
é melhoria uniforme por canal**. No multimode, p90 sobe de **1,203466 para
1,512117** e erro não fraco de **1,054495 para 1,115751**, apesar da queda de RMSE
global. A regra de canal fraco é diagnóstica e não remove células do score global.

Comparado ao próprio piloto 4x, o alinhamento FFT cai de 0,820360 para 0,796833.
Frente ao standard, seu macro é ligeiramente maior, mas doped e multicore têm
alinhamento menor. O peso dessa métrica depende da amplitude reconstruída; não
atribuir automaticamente a queda a erro de fase em todas as componentes.

## 3. Controle pareado: mesma janela e teto, não taxa efetiva idêntica

O DCT recebe por janela um teto de bytes igual a
`ceil(bits_indices_reais / 8) + side_bytes` de cada rede. Sua seleção de coeficientes,
quantização 8-bit, header e normalização/escapes cabem nesse teto. Dados não são
decimados e escapes permanecem bit-exatos.

| Par | Rede RMSE macro | DCT RMSE macro | Bytes rede nominais / DCT reais |
|---|---:|---:|---:|
| Standard | 0,867342 | 0,884017 | 52.488 / 52.410 |
| Longo 2x | 0,848671 | 0,864460 | 100.032 / 99.816 |
| Longo 4x | **0,808423** | 0,835580 | 195.918 / 195.762 |

Os bytes são totais da amostra de 8.196.000 células. Diferenças de preenchimento
são pequenas, mas a taxa neural continua nominal, enquanto o DCT transmite um
stream real. Não anunciar equivalência de codecs implantados nem lower bound.
Também não comparar médias dessa amostra com o conjunto completo como se fossem
a mesma população.

### Detalhe do par 4x

| Família/configuração | Rede 4x RMSE | DCT ao teto 4x RMSE | Melhor no par |
|---|---:|---:|---|
| Doped / 5m SMF | 0,764791 | 0,912088 | Rede |
| Multicore / C1 and C4 | 0,851058 | 0,889361 | Rede |
| Multimode / Configuration 7 | 0,917004 | 0,917914 | Rede, diferença pequena |
| Single-mode / Configuration 11 | 0,680116 | 0,666728 | DCT |
| Single-mode / Configuration 3 | 0,728701 | 0,648362 | DCT |
| Single-mode / Configuration 7 | 0,693700 | 0,553782 | DCT |

O DCT também tem maior correlação nas três configurações single-mode. Contudo,
no par 4x a rede tem p90 menor em Configuration 3/7 e maior em Configuration 11;
não extrapolar a vitória de RMSE para todos os requisitos de canal.

Na amostra, fase FFT macro rede/DCT = **0,795061 / 0,729399**, p90 macro =
**1,198394 / 1,358481**, mediana macro = **0,875167 / 0,923897**. A rede tem
vantagem macro nesses diagnósticos, mas não domina todos os grupos. Um sistema
roteado por metadados é hipótese futura, não resultado de robustez: escolher
rotas pela validação e pontuá-las nos mesmos grupos não constitui confirmação
independente, e roteamento/modelos/overhead precisam ser cobrados.

## 4. Próximo diagnóstico iniciado: separar fase de ponderação

Antes de alterar loss ou treinar mais, comparar **4x inicial e longo** nas mesmas
102 janelas com um diagnóstico de fase cujo peso é somente a energia do original,
e não o produto de magnitudes original/reconstruída. Registrar:

- Alinhamento com pesos do alvo e cobertura de energia com fase definida.
- Alinhamento condicional, expondo a convenção para componentes não reconstruídas.
- Erro FFT complexo relativo por bandas normalizadas low/mid/high de Nyquist.
- Métricas por configuração e o diagnóstico anterior para referência.

Runner: `finetuning/audit_phase_tradeoff.py`.
Saída: `results/experiments/temporal_phase_audit_20261010/`.
Log: `results/experiments/temporal_phase_audit_20261010.log`.
Serviço: `febus-phase-audit-20261010`. Resultados concluídos no relatório 20;
a descrição abaixo registra o protocolo definido antes da análise.
Não há updates, seleção automática, banda física inventada ou acesso ao teste.

O limiar diagnóstico de fase é `max(max_amplitude_alvo * 1e-8, 1e-12)` por janela.
Componentes sem fase definida contribuem zero ao score incondicional e sua
fração é explicitada na cobertura; alinhamento condicional não deve ocultá-las.

## 5. Decisão de pesquisa, não novo treino automático

1. Usar o diagnóstico de fase para distinguir alteração de pesos de regressão
   nas componentes energéticas e localizar bandas/configurações problemáticas.
2. Se persistir falha nos canais não fracos, priorizar uma ablação de objetivo
   ou residual/representação que ataque a cauda, com bitrate explícito. Não
   substituir automaticamente o alvo por um sinal filtrado ou reusar uma loss
   STFT de magnitude como se já tivesse demonstrado melhora de fase.
3. Se os trade-offs forem aceitáveis para a aplicação, repetir o candidato com
   outra seed e grupos reservados antes de escolher um sistema definitivo.
4. Uma alternativa DCT/híbrida para single-mode precisa de escolha por dados de
   desenvolvimento e confirmação em novos grupos, cobrando todos os streams.

Limiares por canal/família, bitrate aceitável e tolerância de fase ainda não foram
definidos pelo uso pretendido. A configuração single-mode que regrediu no braço
2x e o multimode fraco continuam limites explícitos, não resolvidos pela média.

**119 testes passaram**, incluindo invariância do diagnóstico de fase a escala,
inversão, ausência de reconstrução, constantes e validação de shapes.
