# Fase com pesos do alvo: interpretação da continuação 4x

## 1. Resultado e mudança de interpretação

O diagnóstico pareado esclarece o aparente conflito entre RMSE/correlação e fase:
**o alinhamento com pesos fixados pelo alvo melhora nas seis configurações**,
apesar da queda da métrica ponderada pelas magnitudes original/reconstruída.
Não há suporte para tratar a continuação como piora geral de fase ou lançar uma
loss espectral genérica para corrigir esse suposto problema.

Ainda há regressões localizadas na banda baixa e na cauda por canal. O candidato
4x continua referência neural de pesquisa, sem aprovação de fidelidade final.

## 2. Protocolo e rastreabilidade

Fonte: `results/experiments/temporal_phase_audit_20261010/`, com protocolo,
hashes dos dois checkpoints, seleção das janelas, tabelas e `COMPLETE`.
Runner: `finetuning/audit_phase_tradeoff.py`.

- 4x selecionado no piloto versus 4x selecionado na continuação longa.
- Mesmas 102 janelas de validação do controle DCT, quatro famílias/seis configurações.
- Sem updates, mudança de pesos ou acesso ao teste.
- FFT por canal, DC excluído, pesos = magnitude original ao quadrado.
- Fase utilizável quando amplitudes original e prevista excedem
  `max(max_amplitude_alvo * 1e-8, 1e-12)` por janela.
- Componentes sem fase definida contribuem zero; cobertura e score condicional
  registrados. Cobertura foi 1,0 nessa amostra para ambos os candidatos.
- Bandas relativas a Nyquist: low (0–0,1], mid (0,1–0,5], high (0,5–1].
  Não são bandas físicas iguais entre arquivos com sample rates distintos.

## 3. Resultado macro na amostra

| Métrica | Inicial 4x | Longo 4x |
|---|---:|---:|
| RMSE | 0,839339 | 0,808423 |
| Fase ponderada por magnitudes original/reconstruída | 0,818814 | 0,795061 |
| Fase com pesos do alvo, não DC | **0,427790** | **0,561235** |
| Fase alvo low | 0,792278 | 0,788875 |
| Fase alvo mid | 0,455932 | 0,672760 |
| Fase alvo high | 0,179314 | 0,278468 |
| Erro FFT complexo relativo, não DC | 0,836628 | 0,805232 |
| Erro FFT complexo low | 0,715640 | 0,718684 |
| Erro FFT complexo mid | 0,833492 | 0,773603 |
| Erro FFT complexo high | 0,945982 | 0,932657 |

O ganho mais forte está em mid. High melhora, mas permanece pouco alinhada;
low piora ligeiramente no macro. Erro FFT complexo inclui amplitude e fase e
concorda com a melhora global de forma de onda, sem resolver todas as bandas.

## 4. Configurações

| Família/configuração | Fase alvo total inicial | Longo | Fase low inicial → longo |
|---|---:|---:|---:|
| Doped / 5m SMF | 0,506849 | 0,620830 | 0,779040 → 0,790320 |
| Multicore / C1 and C4 | 0,316754 | 0,472993 | **0,767576 → 0,741947** |
| Multimode / Configuration 7 | 0,284669 | 0,451237 | 0,801427 → 0,802922 |
| Single-mode / Configuration 11 | 0,600669 | 0,705336 | 0,709485 → 0,722209 |
| Single-mode / Configuration 3 | 0,611589 | 0,701444 | **0,849801 → 0,828741** |
| Single-mode / Configuration 7 | 0,596407 | 0,692862 | 0,903923 → 0,909984 |

Mid e high melhoram nas seis configurações. As regressões low de multicore e
Configuration 3 não devem ser ocultadas pelo alinhamento total favorável.

## 5. Implicação para o roadmap

O diagnóstico anterior não era inválido: ele responde a uma pergunta com pesos
dependentes da energia reconstruída. Mas sua queda não basta para concluir perda
geral de fase. O resultado atual é **compatível com mudança de ponderação e melhor
reconstrução de componentes antes atenuadas**; não prova causalidade completa.

Próximo experimento de modelo deve atacar os gargalos demonstrados:

1. Localizar quais canais/regiões/bandas geram a piora p90/nonweak em multimode,
   mantendo todas as células ordinárias no score principal.
2. Comparar uma única alteração de objetivo ou representação com controle 4x de
   mesma geometria, inicialização, exposição e taxa; avaliar ganhos globais e
   cauda conjuntamente. Não introduzir filtro no alvo nem normalização por canal
   como default sem repetir a comparação de custos/transferência.
3. Preservar a referência atual e confirmar candidatos promissores por seed e
   grupos adicionais. DCT segue controle competitivo para single-mode, não um
   fallback já validado por generalização independente.
4. Definir tolerâncias físicas para escolher entre checkpoints não dominados;
   sem elas, usar a fronteira de trade-offs em vez de declarar aceitação automática.

Não foi iniciado novo treinamento nesta etapa: o diagnóstico evita otimizar uma
hipótese de perda geral de fase que os dados não sustentam. Os três trabalhos
de auditoria/controle/diagnóstico estão concluídos; nenhum deles alterou checkpoints.

**119 testes passaram**, incluindo controles analíticos da métrica de fase.
