# Teste de capacidade: memorização de janelas de treino FEBUS

Data: 2026-10-07. Artefatos: `results/experiments/capacity_probe_20261007/`.

## Conclusão

O VQ-VAE atual consegue reconstruir substancialmente melhor exemplos vistos quando
otimizado especificamente para eles. Nas três janelas escolhidas de doped, multicore
e single-mode, atingiu correlação >0,96. Isso enfraquece a hipótese de incapacidade
absoluta de representar FEBUS nessas geometrias, **não demonstra generalização**.

A janela multimode é mais difícil: RMSE relativo 0,670 no VQ contra 0,553 no
autoencoder contínuo realmente treinado. O caminho contínuo tem vantagem nesse caso,
mas quantização, regularização VQ e otimização mudam juntas entre os braços; não é
prova de que um codebook maior, isoladamente, resolveria o problema.

**Todos os braços ainda melhoravam no passo 500, que foi também o melhor passo.**
Não foi demonstrado um piso de erro nem convergência. Não assumir que apenas faltam
dados, não começar treinamento final e não promover estes pesos como especialistas.

## 1. Protocolo executado

- Fonte: `sampling_full_20261005/fold_1/window/best.pt`, época 10. Mesmos pesos de
  encoder/decoder/codebook no início de cada par, com otimizador novo.
- Seleção apenas por metadados de **treino** do fold 1, seed 2026: configuração
  uniforme dentro da família, arquivo uniforme dentro da configuração e janela
  uniforme dentro do arquivo. Nenhum sinal/métrica de validação ou teste foi usado
  para escolher os casos. Todos os casos sorteados pertencem ao grupo de fold 0,
  que é treino no experimento de teste fold 1 / validação fold 2.
- Uma janela por família, um modelo independente por janela/braço: quatro casos,
  dois braços, 500 atualizações repetindo exatamente o alvo, total 4.000 atualizações.
- AdamW novo, LR `1e-4`, weight decay 0, grad clip 1. LR mais alta é para diagnóstico
  de overfit; não foi escolhida como LR recomendada de treino generalista.
- Geometria nativa, 1.000 amostras temporais, normalização global por janela idêntica
  ao fine-tuning. Dataset Strain Rate; `dt_s=5e-5`, duração 0,05 s, FFT de resolução 20 Hz.
- Braço VQ: `0,5*MSE + 0,5*L1 + VQ`, modelo completo ajustável.
- Braço contínuo: mesma arquitetura/padding/crop de encoder/decoder, **sem usar
  quantizador no forward**, embeddings congelados, `0,5*MSE + 0,5*L1`. Encoder e
  decoder realmente otimizados, diferente do bypass sem treino do relatório 4.
- Avaliação no próprio alvo a cada 100 passos; início como candidato no passo 0.
  Melhor checkpoint pelo RMSE do alvo visto. Não há early stopping nem holdout.

| Família | Configuração | Arquivo | Bloco / início temporal | Canais |
|---|---|---|---:|---:|
| Doped | 5m i29 | Copy of 1000Hz.h5 | 3 / 4210 | 45 |
| Multicore | C2 and C4 | Copy of 100Hz_CH1.h5 | 1 / 8421 | 70 |
| Multimode | Configuration 5 | Copy of _2025-12-08_09-15-11_UTC.h5 | 3 / 10526 | 512 |
| Single-mode | Configuration 9 | Copy of 5000Hz.h5 | 4 / 16842 | 23 |

Frequência no título é nominal do experimento, não sample rate. IDs/caminhos exatos
constam em `cases.csv` e `protocol.json`. Somente quatro janelas: não representam
toda a diversidade das famílias nem permitem inferência estatística por família.

## 2. Resultados no alvo memorizado

| Família | Modelo | RMSE inicial → melhor | Correlação final | Erro magnitude FFT final | RMS ratio final |
|---|---|---:|---:|---:|---:|
| Doped | VQ | 0,6616 → 0,2588 | 0,9660 | 0,1920 | 0,9521 |
| Doped | Contínuo | 0,6450 → 0,2689 | 0,9632 | 0,1963 | 0,9568 |
| Multicore | VQ | 0,9508 → 0,2298 | 0,9733 | 0,1621 | 0,9652 |
| Multicore | Contínuo | 0,9456 → 0,2272 | 0,9739 | 0,1600 | 0,9674 |
| Multimode | VQ | 0,9878 → 0,6701 | 0,7423 | 0,4822 | 0,7377 |
| Multimode | Contínuo | 0,9849 → 0,5527 | 0,8334 | 0,3882 | 0,8299 |
| Single-mode | VQ | 0,7442 → 0,2148 | 0,9767 | 0,1698 | 0,9692 |
| Single-mode | Contínuo | 0,7610 → 0,2003 | 0,9798 | 0,1578 | 0,9729 |

As métricas iniciais diferem entre braços apesar de pesos iguais: VQ e latente
contínuo são forwards diferentes. Ambos usam mesma geometria, alvo e orçamento.
Não comparar perdas totais como se a regularização VQ estivesse presente no contínuo.

### As curvas ainda não estabilizaram

| Família | RMSE VQ, 400 → 500 | RMSE contínuo, 400 → 500 |
|---|---:|---:|
| Doped | 0,3161 → 0,2588 | 0,3216 → 0,2689 |
| Multicore | 0,2651 → 0,2298 | 0,2593 → 0,2272 |
| Multimode | 0,7331 → 0,6701 | 0,6147 → 0,5527 |
| Single-mode | 0,2574 → 0,2148 | 0,2372 → 0,2003 |

Portanto a vantagem contínua em multimode é um resultado em orçamento finito, não
um limite assintótico nem uma estimativa isolada do custo da quantização.

### Espectro e visualização

Foram geradas figuras comparando original, VQ, contínuo, erros e FFT média por canal.
Inspecionadas diretamente as figuras de multimode e doped. A janela doped recupera
estrutura e amplitude dominantes com ambos os modelos; multimode mantém erro difuso
e potência atenuada, com vantagem do contínuo. Razão de potência na banda alta
(50–100% Nyquist, 5–10 kHz) no melhor modelo:

| Família | VQ | Contínuo |
|---|---:|---:|
| Doped | 0,804 | 0,773 |
| Multicore | 0,921 | 0,925 |
| Multimode | 0,659 | 0,770 |
| Single-mode | 0,862 | 0,855 |

O objetivo permanece reconstruir o dado original, inclusive componentes ruidosas;
estas não foram removidas para facilitar o teste. FFT curta sem taper tem leakage
e não certifica fidelidade física nas bandas abaixo de sua resolução.

## 3. Taxa de informação: diagnóstico nominal, não benchmark de codec

Quatro níveis de downsampling, 128 dimensões latentes, 2.048 códigos. Com padding
para múltiplos de 16, cada índice requer nominalmente 11 bits.

| Caso | Posições latentes | Input float32 / índices VQ | Input float32 / latente contínuo float32 |
|---|---:|---:|---:|
| Doped, 45×1000 | 189 | 692,6× | 1,86× |
| Multicore, 70×1000 | 315 | 646,5× | 1,74× |
| Multimode, 512×1000 | 2016 | 738,8× | 1,98× |
| Single-mode, 23×1000 | 126 | 531,0× | 1,43× |

Não são taxas efetivas de arquivos. Ignoram metadados, normalização, codificação
real dos índices, entropia e custo dos pesos. O contínuo **não** é um codec na mesma
taxa do VQ. Além disso, pesos treinados separadamente para cada janela podem guardar
informação do alvo: a reconstrução memorizada não prova compressão generalista nesses
ratios. Nenhuma curva rate-distortion de um codec fixo foi medida nesta etapa.

## 4. Interpretação e sequência revisada

1. O sistema tem capacidade de ajustar exemplos específicos em três geometrias,
   e também melhorou muito o caso multimode. Há agora evidência de otimização/
   representação útil, não de capacidade absolutamente insuficiente em todo FEBUS.
2. O contraste com reconstrução generalista fraca pode envolver cobertura, geometria,
   normalização, exposição por amostra e compartilhamento de capacidade. Não atribuir
   exclusivamente à massa de dados: aqui mudaram repetição do alvo, LR, weight decay
   e número de modelos; não se isolou nenhuma dessas causas.
3. **Próximo passo:** confirmar convergência em mais janelas de treino, sobretudo
   multimode, com orçamento maior e sensibilidade a LR. Como as curvas ainda caem,
   não declarar piso de capacidade a partir de 500 passos.
4. Depois, ablação pequena e controlada de VQ/regularização e taxa/stride, com modelo
   compartilhado por múltiplas janelas e validação por configuração. Separar remoção
   da quantização de mudanças da perda quando possível. Comparar fidelidade na mesma
   taxa ou reportar curvas; não tratar o contínuo como alternativa gratuita ao VQ.
5. Se treino compartilhado ajustar bem e configurações reservadas continuarem ruins,
   priorizar curvas de aprendizagem por grupos e ampliação FEBUS-like. Se mesmo treino
   continuar ruim após otimização suficiente, priorizar arquitetura/taxa. As causas
   podem coexistir. Uma única amostra bem memorizada não encerra esse diagnóstico.
6. MoS/ensemble continua candidato, mas estes oito modelos são **probes por janela**,
   não especialistas por fibra. Não usar seus pesos em produção nem avaliar generalização
   como se uma janela tivesse representado a família. Modelo final segue adiado.

## 5. Operação e reprodução

Comando executado (concluído):

```bash
.venv/bin/python finetuning/run_capacity_probe.py \
  --output-dir results/experiments/capacity_probe_20261007
```

O script mostra progresso e métricas a cada 100 atualizações. Usa trava por experimento,
`--dry-run` para plano e `--resume` com os mesmos parâmetros para retomar e pular
braços concluídos. Salva atomicamente `probe_last.pt` e `probe_best.pt` a cada 100
passos e no início; uma interrupção perde no máximo o trecho após o checkpoint.
Histórico, resumo e reconstruções ficam por família/braço; `comparison.json`,
`REPORT.md` e `COMPLETE` globais são gerados ao final. Checkpoints carregam
`probe_only=true` e `probe_mode`: **não são pesos de produção do pipeline**.

Para investigar orçamento maior, usar outro diretório e `--steps`, preservando
seed/casos/hiperparâmetros. `--resume` não altera o orçamento salvo, nem reinicia
treinos concluídos. Repetição independente recomeça do checkpoint uniforme original.

**35 testes passaram**, incluindo fluxo de gradiente no controle contínuo, ausência
de gradiente no quantizador, seleção restrita ao treino, taxa com padding e gravação
atômica de checkpoint. Experimento terminou; **nenhum processo em background ou
novo treino generalista/final foi iniciado**. Checkpoints anteriores preservados.
