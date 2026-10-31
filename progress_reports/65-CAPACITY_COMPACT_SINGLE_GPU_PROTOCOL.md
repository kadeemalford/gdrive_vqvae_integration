# Escala de capacidade na taxa compacta — uma RTX 4090

**Estudo concluído:** ambos os braços completaram 24 épocas e avaliações
completas. Resultados e limites estão no
[relatório 66](66-CAPACITY_COMPACT_RESULTS.md); registros de início abaixo são históricos.

## Pergunta e protocolo

Testar se expandir a CNN nativa melhora fidelidade FEBUS sem enviar mais índices.
O estudo usa `finetuning/run_capacity_study.py`; não introduz ROI, atenção, nova
loss ou outro quantizador. A escala é uma hipótese, não garantia de recuperar os
~744× offshore. Nenhum candidato será promovido automaticamente.

| Braço | Base channels | Parâmetros | Inicialização |
|---|---:|---:|---|
| `large96M` | 120 | **96.517.033** | Encoder/decoder aleatórios |
| `control28M` | 64 | **27.958.081** | Encoder/decoder aleatórios |

Mesma família residual e topologia; aumento apenas de largura interna. Não se
carregam tensores offshore. O contraste causal é scratch pequeno versus scratch
grande, não grande scratch versus pequeno pré-treinado. A transferência continua
referência histórica prática, mas seu resultado denso não é controle de taxa
compacta. A variante C não é reutilizada como se fosse uma expansão nativa.

- Corpus: `data/finetune/spatial_band_v1_fold1_20261026/`, **14.620 janelas de
  treino e 6.940 de validação**; test fold 1 não transferido/não inferido.
- Folds de treino 0/3, validação 2; seed **2033**, igual nos dois braços.
- `persistent-amplitude-v1` fixo, normalização `robust-escape-v1`; NPZ preservados.
  Runner verifica todos os SHA-256 dos arrays de treino/validação contra a
  transferência e cruza mapas/decisões de filtragem. Mesma preparação em ambos.
- Geometria **`spatial-4x`**, K=2.048, dimensão 128, commitment 0,25, 11 bits por
  índice; **8.064 índices para 512×1.000**, sem features/skips gratuitos.
- Objetivo: 0,5 MSE + 0,5 L1 nas células válidas + VQ, sem auxiliares espectrais
  ou por canal; AdamW LR 1e-4, weight decay 1e-5, clipping de gradiente do trainer.
- Batch de uma janela, amostragem window, FP32, activation checkpointing e replay
  discreto exato na avaliação. Não introduzir AMP simultaneamente à capacidade.
- Codebook inicializado em reservatório exclusivamente de treino: oito janelas
  por família, até 256 vetores por janela, mesmos caminhos/seed nos braços.
- **16×4.096 updates por braço**, sem parada precoce dentro desse orçamento.
  Extensão única de oito épocas se o melhor RMSE do último bloco de quatro épocas
  melhorar ≥0,2% sobre o bloco anterior. Teto **24 épocas/98.304 updates por braço**.
- Ordem: grande → controle nas fases iniciais; depois extensões elegíveis. Não
  inferir superioridade só porque a rede maior foi observada primeiro.
- Seleção: RMSE macro em triagem fixa com até duas janelas por arquivo/região.
  Avaliações completas dos selecionados nas fases 16 e, quando elegível, 24.
  Ponderação por células dentro de configuração e pesos iguais entre configurações
  da família/famílias, com RMSE/correlação/p90 e diagnósticos físicos por grupo.
- Comparar também fases 16 pareadas se as extensões diferirem. Updates iguais não
  equivalem a compute igual. Guardas descritivas de grande contra controle não
  substituem limiares físicos ou dados externos; nenhum teste reservado é usado.

## Preparação e execução remota

Máquina: `aumo-gpu-001`, workspace
`/home/murilo/source/das_febus/gdrive_vqvae_integration`.
As placas reais são **duas RTX 4090 de 24 GB**, não RTX 5090. A execução restringe
`CUDA_VISIBLE_DEVICES=0`, e o runner exige que só uma GPU esteja visível; GPU 1
não é usada. Não usar DDP ou somar memórias.

Código-base clonado no commit `a50ae69`; runner/testes de capacidade são adições
ainda não commitadas, transferidas com SHA-256 e fixadas no protocolo. O clone
não deve ser atualizado durante o estudo. `code_snapshot/` conserva os fontes
efetivamente usados, incluindo implementação VQ-VAE externa em `vendor/smartgrids`.
Dados, checkpoints, venv, dependência externa e resultados permanecem ignorados
pelo Git; nenhuma credencial é copiada para o projeto.

O ambiente foi instalado em `.venv`, com PyTorch **2.11.0+cu128**. O lock de
dependências e o log dos testes ficam em `results/remote_setup/`. Suíte local:
**217 testes aprovados**; remota: **217 coletados, sucesso com um skip**, relativo
a teste que exige caminho absoluto da máquina local.

Antes do treino, o runner executa cinco passos AdamW descartáveis em uma janela
real de treino 512×1.000 para cada largura; testa gradiente finito, contagem de
índices, replay bit-exato e shapes de 1/111/512 canais. O checkpoint de treino é
criado depois com seed reinicializada: nenhum update do smoke entra no estudo.
`smoke.json` registra segundos por passo e picos de memória. Esses passos medem
shape máximo, não throughput médio do corpus; estimativa total deve incluir
validação, checkpoints e a distribuição real de shapes.

Diretório do estudo:
`results/experiments/native_capacity_compact_fold1_20261029/`.
Unidade: `febus-capacity-20261029.service`, com GPU 0, limite operacional de 48 h
e RAM de 64 GiB. O runner interrompe fases com menos de 20 GiB livres, sem apagar
arquivos. Checkpoints são atômicos; resume preserva orçamento, optimizer e RNG,
exige protocolo/hashes iguais e verifica coerência checkpoint/CSVs.

```bash
systemctl --user status febus-capacity-20261029.service --no-pager
cat results/experiments/native_capacity_compact_fold1_20261029/STATUS.json
```

Após interrupção, conferir processos e checkpoint antes de retomar a mesma fila:

```bash
results/remote_setup/run_single_gpu.sh finetuning/run_capacity_study.py \
  --samples-dir data/finetune/spatial_band_v1_fold1_20261026 \
  --repo-root "$PWD/vendor/smartgrids" \
  --output-dir results/experiments/native_capacity_compact_fold1_20261029 \
  --resume
```

`STATUS.json`, `smoke.json`, logs, CSVs e `COMPLETE` são fontes de estado;
cancelamento de monitor de chat não estabelece término do treinamento.

## Preflight concluído e início efetivo

O smoke completo passou para as duas larguras e a fila entrou em `training`
para `large96M`, alvo inicial 16 épocas, em **29/10/2026 às 19:56 UTC**.

| Braço | Passo máximo após warm-up | Pico alocado | Pico reservado |
|---|---:|---:|---:|
| 96,5M | ~0,305 s | ~5,81 GiB | ~6,76 GiB |
| 28,0M | ~0,110 s | ~3,30 GiB | ~3,68 GiB |

Todos os shapes/replays passaram e os dois modelos enviam 8.064 índices no
shape máximo. Cinco passos sintéticos de benchmark sobre uma janela real não
medem a distribuição completa: adotar provisoriamente **6–9 h para 16 épocas da
rede maior**, com margem para avaliações/overhead. Se elegível até 24, o treino
máximo isolado na velocidade do smoke equivale a ~8,3 h, antes das avaliações;
o controle e suas avaliações têm duração própria. Refinar ETA com épocas reais.

Uma primeira tentativa de preflight detectou diferenças de até ~7,9e-5 entre
duas execuções CUDA da reconstrução. O replay estrito passou ao fixar kernels
cuDNN determinísticos no smoke, sem relaxar tolerâncias: diferença zero no
controle de replay. A política determinística é limitada ao teste e restaurada;
subprocessos de treino/avaliação mantêm a política numérica anterior. Nenhum
checkpoint de treino havia sido criado. Artefatos dessa tentativa foram
preservados em diretório com sufixo `_preflight_nondeterministic_cudnn`; o
protocolo e snapshot novos foram gerados antes do treino efetivo.
