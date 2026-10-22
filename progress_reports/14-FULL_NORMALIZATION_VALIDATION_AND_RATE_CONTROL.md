# Validação completa da normalização e mudança de foco para rate-distortion

> O controle clássico foi concluído e analisado no [relatório 15](15-SPARSE_DCT_RATE_DISTORTION_RESULTS.md).
> A hipótese neural derivada está no [relatório 16](16-TEMPORAL_LATENT_DENSITY_PILOT.md).

## 1. Resultado e decisão

A validação completa confirmou a regressão generalista da normalização por canal:
**RMSE macro 0,889889 versus 0,867810 no controle global**. O ganho de single-mode
persistiu, mas channel perdeu nas outras três famílias e teve correlação/fase
piores. Ao mesmo tempo, melhorou a distribuição de erros relativos por canal,
inclusive fora do grupo de baixa variância. O trade-off não é apenas um artefato
de denominadores próximos de zero.

**Não promover channel como padrão nem iniciar outra continuação longa igual.**
Investigar a relação entre fidelidade e taxa, comparando a rede com um controle
clássico explícito antes de atribuir tudo à normalização, falta de dados ou duração
do treino. Modelos e dados existentes permanecem preservados.

## 2. Avaliação completa: protocolo e integridade

Artefatos: `results/experiments/channel_full_validation_20261008/`.
Runner: `finetuning/evaluate_physical_candidates.py`.

- Mesmas **5.820 janelas**, 51 arquivos, quatro famílias, seis configurações do fold 1.
- Três candidatos selecionados previamente: preservation full epoch 6, piloto global
  epoch 6 e piloto channel epoch 4; checkpoint hashes registrados no protocolo.
- Validação em alvo comum após restauração, mesma região ordinária e hash de janelas.
- **Nenhum update, acesso ao teste ou seleção de outro epoch nesta avaliação.**
- Mesma população usada em decisões de desenvolvimento: expansão de validação não
  é auditoria externa intocada. Não confundir 5.820 janelas com réplicas independentes.

| Candidato | RMSE macro | Correlação macro | Pior família RMSE | Razão nominal pooled índices+side |
|---|---:|---:|---:|---:|
| Preservation full | 0,868975 | 0,446423 | 0,968188 | 620,23× |
| Piloto global | **0,867810** | **0,448959** | **0,967959** | **620,23×** |
| Piloto channel | 0,889889 | 0,396469 | 0,976572 | 280,72× |

A continuação global reduz ~0,001165 absoluto frente ao preservation full, ainda
um ganho pequeno. Channel tem ~2,54% mais erro que o piloto global no RMSE macro.

### Famílias e configurações

| Família | Global | Channel |
|---|---:|---:|
| Doped | 0,827220 | 0,899845 |
| Multicore | 0,921122 | 0,937643 |
| Multimode | 0,967959 | 0,976572 |
| Single-mode | 0,754937 | 0,745496 |

Channel melhora single-mode 3 e 7, mas piora 11:

| Configuração single-mode | Preservation full | Piloto global | Piloto channel |
|---|---:|---:|---:|
| 11 | 0,707881 | 0,709829 | 0,711412 |
| 3 | 0,796269 | 0,792326 | 0,776672 |
| 7 | 0,764816 | 0,762655 | 0,748403 |

Nenhum candidato domina todas as configurações. Até o global melhor no macro
regrede em Configuration 11 frente ao preservation full. Não selecionar um modelo
por janela usando o erro contra o original como se fosse um roteador disponível
no decoder. Um roteamento por metadados é hipótese separada, com fallback/custos,
não uma promoção automática baseada nessas seis configurações.

## 3. Robustez por canal, amplitude e custo

| Métrica macro | Piloto global | Piloto channel |
|---|---:|---:|
| Mediana de RMSE relativo por canal | 0,918698 | 0,861184 |
| p90 de RMSE relativo por canal | 1,118778 | 1,007186 |
| Média por canal fora do grupo fraco diagnóstico | 0,949065 | 0,874274 |
| RMS ratio reconstrução/original | 0,448743 | 0,403618 |
| Alinhamento de fase FFT | 0,793196 | 0,680033 |

As estatísticas por canal são resumidas por janela/configuração/família, não uma
quantila única de todos os canais do corpus. O grupo fraco usa regra heurística
de std descrita no relatório 13 e não foi excluído do score global.
Não usá-la como validade física ou tolerância definitiva. Channel é melhor nessas
estatísticas, mas a forma de onda global e a amplitude ainda têm grandes erros.

Taxa pooled calculada por soma dos bits, não média de razões:

- Mesmas 465.920.000 células; 1.863.680.000 bytes do alvo float32 preparado.
- Índices VQ fixos: 22.176.000 bits (2.772.000 bytes) em ambos.
- Side stream: **232.800 bytes global / 3.867.040 bytes channel**.
- Payload nominal: **3.004.800 bytes global / 6.639.040 bytes channel**.

Esses números não incluem pesos/model IDs, container, mapas físicos e codificação
entrópica. A taxa VQ é nominal, não um bitstream completo de produção. A versão
channel mais que dobra esse payload parcial; ganhos por canal precisam justificar
o custo conforme requisitos de uso, não apenas uma métrica normalizada favorável.

## 4. Próxima rota: controle clássico de taxa, sem treino

Benchmark iniciado em background, **CPU somente**, sem novo treino/GPU:
`results/experiments/sparse_dct_rate_control_20261008/`.
Runner: `finetuning/benchmark_rate_controls.py`.
Codec: `finetuning/sparse_dct_codec.py`, formato experimental `FSD1`.

### Controle transparente

- DCT 2D ortonormal do array global-normalizado, sem decimar tempo ou mudar o alvo.
- Seleção dos coeficientes de maior magnitude; índices **explicitamente transmitidos**.
- Quantização escalar fixa de oito bits com escala transmitida.
- Bit packing dos índices/valores, header, normalização global reversível e escapes.
- Decodificação do byte stream real; escapes bit-exatos e demais células lossy.
- Nenhum modelo ajustado, informação de alvo gratuita no decoder ou seleção oracle
  de reconstruções. Escolher coeficientes durante encoding é operação do codec,
  cujo custo de localização/quantização entra no stream.

A truncação de componentes causa distorção avaliada contra o **original completo**;
não é denoising/preprocessamento que modifica o alvo para melhorar a métrica.
Os mapas de origem do relatório 12 confirmam contiguidade dos recortes, mas o
bitstream experimental não contém metadados físicos/mapas de implantação completos.

### Orçamentos e dados

- Duas janelas por arquivo do treino e validação fold 1: **210 + 102**, seed 2026.
- Mesmas janelas de validação do piloto para comparação com os scores salvos do
  candidato global, sem novos forwards de rede nem acesso ao teste.
- Orçamentos de **1×, 4×, 16× e 64×** o tamanho nominal índices+side do VQ global
  de cada geometria; comprimento real do stream deve ficar dentro do orçamento.
- Registrar comprimento real, taxa, erro raw comum, distribuição por canal e
  fração de energia retida antes da quantização.

O DCT pode ficar abaixo do orçamento por discretização/overhead; não alegar taxas
exatamente iguais. Os níveis maiores usam **mais bits**, não são concorrentes
equivalentes ao VQ fixo. Fração de energia antes da quantização é diagnóstico de
compacidade da base, não qualidade do stream final nem lower bound universal.

### Critério de interpretação

1. Se um controle simples melhorar bastante no orçamento próximo do VQ, investigar
   representação/otimização/arquitetura, não tratar a taxa como única causa.
2. Se ganho exigir muitos mais bits, investigar curvas rate-distortion de modelos,
   antes de aumentar dados ou seguir um mesmo bitrate extremo sem requisito físico.
3. Se DCT também falhar, isso limita este codec/base/quantizador; **não prova um
   limite informacional da fibra** nem elimina modelos mais adequados.
4. Continuar acompanhando todas as famílias, piores configurações, amplitude/fase
   e custo total, com novas aquisições/configurações para auditoria final.

Resultados do benchmark não são presumidos neste relatório. A atividade é um
controle diagnóstico das etapas 3/8 do roadmap, não o modelo final.

## 5. Operação e verificações

Log: `results/experiments/sparse_dct_rate_control_20261008.log`.
O benchmark não grava novos datasets/checkpoints nem remove antigos; streams
temporários ficam em memória e resultados pequenos em tabelas/JSON.
Se interrompido, executar com outro diretório de saída, preservando tentativas.

Dependência SciPy explicitada em `requirements.txt`; disponível no ambiente usado.
**95 testes passaram** antes de iniciar o benchmark: bit packing, limites de bytes,
decodificação, escapes, streams inválidos, orçamento sem coeficientes e monotonicidade
da energia retida pré-quantização. Isso não garante monotonicidade do erro final
quantizado nem substitui avaliação de fidelidade. Dados e modelos preservados.
