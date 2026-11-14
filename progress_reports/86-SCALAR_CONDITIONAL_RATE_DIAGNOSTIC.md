# Diagnóstico de taxa condicional dos coeficientes escalares

**Protocolo: 12/11/2026, registrado antes dos resultados.** Sem treino neural.

## 1. Motivo do redirecionamento

Os adaptadores congelados e a ablação MSE tiveram benefícios pequenos e não
atingiram os critérios de investimento. Não iniciar outra arquitetura longa
sem mecanismo melhor sustentado. O hyperprior do relatório 77 ainda depende
de evidência de que a informação adicional que ele transmitiria se paga.

O prior checkerboard de índices VQ do relatório 75 não pagou seus pesos. Aqui
o alvo é diferente: **coeficientes escalares assinados**, com magnitudes locais
variáveis, dos codecs E2 funcionais. O diagnóstico mantém a reconstrução intacta
e mede side explícito de escala, sem alegar que um novo prior aprendeu fidelidade.

## 2. Fontes e controles fixos

Usar dois checkpoints retidos do fold 1:

- `scalar_rd_c4_lambda005_fold1_20261031/best.pt`;
- `scalar_rd_warm_c16_lambda02_fold1_20261031/best.pt`.

Não usar o C16 colapsado. Verificar hashes contra os resultados históricos,
manifesto e implementação escalar. Os mesmos coeficientes arredondados de cada
fonte serão codificados por três probabilidades, com framing comum:

1. **Logístico aprendido:** escalas do próprio checkpoint, sem recalibração.
2. **Marginal empírico:** histograma por canal, suavização +1, ajustado somente
   em arquivos de treino de calibração. Se ele resolver o custo sozinho, não
   atribuir o ganho à informação condicional.
3. **Escala local:** marginal empírico ou um de 64 logísticos de escala fixa
   geométrica entre 0,05 e 256. O compressor escolhe por custo dos símbolos num
   bloco 8×8 da grade latente e transmite **um byte por bloco e canal**.

Sem centrar coeficientes, remover caudas ou mudar o produto físico. Valores fora
de [-255,255] têm escape int32 explícito. As tabelas são transmitidas em float32,
inclusive todo o banco extra: não supor que o receptor reproduzirá CDFs de float
ou que os pesos/side são gratuitos.

## 3. Seleção e integridade

Uma janela por arquivo/região de treino, seed 2104, com partição interna por
IDs de arquivo para calibração versus checks. Não abrir formas de onda de
validação ou teste. Esses arquivos já participaram do treinamento histórico:
o diagnóstico não certifica aquisições independentes ou generalização.

Formatos comuns **LSS1 + FSB1**, com um range stream por frame, mesmo percurso
dos símbolos e overhead de cabeçalho comparável. Cobrar side original, escolhas
de escala, escapes, CDFs, registry, manifesto e mapas. Reportar frames e bundle
completo, sem comparar o framing novo diretamente com os containers antigos
como se toda diferença fosse mérito do prior.

Os três decoders precisam recuperar coeficientes int32 idênticos; a síntese neural
é repetida e comparada bit a bit. O reader independente recupera as tabelas do
bundle e verifica os hashes de todos os coeficientes. Os pesos neurais ficam
congelados e seu custo compartilhado, igual entre variantes da mesma fonte,
continua discriminado. Não comparar C4/C16 como ablação causal de prior.

## 4. Critério anterior aos resultados

Só considerar inferência completa de desenvolvimento se o local economizar:

- **≥10% dos bytes reais dos frames frente ao marginal empírico** nos checks;
- **≥10% do pacote projetado** para o número conhecido de janelas de validação,
  cobrando o overhead fixo adicional e o side médio observado.

A projeção é um filtro de investimento, **não uma economia medida na validação**;
a variabilidade do tamanho/geometria e dos símbolos pode invalidá-la. O bundle
completo efetivamente observado nos checks também será reportado, inclusive se
perder por custo das tabelas na coorte curta. Nenhuma variante será promovida
pela projeção, e **nenhum treino de hyperprior é disparado automaticamente**.

Se negativo, registrar e não expandir a busca de tiles/bancos na mesma coorte.
Se positivo, confirmar os bytes completos e a identidade de reconstrução no
desenvolvimento antes de decidir por qualquer modelo aprendido. Um codec escalar
mais barato ainda precisa ser confrontado com as referências VQ de fidelidade
compatível, não apenas com seu próprio transporte anterior dominado.

Execução: inferência dos dois checkpoints sequencialmente, somente na GPU 0
remota, sem updates neurais. Implementação em
`compression/local_scale_stream.py`, `finetuning/audit_scalar_conditional_rate.py`
e `tests/test_local_scale_stream.py`. Saída própria
`results/experiments/scalar_conditional_rate_train_fold1_20261112/`.

## 5. Auditoria de compatibilidade antes da inferência

A primeira tentativa foi interrompida **antes de executar os modelos** porque
o checkpoint C4 registrava versões antigas de dois arquivos. Os snapshots
preservados foram comparados:

- `run_scalar_rd_study.py`: inclusão posterior de warm-start e registros da
  inicialização; avaliação passou a validar float integral antes do cast int32.
  O runner histórico de treino não é executado pelo novo diagnóstico.
- `scalar_stream.py`: correção das guardas dos limites int32. Somente construtor
  e geração das tabelas são usados pelo diagnóstico; o encoder antigo de frames
  SRD1 não é chamado. Os métodos de tabelas não mudaram.
- `scalar_rd.py` e `representation_study.py`: **hashes idênticos**. O checkpoint
  C16 warm já registra as versões atuais de todos esses arquivos.

Não foram restaurados arquivos antigos sobre o trabalho atual nem ignoradas
divergências genericamente. `finetuning/scalar_source_lineage.py` autoriza somente
os **pares exatos de SHA-256 inspecionados**, verifica o snapshot original e
registra a exceção como compatibilidade de inferência, nunca como autorização
para retomar treino histórico. Qualquer outra mudança continua bloqueada.

A tentativa original e seu protocolo ficam preservados com
`BLOCKED_LINEAGE_REVIEW.json`, sem resultados científicos. A retomada usa
`scalar_conditional_rate_train_v2_fold1_20261112/`, com novo protocolo e snapshot.
O cálculo de projeção usa o custo exato do arquivo de metadados/registry da
coorte proposta e seus comprimentos de framing, extrapolando **somente os bytes
dos frames**; não presume overhead de uma coorte curta constante na longa.
Isso continua não sendo resultado medido sobre waveforms de validação.

## 6. Resultado: não sustenta investir em hyperprior por este mecanismo

A versão 2 concluiu com **113 janelas de calibração / 22 checks**, todas de
treino, separados por IDs de arquivo. Nenhum update neural ou waveform de
validação/teste foi usado. Os coeficientes e as reconstruções neurais foram
bit-exatos nos três transportes, incluindo reader independente de bundle.

| Fonte / probabilidade | Frames (bytes) | Tabelas antes de ZIP (bytes) | Bundle completo de checks (bytes) |
|---|---:|---:|---:|
| C4 / logístico aprendido | 102.658 | 8.192 | 1.007.946 |
| C4 / marginal empírico | **102.522** | 8.192 | **1.007.425** |
| C4 / escala local | 102.755 | 139.264 | 1.054.657 |
| C16 warm / logístico aprendido | 174.470 | 32.768 | 1.079.535 |
| C16 warm / marginal empírico | 172.382 | 32.768 | **1.077.489** |
| C16 warm / escala local | **171.414** | 163.840 | 1.123.588 |

Contra o marginal empírico, a escala local **piorou frames em 0,2273% no C4** e
economizou apenas **0,5615% no C16**. Os bundles efetivamente observados ficaram
**4,6884% / 4,2784% maiores**, respectivamente, mesmo cobrando as tabelas com ZIP
no transporte. Os pesos neurais compartilhados são idênticos entre os três
modos de cada fonte: 27.104.676 bytes de tensores no C4 e 27.227.652 no C16.

As projeções, **não medições de validação**, foram −0,3617% / +0,4666% de economia
completa. Nem a economia de frames nem a projeção atingiram 10%. **Não abrir
inferência completa, ampliar bancos/tiles ou treinar hyperprior por este sinal.**

Isso não prova ausência de redundância condicional ou refuta hyperpriors
aprendidos. O banco é de escalas logísticas centradas em zero e o side é
deliberadamente explícito. Um modelo mais rico pode comportar-se de outra forma;
o diagnóstico atual, porém, não sustenta esse investimento. A calibração marginal
isolada recuperou pouco, não mudou a fidelidade dos codecs dominados de E2 e
não constitui nova representação de alta fidelidade.

A direção seguinte separa **contexto de análise antes da escolha dos símbolos**
de contexto de síntese após o bottleneck. O negativo do relatório 84 restringe
adaptadores dos tokens já transmitidos; não mede se relações entre features
contínuas do encoder podem selecionar símbolos melhores. Essa hipótese terá
controle local pareado e triagem de treino antes de novo codec completo.
