# FEBUS: o efeito de contexto sobrevive à cobertura e à taxa completas?

**18/11/2026.** Protocolo registrado antes da nova inferência. Continuação dos
[relatórios 106](106-FEBUS_FEASIBILITY_AND_RESEARCH_REDIRECTION.md),
[107](107-FEASIBILITY_CONTRACT_AND_EXECUTION_HANDOFF.md) e
[108](108-FEASIBILITY_DIAGNOSTICS_EXECUTION.md).

## 1. Hipótese e desenho

O replay F1 mostrou que reconstruir um núcleo central usando um contexto de
oito canais frequentemente ajuda os modelos treinados em crops. Não demonstrou
que reconstruir **todos** os canais desse modo compense os índices e frames
extras. Este contraste testa a implementação mais direta dessa hipótese, antes
de consumir orçamento de treino. É diferente do estudo 79: aqui o contraste
é na geometria densa e usa grupos contíguos fixos de largura oito, sem seleção
por amplitude, no painel congelado do F1.

- **Amostra:** as mesmas 20 janelas train históricas, 20 arquivos e dez
  configurações. Todos os canais retidos e as 1.000 amostras de cada janela.
  Nenhuma seleção adicional por métricas ou waveform; não é confirmação
  independente e não representa todos os canais do HDF5.
- **Modelos:** VQ madura como referência e FiLM constante de 16.384 updates
  como representante treinado em oito canais. Escolha prospectiva para este
  contraste, motivada pelo F1; não é ranking independente entre checkpoints.
  O export FiLM histórico (SHA-256
  `c33767748ff7efadfbec9c65249847feea5ea593b384c31843391953d4f11c62`)
  é o receiver do estudo 105 e corresponde à identidade do registry congelado.
  Pesos obtidos por cópia de arquivo, sem uso de GPU remota.
- **Braços por modelo:** `full`, um frame por tile; `part8`, partição contígua
  desde o primeiro canal, sem overlaps, com último fragmento menor se preciso.
  Mesmo modelo, wire, registry e normalização nativa em cada contexto.
  Sem halos, escolha ótima de fronteira, interpolação, ajuste de pesos ou
  fallback seletivo por métrica. Não reestimar probabilidades nos checks.
- **Comparação causal limitada:** mede o efeito conjunto da política de
  particionamento (contexto, normalização, bordas e densidade de tokens).
  Não isola componente interna de arquitetura; não mede efeito de treinar
  com contexto alinhado.

## 2. Contrato e critérios predefinidos

Usar FAK1 com registry histórico por modelo, FiLM com wrapper de condição,
e FPB1 nos dois braços, incluindo índices, normalização, escapes, geometria,
integridade e overhead de fragmentos. Réplica do decoder recebe apenas os
bytes, geometria e a instalação congelada. Contabilizar o modelo à parte e
também no objeto com instalação, sob a plataforma de execução compartilhada;
não alegar binário independente do runtime. Contar cada célula retida uma vez.
Medir taxa real, não bits nominais. Publicar bytes por componente e por
família, latência medida do diagnóstico e limites de amortização.

Registrar RMSE/SSE físicos, NRMSE* macro de famílias, P90 de NRMSE* por canal,
máximo absoluto, viés e espectro com suporte definido pelo alvo por banda.
Manter gaps, constantes, zeros com sinal e escapes auditáveis. Os números
espectrais históricos são apenas trilha de compatibilidade com estudos prévios.

**Gate de eficiência diagnóstica:** `part8` precisa reduzir o NRMSE* macro em
pelo menos 5% contra seu próprio `full`, sem mais bytes completos, sem aumento
de SSE físico agregado, e sem degradação acima de 1% em NRMSE* de qualquer
família, P90 de canal por configuração ou erro espectral agregado por banda.
Se o controle tem erro zero, exigir ausência de aumento. Fase ausente e
piores canais ficam explícitos mesmo sem limiar operacional. Comparar também
ao ponto fixo da VQ madura; vencer somente controle fraco não promove.

Teto: 800 operações de encode e 1.600 de decode, zero updates. Geometria do
painel implica **440 operações de encode planejadas**. RTX 4060 local para
inferência. Para qualquer treino posterior, consultar as GPUs remotas na ocasião,
usar no máximo uma se livre e recorrer à local caso contrário.

Contagem a priori, somente pelos mapas: **1.533.000 células** por braço;
20 frames e 106.000 posições de índices no `full`, contra 200 frames e
200.000 posições no `part8` (constantes podem usar frame literal em vez de
índices). A taxa nominal de índices seria 0,761 contra 1,435 bpc antes de
entropia e side. Isso antecipa uma barreira de taxa, mas não substitui o wire
medido.

## 3. Como agir após o resultado

| Resultado | Interpretação e ação |
|---|---|
| Mais fidelidade sem mais bytes, com guardas | Vale investigar compatibilidade de contexto; ainda testar contra a VQ madura e fora da seleção exploratória |
| Mais fidelidade somente com mais bytes | Ponto de viabilidade, não eficiência. Não treinar automaticamente um codec part8 mais caro |
| Ganho central não transfere para todos os canais | Bordas e/ou distribuição de erro impedem extrapolação. Não promover o particionamento |
| Regressão ou falha de execução | Preservar resultado e distinguir negativo científico de erro técnico |

O resultado não libera teste fold0, validação consumida ou EDAS. F3 segue
bloqueado por falta de novas sessões. F2 temporal tem uma investigação
documental separada; não abrir sinais a partir do plano candidato de 108.

## 4. Resultado

[Protocolo arquivado](../results/experiments/feasibility_full_coverage_20261118/scientific_protocol)
e [resultados completos](../results/experiments/feasibility_full_coverage_20261118/results.json).
A primeira tentativa parou antes de qualquer forward por falta do pacote
`vqvae` no `PYTHONPATH`. A segunda usou as fontes congeladas do replay F1,
sem mudar o desenho. Na RTX 4060: 440 encodes, 880 redecodificações,
80 bundles com replay bit a bit, 40/40 SSEs centrais compatíveis com F1 a
`rtol=1e−5`, hashes dos 80 bundles conferidos e zero updates. Não havia
valores não finitos ou zeros no painel.

| Modelo | Política | NRMSE* macro | SSE físico | Bytes¹ | bpc¹ |
|---|---|---:|---:|---:|---:|
| VQ madura | full | 0,46552 | 3,1554 × 10¹⁶ | 133.727 | 0,69786 |
| VQ madura | part8 | 0,47433 | 3,7244 × 10¹⁶ | 285.251 | 1,48859 |
| FiLM constante | full | 0,95824 | 8,1978 × 10¹⁶ | 119.871 | 0,62555 |
| FiLM constante | part8 | 0,49331 | 2,9064 × 10¹⁶ | 288.087 | 1,50339 |

¹ Receiver e registry instalados; inclui bundles e geometria. A instalação
é medida separadamente. O FiLM sobe de 115.149.272 para 115.317.488 bytes com
instalação; o runtime não é empacotado.

- **FiLM:** −48,52% no NRMSE* macro, melhora nas quatro famílias, no SSE e nas
  guardas físicas. **+140,33% em bytes** de transporte; reprovação nos dois
  gates de bytes. Codificação agregada passou de 1,21 para 3,58 s; decodificação,
  de 0,58 para 1,71 s, sem I/O e sem ser benchmark de produção.
- **VQ madura:** `part8` aumentou NRMSE* em 1,89% e bytes em 113,31%, com
  regressões de cauda e espectro.
- FiLM `part8` tem 5,97% mais NRMSE* e 115,43% mais bytes que a madura `full`,
  embora o SSE bruto seja menor. Isso é conflito de métricas, não dominância.

**Decisão: encerrar `part8` como candidata direta.** A melhora no FiLM
transfere à cobertura, mas não à eficiência por bit. Não buscar partições
ou halos após o resultado. A alternativa distinta é treinar em geometria
full-tile, não transmitir mais tokens por fragmentação; protocolo no
[relatório 111](111-FEBUS_TRAIN_INFERENCE_CONTEXT_ALIGNMENT.md).

**Verificação:** 572 testes locais aprovados. A checagem dos arquivos tocados
não acusou whitespace; o `git diff --check` global aponta whitespace em
`consideracoes_futuras.txt`, alteração preexistente não modificada.
