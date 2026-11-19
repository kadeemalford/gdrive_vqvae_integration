# FEBUS: triagem de redundância em janelas de 50 ms

**18/11/2026. Protocolo prospectivo.** Depois do negativo de eficiência do
[F4-B](112-FEBUS_CONTEXT_ALIGNMENT_RESULTS.md), o próximo investimento é CPU,
não um novo treino. O [F2 contínuo](110-FEBUS_TEMPORAL_CONTRACT_SENSITIVITY.md)
permanece bloqueado. Este protocolo testa outro estimando: **economia de
codificação dentro de janelas preparadas com resets a cada 1.000 amostras**.
Não mede continuidade entre janelas nem valida timestamps.

## População e método

- Usar somente os papéis candidatos do envelope do relatório 110 como
  **exclusão conservadora**, não como certificação temporal. Entram apenas
  configurações com fit e check por componentes: oito configurações, três
  famílias, sem multicore. Não reatribuir arquivos ou preencher o grupo ausente.
- Uma janela preparada por arquivo elegível, escolhida pelo menor SHA-256
  de `feasibility-npz-v1|sample_path` (desempate lexicográfico). Fixar
  seleção e hashes antes da codificação. Preservar todos os canais retidos,
  contar todos os valores e nunca concatenar janelas.
- O censo por metadados encontrou **18 janelas fit (656.000 células)** e
  **oito check (271.000 células)**. Em single-mode/Configuration 8 o fit
  tem largura dez e o check 23: manter a diferença, cobrar sua geometria
  e interpretá-la como limitação. Cada configuração tem apenas um arquivo
  check; não inferir estabilidade entre novas sessões a partir desta triagem.
- Reusar a quantização direta e o coder rANS inteiro de
  [107](107-FEASIBILITY_CONTRACT_AND_EXECUTION_HANDOFF.md), nas duas precisões
  primárias `ε/σ_cal = 0,03 e 0,1`, com escala P0 congelada. Para cada
  configuração, ajustar no fit o marginal e os modelos causais; escolher
  suporte pelo marginal. Usar apenas contexto que o decoder consegue
  reproduzir e pagar geometria, resets, escapes, tabelas e instalação.
- Comparar marginal, temporal e espaço-temporal no mesmo `q`; todas as
  reconstruções devem coincidir por bits entre braços e respeitar o bound.
  Publicar erro físico, NLL/CDF, bytes completos, ambos os contratos de
  instalação, valores especiais, tamanho da amostra e custos por configuração.

## Decisões após a execução

Sem seleção de modelo no check. Os limiares de **10% e 0,05 bpc** do 107
servem apenas como referência de materialidade **neste recorte**, não como
gate generalista. Separar falta de economia na CDF de economia absorvida
pela instalação. Economia localizada merece registro, não novo treino
automático. Ausência de economia não descarta dependência de longo alcance.

Se houver inconsistência física, de cobertura, roundtrip ou de proveniência,
registrar bloqueio em vez de substituir dados. F3 e F2 contínuo permanecem
condicionados às evidências já especificadas. Não ler HDF5, validação, teste
ou EDAS. Não há GPU ou updates previstos.
