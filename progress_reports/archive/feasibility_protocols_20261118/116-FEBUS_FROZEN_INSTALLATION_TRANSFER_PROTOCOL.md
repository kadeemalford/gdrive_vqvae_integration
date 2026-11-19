# FEBUS: teste condicional de transferência da instalação congelada

**Protocolo prospectivo.** Executar somente se o
[teste de amortização](115-FEBUS_FROZEN_INSTALLATION_AMORTIZATION.md)
mostrar economia agregada de transporte mais instalação **nas duas precisões**
no horizonte de 20 janelas por arquivo (ou no máximo disponível, registrado).
Não retunar limiares, tabelas ou população se o gatilho falhar.

O censo de metadados encontrou **25 arquivos candidatos** para checar
transferência: checks históricos do split train, nos oito grupos de 114,
fora dos componentes do envelope de 110 que tocam qualquer arquivo fit.
São arquivos novos para **este contraste de coder**, não sessões independentes:
o acervo e a escala P0 já foram usados. Multicore continua fora.
O filtro mais estrito, mantendo exatamente o dataset HDF5 identificado no
manifesto, `dt` e os índices físicos, confirma **19 arquivos compatíveis**
nos oito grupos, antes da seleção de janelas. Esse subconjunto estrito é o
que será usado; não será necessário invocar equivalência de produtos.

- Usar somente esses candidatos, congelando IDs e seleção pela ordem
  SHA-256 de `feasibility-transfer-v1|sample_path`, com desempate por caminho.
  Até 20 janelas por arquivo, sem overlap no mesmo bloco, mesmo dataset
  (ou mesmo produto/canal de aquisição com identificação explícita), dt e
  **mesmos índices físicos** da geometria da instalação de 114. Se não houver
  compatibilidade, marcar ausência; não escolher uma geometria diferente.
- Replicar as duas precisões, o roundtrip, o ledger e as curvas da etapa
  anterior; pagar instalação uma vez por configuração no objeto agregado.
  Publicar resultados e cobertura por arquivo e configuração, inclusive
  grupos sem dados compatíveis. Não calibrar nem escolher o braço no check.
- Sucesso de transferência neste painel exige economia agregada de pelo
  menos 10% e 0,05 bpc, incluindo instalação, nas duas precisões, sem perdas
  acima de 1% por configuração com cobertura. Mesmo um sucesso continua
  restrito à população avaliada e não libera F4-A automaticamente.

Apenas NPZ de treino e CPU; sem validação, teste, HDF5, EDAS ou GPU.
