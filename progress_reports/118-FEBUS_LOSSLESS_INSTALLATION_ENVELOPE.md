# FEBUS: compressão reversível da instalação do coder

**Protocolo prospectivo.** O [F2 contínuo](117-FEBUS_CONTINUOUS_F2_PROBE.md)
mostrou forte economia de transporte, mas a instalação anulou parte substancial
do ganho. Em vez de alterar probabilidades (o compartilhamento de linhas
regrediu no [115](115-FEBUS_POOLED_CONTEXT_TABLES.md)), testar se os mesmos
bytes de instalação podem ser entregues em formato menor.

- Uma única intervenção: envelope XZ/LZMA2 preset 6 da biblioteca padrão,
  com fallback ao conteúdo original quando o envelope comprimido não for menor.
  Cobrar identificador, framing, integridade e fonte incremental do decoder
  uma vez no objeto declarado. A plataforma Python/stdlib é compartilhada.
- Usar exclusivamente os ZIPs de instalação e os transportes congelados do
  F2 contínuo. Verificar SHA-256 e reconstituir **exatamente cada instalação
  original**. O transporte, as CDFs, a quantização, a cobertura e a
  reconstrução permanecem inalterados. Sem leitura de novos sinais.
- Aplicar o mesmo envelope ao marginal e aos contextuais. Medir tamanho
  efetivo e decompor tabela, framing e leitor. Avaliação primária em
  `ε/σ_cal = 0,03 e 0,1`; reportar economia contra a própria instalação
  original e contra o marginal reempacotado.
- Conferir os limites de 107, mantendo guardas por configuração/família.
  O contrato é instalação entregue uma vez mais transporte: não misturar
  com os antigos objetos executáveis por trecho. A redução exata conta
  como melhoria de implantação, mas não seleciona um modelo neural.

CPU; sem treino, alteração de gates, HDF5, validação, teste ou reservas.
Se o empacotamento não resolver o gargalo, não varrer formatos/presets.
