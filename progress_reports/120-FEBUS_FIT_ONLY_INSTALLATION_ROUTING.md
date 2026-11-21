# FEBUS: decidir pelo fit quando vale instalar o contexto

**Protocolo prospectivo.** O [resultado 119](119-FEBUS_LOSSLESS_INSTALLATION_RESULTS.md)
reduziu a instalação sem alterar a reconstrução, mas as famílias não
economizaram uniformemente. Testar uma única política que dispense a
instalação contextual quando o benefício não se repetir nos arquivos de ajuste.

- Usar somente o fit congelado do [117](117-FEBUS_CONTINUOUS_F2_PROBE.md) para
  decidir. Para cada configuração e `ε/σ_cal` de 0,03 e 0,1, retirar um arquivo
  de fit por vez, ajustar K/CDF no restante e simular no arquivo retirado.
  Cobrar a instalação reversível do 118 e transporte completo.
- Escolher espaço-temporal somente se **todos** os arquivos retirados
  economizarem **≥3%** em bytes completos contra o marginal. Com menos de dois
  arquivos fit, escolher marginal. Sem ajuste de threshold ou novos braços.
- Congelar a escolha antes de consultar o ledger check. Reusar os bytes
  verificados de transporte e instalação de cada braço original em 117–119,
  sem reler sinais de check. Cobrar somente a instalação selecionada, a fonte
  compartilhada e um identificador por configuração/precisão. Auditar se os
  braços originais restauram os mesmos q/valores.
- Avaliar os limites do 107 sem relaxar famílias/configurações. Reportar
  separadamente economia e ausência de regressão. Uma família em fallback
  sem economia não é prova de vantagem generalista. Este check foi consumido,
  portanto o contraste é de desenvolvimento, não confirmação independente.

CPU, até 60 minutos, sem reserva, GPU, novo modelo ou treino neural.
Se a regra não resolver a troca entre taxa e cobertura, não selecionar
retrospectivamente os casos favoráveis nem varrer regras. Sem disparo de F4-A.
