# Protocolo de Personalização

## Definições
- Pré-adaptação: avaliação com pesos globais, sem usar amostras do sujeito alvo.
- Pós-adaptação: avaliação após usar K amostras rotuladas do sujeito alvo para atualizar apenas a cabeça.
- On-device head adaptation: adaptação local só da cabeça final, sem backprop no backbone.

## Passos
1. Selecionar sujeito/registro alvo fora do treino global.
2. Executar baseline pré-adaptação.
3. Sortear K amostras por classe (few-shot) com seed fixa.
4. Adaptar cabeça:
   - linear: gradiente somente em W,b da última camada
   - protótipos: média por classe no embedding
5. Reavaliar no conjunto de teste do alvo.
6. Reportar ganho e custo.

## Semântica dos logs embarcados
- `pre_linear`: inferência on-device antes de qualquer adaptação local.
- `post_linear_sgd`: atualização on-device de `W,b` com `linear_head_adapt()`.
- `post_prototype`: atualização on-device das médias por classe com `prototype_head_update()`.

Os logs da UART não significam treino completo on-device da rede. Eles medem somente adaptação da cabeça final.

## Evitar vazamento
- Nenhuma amostra de teste alvo entra na adaptação.
- Splits por registro/sujeito são fixados em metadata.
