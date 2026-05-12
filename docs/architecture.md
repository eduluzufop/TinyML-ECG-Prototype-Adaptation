# System Architecture

Este projeto tem dois blocos principais:
1. **Pipeline offline (Python)**: MIT-BIH real, split inter-paciente `DS1/DS2`, treino global e export de artefatos.
2. **Pipeline embarcado (C/MTB)**: inferência do backbone congelado e adaptação on-device apenas da cabeça final no `CY8CPROTO-063-BLE`.

## Fluxo offline
- `scripts/download_mitbih.py`: baixa os 44 registros do protocolo.
- `scripts/prepare_dataset.py`: prepara janelas por batimento e grava metadata do split `DS1/DS2`.
- `scripts/train_backbone.py`: treina o modelo global offline.
- `scripts/evaluate_pre_adaptation.py`: mede pré e pós-adaptação em `DS2`.
- `scripts/export_model.py`: exporta pesos do backbone e da cabeça para headers C.
- `scripts/export_replay_support.py`: exporta `support/query` por registro de `DS2`.
- `scripts/export_firmware_demo_data.py`: gera um `demo_replay.h` compacto para validação rápida no firmware.

## Fluxo embarcado
- **CM0+**: boot e wake-up do `CM4`.
- **CM4**: executa o app de adaptação da cabeça e imprime métricas por UART.
- **Backbone**: congelado; atualmente executado por runtime nativo em C a partir de `firmware/psoc6/generated/model_params.h`.
- **Cabeça adaptável**:
  - `pre_linear`: usa a cabeça global sem adaptação local
  - `post_linear_sgd`: adapta somente `W,b` com `SGD`
  - `post_prototype`: adapta por protótipos no espaço latente

## Fluxo on-device passo a passo

```text
ECG beat window (200 amostras)
        |
        v
ecg_backbone_infer()
conv1 -> pool1 -> conv2 -> pool2 -> avg pool -> projection
        |
        v
embedding z \in R^D
        |
        +----------------------------+
        |                            |
        v                            v
linear head                    prototype head
(W,b)                          (p_0, p_1, ..., p_C)
        |                            |
        |                            +-- support: update da media por classe
        |                            |
        +-- support: SGD so em W,b   +-- query: distancia ao prototipo
        |
        +-- query: argmax_c (W_c z + b_c)
```

### 1. Extracao do embedding

O sinal de entrada entra em `ecg_backbone_infer()` em
`firmware/psoc6/common/source/backbone_runtime.c`.

Ordem das operacoes:
- `conv1 + ReLU`
- `maxpool1`
- `conv2 + ReLU`
- `maxpool2`
- `global average pooling`
- `projection layer`

O resultado final e um vetor `embedding_out` de dimensao `EMBEDDING_DIM`.

### 2. Cabeca linear global

A cabeca linear e representada por:
- `w[NUM_CLASSES][EMBEDDING_DIM]`
- `b[NUM_CLASSES]`

Essas estruturas estao em `firmware/psoc6/common/include/personalization.h`
como `linear_head_t`.

Os pesos offline exportados sao carregados com `linear_head_load()` em
`firmware/psoc6/common/source/personalization.c`.

### 3. Predicao linear sem adaptacao

A inferencia linear base acontece em duas etapas:
- `linear_head_logits()`: calcula `score_c = b_c + \sum_i w_{c,i} z_i`
- `linear_head_predict()`: escolhe a classe de maior score

Isso implementa o baseline `pre_linear`.

### 4. Adaptacao da ultima camada com SGD

O SGD on-device esta em `linear_head_sgd_step()` em
`firmware/psoc6/common/source/personalization.c`.

Para cada embedding de suporte:
- calcula logits atuais
- calcula os termos do softmax
- calcula o gradiente da cross-entropy por classe
- atualiza apenas:
  - `b[c]`
  - `w[c][i]`

Ou seja: o backbone nao muda; apenas a ultima camada e ajustada.

A rotina que aplica varios passos sobre o conjunto de suporte e
`linear_head_adapt()`, que percorre:
- `epochs`
- `sample_count`
- e chama `linear_head_sgd_step()` para cada amostra

### 5. Atualizacao dos prototipos

A cabeca por prototipos e representada por:
- `prototypes[NUM_CLASSES][EMBEDDING_DIM]`
- `counts[NUM_CLASSES]`

Isso esta em `prototype_head_t` em
`firmware/psoc6/common/include/personalization.h`.

O update do prototipo ocorre em `prototype_head_update()`:
- incrementa `counts[label]`
- atualiza a media incremental do embedding da classe:
  `p <- p + (z - p) / n`

### 6. Classificacao por prototipos

A predicao e feita em `prototype_head_predict()` em
`firmware/psoc6/common/source/personalization.c`.

Para cada classe valida:
- calcula a distancia euclidiana quadratica entre o embedding da query e o prototipo
- escolhe a classe de menor distancia

Em formula:
- `d_c(z) = || z - p_c ||_2^2`
- `y_hat = argmin_c d_c(z)`

### 7. Onde isso e orquestrado na aplicacao

No app de demo em `firmware/psoc6/app/source/demo_ecg_app.c`:
- `linear_head_load()`: carrega a cabeca global
- `linear_head_copy()`: cria a cabeca que sera adaptada por SGD
- `prototype_head_reset()`: zera os prototipos
- `ecg_backbone_batch_infer()`: extrai embeddings de suporte
- `prototype_head_update()`: monta os prototipos com o suporte
- `linear_head_adapt()`: faz o SGD leve em `W,b`
- `ecg_backbone_infer()`: extrai embedding de cada query
- `linear_head_predict()`: gera `pre_pred` e `sgd_pred`
- `prototype_head_predict()`: gera `proto_pred`

### 8. Semantica experimental no firmware

No device, os tres caminhos significam:
- `pre_pred`: cabeca global, sem adaptacao local
- `sgd_pred`: mesma cabeca linear, mas atualizada com o suporte do episodio
- `proto_pred`: classificador por centroides no espaco de embeddings

## Fronteira atual do on-device learning
- Já existe adaptação on-device da última camada.
- Ainda não existe treino on-device do backbone.
- O workspace MTB já inclui `ml-tflite-micro` para futura troca do backend do backbone, mas o caminho validado hoje é o runtime nativo em C.
