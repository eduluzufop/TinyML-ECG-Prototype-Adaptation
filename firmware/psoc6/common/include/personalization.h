#pragma once
#include <stdint.h>

#include "model_params.h"

#define EMBEDDING_DIM ECG_PROJ_OUT_DIM
#define NUM_CLASSES ECG_NUM_CLASSES

typedef struct {
    float w[NUM_CLASSES][EMBEDDING_DIM];
    float b[NUM_CLASSES];
} linear_head_t;

typedef struct {
    float prototypes[NUM_CLASSES][EMBEDDING_DIM];
    uint32_t counts[NUM_CLASSES];
} prototype_head_t;

void linear_head_init(linear_head_t *head);
void linear_head_load(linear_head_t *head, const float *weight, const float *bias);
void linear_head_copy(linear_head_t *dst, const linear_head_t *src);
void linear_head_logits(const linear_head_t *head, const float *embedding, float *logits_out);
int linear_head_predict(const linear_head_t *head, const float *embedding);
void linear_head_sgd_step(linear_head_t *head, const float *embedding, int label, float learning_rate);
void linear_head_adapt(
    linear_head_t *head,
    const float *embeddings,
    const uint8_t *labels,
    uint32_t sample_count,
    uint32_t epochs,
    float learning_rate
);
void prototype_head_reset(prototype_head_t *head);
int prototype_head_predict(const prototype_head_t *head, const float *embedding);
void prototype_head_update(prototype_head_t *head, const float *embedding, int label);
