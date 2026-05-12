#pragma once

#include <stdint.h>

#include "personalization.h"

#define ECG_BACKBONE_INPUT_LENGTH 200

void ecg_backbone_infer(const float *signal, float *embedding_out);
void ecg_backbone_batch_infer(const float *signals, uint32_t sample_count, float *embeddings_out);
