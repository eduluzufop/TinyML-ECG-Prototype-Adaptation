#include "personalization.h"
#include <math.h>
#include <float.h>

void linear_head_init(linear_head_t *head) {
    for (int c = 0; c < NUM_CLASSES; ++c) {
        head->b[c] = 0.0f;
        for (int i = 0; i < EMBEDDING_DIM; ++i) {
            head->w[c][i] = 0.0f;
        }
    }
}

void linear_head_load(linear_head_t *head, const float *weight, const float *bias) {
    for (int c = 0; c < NUM_CLASSES; ++c) {
        head->b[c] = bias[c];
        for (int i = 0; i < EMBEDDING_DIM; ++i) {
            head->w[c][i] = weight[(c * EMBEDDING_DIM) + i];
        }
    }
}

void linear_head_copy(linear_head_t *dst, const linear_head_t *src) {
    for (int c = 0; c < NUM_CLASSES; ++c) {
        dst->b[c] = src->b[c];
        for (int i = 0; i < EMBEDDING_DIM; ++i) {
            dst->w[c][i] = src->w[c][i];
        }
    }
}

void linear_head_logits(const linear_head_t *head, const float *embedding, float *logits_out) {
    for (int c = 0; c < NUM_CLASSES; ++c) {
        float score = head->b[c];
        for (int i = 0; i < EMBEDDING_DIM; ++i) {
            score += head->w[c][i] * embedding[i];
        }
        logits_out[c] = score;
    }
}

int linear_head_predict(const linear_head_t *head, const float *embedding) {
    float logits[NUM_CLASSES];
    linear_head_logits(head, embedding, logits);
    float best = -FLT_MAX;
    int best_c = 0;
    for (int c = 0; c < NUM_CLASSES; ++c) {
        float score = logits[c];
        if (score > best) {
            best = score;
            best_c = c;
        }
    }
    return best_c;
}

void linear_head_sgd_step(linear_head_t *head, const float *embedding, int label, float learning_rate) {
    float logits[NUM_CLASSES];
    float probs[NUM_CLASSES];
    float max_logit = -FLT_MAX;
    float sum_exp = 0.0f;

    linear_head_logits(head, embedding, logits);
    for (int c = 0; c < NUM_CLASSES; ++c) {
        if (logits[c] > max_logit) {
            max_logit = logits[c];
        }
    }
    for (int c = 0; c < NUM_CLASSES; ++c) {
        probs[c] = expf(logits[c] - max_logit);
        sum_exp += probs[c];
    }
    if (sum_exp <= 0.0f) {
        return;
    }

    for (int c = 0; c < NUM_CLASSES; ++c) {
        float gradient = (probs[c] / sum_exp) - (c == label ? 1.0f : 0.0f);
        head->b[c] -= learning_rate * gradient;
        for (int i = 0; i < EMBEDDING_DIM; ++i) {
            head->w[c][i] -= learning_rate * gradient * embedding[i];
        }
    }
}

void linear_head_adapt(
    linear_head_t *head,
    const float *embeddings,
    const uint8_t *labels,
    uint32_t sample_count,
    uint32_t epochs,
    float learning_rate
) {
    for (uint32_t epoch = 0; epoch < epochs; ++epoch) {
        for (uint32_t sample = 0; sample < sample_count; ++sample) {
            linear_head_sgd_step(
                head,
                &embeddings[sample * EMBEDDING_DIM],
                (int)labels[sample],
                learning_rate
            );
        }
    }
}

void prototype_head_reset(prototype_head_t *head) {
    for (int c = 0; c < NUM_CLASSES; ++c) {
        head->counts[c] = 0;
        for (int i = 0; i < EMBEDDING_DIM; ++i) {
            head->prototypes[c][i] = 0.0f;
        }
    }
}

void prototype_head_update(prototype_head_t *head, const float *embedding, int label) {
    uint32_t n = ++head->counts[label];
    for (int i = 0; i < EMBEDDING_DIM; ++i) {
        float prev = head->prototypes[label][i];
        head->prototypes[label][i] = prev + (embedding[i] - prev) / (float)n;
    }
}

int prototype_head_predict(const prototype_head_t *head, const float *embedding) {
    float best_dist = FLT_MAX;
    int best_c = 0;
    for (int c = 0; c < NUM_CLASSES; ++c) {
        if (head->counts[c] == 0u) {
            continue;
        }
        float d = 0.0f;
        for (int i = 0; i < EMBEDDING_DIM; ++i) {
            float diff = embedding[i] - head->prototypes[c][i];
            d += diff * diff;
        }
        if (d < best_dist) {
            best_dist = d;
            best_c = c;
        }
    }
    return best_c;
}
