#include "backbone_runtime.h"

#include "model_params.h"

static float g_conv1_out[ECG_CONV1_OUT_CHANNELS * ECG_INPUT_LENGTH];
static float g_pool1_out[ECG_CONV1_OUT_CHANNELS * (ECG_INPUT_LENGTH / 2)];
static float g_conv2_out[ECG_CONV2_OUT_CHANNELS * (ECG_INPUT_LENGTH / 2)];
static float g_pool2_out[ECG_CONV2_OUT_CHANNELS * (ECG_INPUT_LENGTH / 4)];
static float g_avg_out[ECG_CONV2_OUT_CHANNELS];

static int _same_padded_index(int index, int length) {
    if ((index < 0) || (index >= length)) {
        return -1;
    }
    return index;
}

static float _relu(float value) {
    return (value > 0.0f) ? value : 0.0f;
}

void ecg_backbone_infer(const float *signal, float *embedding_out) {
    const int input_length = ECG_INPUT_LENGTH;
    const int conv1_out_length = ECG_INPUT_LENGTH;
    const int pool1_length = ECG_INPUT_LENGTH / 2;
    const int conv2_out_length = pool1_length;
    const int pool2_length = ECG_INPUT_LENGTH / 4;
    const int conv1_pad = ECG_CONV1_KERNEL_SIZE / 2;
    const int conv2_pad = ECG_CONV2_KERNEL_SIZE / 2;

    for (int oc = 0; oc < ECG_CONV1_OUT_CHANNELS; ++oc) {
        for (int x = 0; x < conv1_out_length; ++x) {
            float acc = ecg_conv1_bias[oc];
            for (int k = 0; k < ECG_CONV1_KERNEL_SIZE; ++k) {
                int src = _same_padded_index(x + k - conv1_pad, input_length);
                if (src >= 0) {
                    int weight_index = ((oc * ECG_CONV1_IN_CHANNELS) * ECG_CONV1_KERNEL_SIZE) + k;
                    acc += ecg_conv1_weight[weight_index] * signal[src];
                }
            }
            g_conv1_out[(oc * conv1_out_length) + x] = _relu(acc);
        }
    }

    for (int oc = 0; oc < ECG_CONV1_OUT_CHANNELS; ++oc) {
        for (int x = 0; x < pool1_length; ++x) {
            float a = g_conv1_out[(oc * conv1_out_length) + (2 * x)];
            float b = g_conv1_out[(oc * conv1_out_length) + (2 * x + 1)];
            g_pool1_out[(oc * pool1_length) + x] = (a > b) ? a : b;
        }
    }

    for (int oc = 0; oc < ECG_CONV2_OUT_CHANNELS; ++oc) {
        for (int x = 0; x < conv2_out_length; ++x) {
            float acc = ecg_conv2_bias[oc];
            for (int ic = 0; ic < ECG_CONV2_IN_CHANNELS; ++ic) {
                for (int k = 0; k < ECG_CONV2_KERNEL_SIZE; ++k) {
                    int src = _same_padded_index(x + k - conv2_pad, conv2_out_length);
                    if (src >= 0) {
                        int weight_index =
                            (((oc * ECG_CONV2_IN_CHANNELS) + ic) * ECG_CONV2_KERNEL_SIZE) + k;
                        acc += ecg_conv2_weight[weight_index] * g_pool1_out[(ic * pool1_length) + src];
                    }
                }
            }
            g_conv2_out[(oc * conv2_out_length) + x] = _relu(acc);
        }
    }

    for (int oc = 0; oc < ECG_CONV2_OUT_CHANNELS; ++oc) {
        for (int x = 0; x < pool2_length; ++x) {
            float a = g_conv2_out[(oc * conv2_out_length) + (2 * x)];
            float b = g_conv2_out[(oc * conv2_out_length) + (2 * x + 1)];
            g_pool2_out[(oc * pool2_length) + x] = (a > b) ? a : b;
        }
    }

    for (int oc = 0; oc < ECG_CONV2_OUT_CHANNELS; ++oc) {
        float sum = 0.0f;
        for (int x = 0; x < pool2_length; ++x) {
            sum += g_pool2_out[(oc * pool2_length) + x];
        }
        g_avg_out[oc] = sum / (float)pool2_length;
    }

    for (int out = 0; out < ECG_PROJ_OUT_DIM; ++out) {
        float acc = ecg_proj_bias[out];
        for (int in = 0; in < ECG_PROJ_IN_DIM; ++in) {
            acc += ecg_proj_weight[(out * ECG_PROJ_IN_DIM) + in] * g_avg_out[in];
        }
        embedding_out[out] = acc;
    }
}

void ecg_backbone_batch_infer(const float *signals, uint32_t sample_count, float *embeddings_out) {
    for (uint32_t sample = 0; sample < sample_count; ++sample) {
        ecg_backbone_infer(
            &signals[sample * ECG_BACKBONE_INPUT_LENGTH],
            &embeddings_out[sample * EMBEDDING_DIM]
        );
    }
}
