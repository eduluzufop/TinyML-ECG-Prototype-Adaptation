#include "demo_ecg_app.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#include "backbone_runtime.h"
#include "cy_result.h"
#include "cybsp.h"
#include "cyhal.h"
#include "demo_replay.h"
#include "head_bias.h"
#include "head_weight.h"
#include "model_params.h"
#include "personalization.h"

#define DEMO_STREAM_DELAY_MS 900U
#define DEMO_SESSION_GAP_MS  2000U
#define DEMO_STREAM_POINTS   32U

typedef struct {
    uint32_t total;
    uint32_t correct;
} accuracy_counter_t;

static cyhal_uart_t g_uart;

static void _uart_init(void) {
    static const cyhal_uart_cfg_t cfg = {
        .data_bits = 8U,
        .stop_bits = 1U,
        .parity = CYHAL_UART_PARITY_NONE,
        .rx_buffer = NULL,
        .rx_buffer_size = 0U,
    };
    cy_rslt_t result = cyhal_uart_init(
        &g_uart,
        CYBSP_DEBUG_UART_TX,
        CYBSP_DEBUG_UART_RX,
        NC,
        NC,
        NULL,
        &cfg
    );
    if (result != CY_RSLT_SUCCESS) {
        CY_ASSERT(0);
        for (;;) {
        }
    }
    result = cyhal_uart_set_baud(&g_uart, 115200U, NULL);
    if (result != CY_RSLT_SUCCESS) {
        CY_ASSERT(0);
        for (;;) {
        }
    }
}

int _write(int fd, const char *buffer, int count) {
    (void)fd;
    for (int i = 0; i < count; ++i) {
        cy_rslt_t result = cyhal_uart_putc(&g_uart, (uint32_t)buffer[i]);
        if (result != CY_RSLT_SUCCESS) {
            break;
        }
    }
    return count;
}

static const char *_label_name(uint8_t label) {
    return (label == 0U) ? "normal" : "arrhythmic";
}

static int _to_milli(float value) {
    float scaled = value * 1000.0f;
    if (scaled >= 0.0f) {
        return (int)(scaled + 0.5f);
    }
    return (int)(scaled - 0.5f);
}

static void _uart_write_str(const char *text) {
    (void)_write(1, text, (int)strlen(text));
}

static void _emit_session_start(uint32_t session_id) {
    char line[192];
    int len = snprintf(
        line,
        sizeof(line),
        "{\"type\":\"session\",\"app\":\"demoECG\",\"session\":%lu,"
        "\"signal_length\":%u,\"signal_scale\":\"milli\","
        "\"support_samples\":%u,\"query_samples\":%u}\r\n",
        (unsigned long)session_id,
        DEMO_STREAM_POINTS,
        ECG_DEMO_SUPPORT_SAMPLES,
        ECG_DEMO_QUERY_SAMPLES
    );
    if (len > 0) {
        (void)_write(1, line, len);
    }
}

static void _emit_beat(
    uint32_t session_id,
    uint32_t beat_index,
    const char *phase,
    const float *signal,
    uint8_t true_label,
    int pre_pred,
    int sgd_pred,
    int proto_pred
) {
    char meta[1024];
    char wave[2048];
    const uint32_t stride = ECG_DEMO_SIGNAL_LENGTH / DEMO_STREAM_POINTS;
    int meta_len = snprintf(
        meta,
        sizeof(meta),
        "{\"type\":\"beat_meta\",\"app\":\"demoECG\",\"session\":%lu,"
        "\"beat_index\":%lu,\"phase\":\"%s\",\"true_label\":%u,"
        "\"true_name\":\"%s\",\"pre_pred\":",
        (unsigned long)session_id,
        (unsigned long)beat_index,
        phase,
        (unsigned)true_label,
        _label_name(true_label)
    );
    int wave_len;

    if (meta_len < 0) {
        return;
    }
    if (pre_pred >= 0) {
        meta_len += snprintf(&meta[meta_len], sizeof(meta) - (size_t)meta_len, "%d", pre_pred);
    } else {
        meta_len += snprintf(&meta[meta_len], sizeof(meta) - (size_t)meta_len, "null");
    }
    meta_len += snprintf(&meta[meta_len], sizeof(meta) - (size_t)meta_len, ",\"pre_name\":");
    if (pre_pred >= 0) {
        meta_len += snprintf(&meta[meta_len], sizeof(meta) - (size_t)meta_len, "\"%s\"", _label_name((uint8_t)pre_pred));
    } else {
        meta_len += snprintf(&meta[meta_len], sizeof(meta) - (size_t)meta_len, "null");
    }
    meta_len += snprintf(&meta[meta_len], sizeof(meta) - (size_t)meta_len, ",\"sgd_pred\":");
    if (sgd_pred >= 0) {
        meta_len += snprintf(&meta[meta_len], sizeof(meta) - (size_t)meta_len, "%d", sgd_pred);
    } else {
        meta_len += snprintf(&meta[meta_len], sizeof(meta) - (size_t)meta_len, "null");
    }
    meta_len += snprintf(&meta[meta_len], sizeof(meta) - (size_t)meta_len, ",\"sgd_name\":");
    if (sgd_pred >= 0) {
        meta_len += snprintf(&meta[meta_len], sizeof(meta) - (size_t)meta_len, "\"%s\"", _label_name((uint8_t)sgd_pred));
    } else {
        meta_len += snprintf(&meta[meta_len], sizeof(meta) - (size_t)meta_len, "null");
    }
    meta_len += snprintf(&meta[meta_len], sizeof(meta) - (size_t)meta_len, ",\"proto_pred\":");
    if (proto_pred >= 0) {
        meta_len += snprintf(&meta[meta_len], sizeof(meta) - (size_t)meta_len, "%d", proto_pred);
    } else {
        meta_len += snprintf(&meta[meta_len], sizeof(meta) - (size_t)meta_len, "null");
    }
    meta_len += snprintf(&meta[meta_len], sizeof(meta) - (size_t)meta_len, ",\"proto_name\":");
    if (proto_pred >= 0) {
        meta_len += snprintf(&meta[meta_len], sizeof(meta) - (size_t)meta_len, "\"%s\"", _label_name((uint8_t)proto_pred));
    } else {
        meta_len += snprintf(&meta[meta_len], sizeof(meta) - (size_t)meta_len, "null");
    }
    meta_len += snprintf(&meta[meta_len], sizeof(meta) - (size_t)meta_len, "}\r\n");

    wave_len = snprintf(
        wave,
        sizeof(wave),
        "{\"type\":\"beat_wave\",\"app\":\"demoECG\",\"session\":%lu,\"beat_index\":%lu,\"samples_milli\":[",
        (unsigned long)session_id,
        (unsigned long)beat_index
    );
    for (uint32_t i = 0U; i < DEMO_STREAM_POINTS; ++i) {
        uint32_t sample_index = i * stride;
        wave_len += snprintf(
            &wave[wave_len],
            sizeof(wave) - (size_t)wave_len,
            (i == 0U) ? "%d" : ",%d",
            _to_milli(signal[sample_index])
        );
    }
    wave_len += snprintf(&wave[wave_len], sizeof(wave) - (size_t)wave_len, "]}\r\n");

    if (meta_len > 0) {
        (void)_write(1, meta, meta_len);
    }
    if (wave_len > 0) {
        (void)_write(1, wave, wave_len);
    }
}

static void _counter_update(accuracy_counter_t *counter, int pred, uint8_t truth) {
    counter->total++;
    if (pred == (int)truth) {
        counter->correct++;
    }
}

static void _emit_summary(
    uint32_t session_id,
    const accuracy_counter_t *pre_counter,
    const accuracy_counter_t *sgd_counter,
    const accuracy_counter_t *proto_counter
) {
    char line[192];
    int len = snprintf(
        line,
        sizeof(line),
        "{\"type\":\"summary\",\"app\":\"demoECG\",\"session\":%lu,"
        "\"pre_acc\":%.3f,\"sgd_acc\":%.3f,\"proto_acc\":%.3f}\r\n",
        (unsigned long)session_id,
        (pre_counter->total > 0U) ? ((float)pre_counter->correct / (float)pre_counter->total) : 0.0f,
        (sgd_counter->total > 0U) ? ((float)sgd_counter->correct / (float)sgd_counter->total) : 0.0f,
        (proto_counter->total > 0U) ? ((float)proto_counter->correct / (float)proto_counter->total) : 0.0f
    );
    if (len > 0) {
        (void)_write(1, line, len);
    }
}

void demo_ecg_app_run(void) {
    static float support_embeddings[ECG_DEMO_SUPPORT_SAMPLES * EMBEDDING_DIM];
    linear_head_t pretrained_head;
    linear_head_t adapted_head;
    prototype_head_t prototype_head;
    uint32_t session_id = 0U;

    _uart_init();

    _uart_write_str("\r\n");
    _uart_write_str("{\"type\":\"banner\",\"app\":\"demoECG\",\"message\":\"PSoC 6 ECG demo ready\"}\r\n");

    for (;;) {
        accuracy_counter_t pre_counter = {0U, 0U};
        accuracy_counter_t sgd_counter = {0U, 0U};
        accuracy_counter_t proto_counter = {0U, 0U};

        session_id++;
        linear_head_load(&pretrained_head, ecg_head_weight, ecg_head_bias);
        linear_head_copy(&adapted_head, &pretrained_head);
        prototype_head_reset(&prototype_head);

        _emit_session_start(session_id);

        ecg_backbone_batch_infer((const float *)ecg_demo_support, ECG_DEMO_SUPPORT_SAMPLES, support_embeddings);

        for (uint32_t i = 0U; i < ECG_DEMO_SUPPORT_SAMPLES; ++i) {
            const float *embedding = &support_embeddings[i * EMBEDDING_DIM];
            const float *signal = &ecg_demo_support[i * ECG_DEMO_SIGNAL_LENGTH];
            uint8_t label = ecg_demo_support_labels[i];

            prototype_head_update(&prototype_head, embedding, (int)label);
            _emit_beat(session_id, i, "support", signal, label, -1, -1, -1);
            cyhal_system_delay_ms(DEMO_STREAM_DELAY_MS);
        }

        linear_head_adapt(
            &adapted_head,
            support_embeddings,
            ecg_demo_support_labels,
            ECG_DEMO_SUPPORT_SAMPLES,
            50U,
            0.05f
        );

        for (uint32_t i = 0U; i < ECG_DEMO_QUERY_SAMPLES; ++i) {
            float query_embedding[EMBEDDING_DIM];
            const float *signal = &ecg_demo_query[i * ECG_DEMO_SIGNAL_LENGTH];
            uint8_t truth = ecg_demo_query_labels[i];
            int pre_pred;
            int sgd_pred;
            int proto_pred;

            ecg_backbone_infer(signal, query_embedding);
            pre_pred = linear_head_predict(&pretrained_head, query_embedding);
            sgd_pred = linear_head_predict(&adapted_head, query_embedding);
            proto_pred = prototype_head_predict(&prototype_head, query_embedding);

            _counter_update(&pre_counter, pre_pred, truth);
            _counter_update(&sgd_counter, sgd_pred, truth);
            _counter_update(&proto_counter, proto_pred, truth);

            _emit_beat(session_id, i, "query", signal, truth, pre_pred, sgd_pred, proto_pred);
            cyhal_system_delay_ms(DEMO_STREAM_DELAY_MS);
        }

        _emit_summary(session_id, &pre_counter, &sgd_counter, &proto_counter);
        cyhal_system_delay_ms(DEMO_SESSION_GAP_MS);
    }
}
