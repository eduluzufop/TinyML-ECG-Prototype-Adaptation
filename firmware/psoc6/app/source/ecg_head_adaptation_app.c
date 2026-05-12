#include "ecg_head_adaptation_app.h"

#include <stdint.h>
#include <stdio.h>

#include "backbone_runtime.h"
#include "benchmark_replay.h"
#include "cy_result.h"
#include "cybsp.h"
#include "cycfg_system.h"
#include "cyhal.h"
#include "demo_replay.h"
#include "dwt_cycles.h"
#include "model_params.h"
#include "personalization.h"

#if defined(__GNUC__)
#define ECG_UNUSED_FN __attribute__((unused))
#else
#define ECG_UNUSED_FN
#endif

/* -----------------------------------------------------------------------
 * Global profiling results — readable via OpenOCD mdw after firmware halts.
 * Sentinel 0xDEADBEEF in magic field confirms the struct was written.
 * ----------------------------------------------------------------------- */
typedef struct {
    uint32_t magic;              /* 0xDEADBEEF when valid */
    uint32_t configured_hf0_hz;  /* Configured clock used for offline decoding, if set */
    uint32_t infer_reps;
    uint32_t dwt_ok;             /* 1 if DWT is functional */
    uint32_t api_hf0_hz;         /* Cy_SysClk_ClkHfGetFrequency(0) */
    uint32_t system_core_clock_hz; /* CMSIS SystemCoreClock after update */
    uint32_t dwt_spin_cyc;       /* cycles for 1000-iter volatile spin — cross-check */
    /* single-beat inference: cycles (mean, min, max) */
    uint32_t infer_mean_cyc;
    uint32_t infer_min_cyc;
    uint32_t infer_max_cyc;
    /* prototype adaptation total cycles for K=1,5,10 */
    uint32_t proto_K1_cyc;
    uint32_t proto_K5_cyc;
    uint32_t proto_K10_cyc;
    /* SGD adaptation total cycles for K=1,5,10 */
    uint32_t sgd_K1_cyc;
    uint32_t sgd_K5_cyc;
    uint32_t sgd_K10_cyc;
} prof_results_t;

volatile prof_results_t g_prof_results;

typedef struct {
    uint32_t magic; /* 0xBATCH001 when valid */
    uint32_t episodes;
    uint32_t total_support;
    uint32_t total_query;
    uint32_t batch_total_cyc;
    uint32_t avg_episode_cyc;
    uint32_t pre_acc_milli;
    uint32_t pre_f1_macro_milli;
    uint32_t sgd_acc_milli;
    uint32_t sgd_f1_macro_milli;
    uint32_t proto_acc_milli;
    uint32_t proto_f1_macro_milli;
} batch_results_t;

volatile batch_results_t g_batch_results;

typedef struct {
    uint32_t cm[NUM_CLASSES][NUM_CLASSES];
} binary_metrics_t;

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
    CY_ASSERT(result == CY_RSLT_SUCCESS);
    result = cyhal_uart_set_baud(&g_uart, 115200U, NULL);
    CY_ASSERT(result == CY_RSLT_SUCCESS);
    (void)result;
}

int _write(int fd, const char *buffer, int count) {
    (void)fd;
    size_t len = (size_t)count;
    (void)cyhal_uart_write(&g_uart, (void *)buffer, &len);
    return (int)len;
}

static void _metrics_reset(binary_metrics_t *metrics) {
    for (uint32_t label = 0U; label < NUM_CLASSES; ++label) {
        for (uint32_t pred = 0U; pred < NUM_CLASSES; ++pred) {
            metrics->cm[label][pred] = 0U;
        }
    }
}

static void _metrics_update(binary_metrics_t *metrics, int pred, int label) {
    if ((label < 0) || (label >= (int)NUM_CLASSES) || (pred < 0) || (pred >= (int)NUM_CLASSES)) {
        return;
    }
    metrics->cm[label][pred]++;
}

static void _metrics_merge(binary_metrics_t *dst, const binary_metrics_t *src) {
    for (uint32_t label = 0U; label < NUM_CLASSES; ++label) {
        for (uint32_t pred = 0U; pred < NUM_CLASSES; ++pred) {
            dst->cm[label][pred] += src->cm[label][pred];
        }
    }
}

static uint32_t _metrics_total(const binary_metrics_t *metrics) {
    uint32_t total = 0U;
    for (uint32_t label = 0U; label < NUM_CLASSES; ++label) {
        for (uint32_t pred = 0U; pred < NUM_CLASSES; ++pred) {
            total += metrics->cm[label][pred];
        }
    }
    return total;
}

static float _metrics_accuracy(const binary_metrics_t *metrics) {
    uint32_t total = _metrics_total(metrics);
    uint32_t correct = 0U;
    for (uint32_t cls = 0U; cls < NUM_CLASSES; ++cls) {
        correct += metrics->cm[cls][cls];
    }
    return (total > 0U) ? ((float)correct / (float)total) : 0.0f;
}

static float _metrics_f1_for_class(const binary_metrics_t *metrics, uint32_t cls) {
    float tp = (float)metrics->cm[cls][cls];
    float fp = 0.0f;
    float fn = 0.0f;
    for (uint32_t other = 0U; other < NUM_CLASSES; ++other) {
        if (other == cls) {
            continue;
        }
        fp += (float)metrics->cm[other][cls];
        fn += (float)metrics->cm[cls][other];
    }
    if ((tp <= 0.0f) && ((fp > 0.0f) || (fn > 0.0f))) {
        return 0.0f;
    }
    {
        float precision_den = tp + fp;
        float recall_den = tp + fn;
        float precision = (precision_den > 0.0f) ? (tp / precision_den) : 0.0f;
        float recall = (recall_den > 0.0f) ? (tp / recall_den) : 0.0f;
        float sum = precision + recall;
        return (sum > 0.0f) ? (2.0f * precision * recall / sum) : 0.0f;
    }
}

static float _metrics_f1_macro(const binary_metrics_t *metrics) {
    float f1_sum = 0.0f;
    for (uint32_t cls = 0U; cls < NUM_CLASSES; ++cls) {
        f1_sum += _metrics_f1_for_class(metrics, cls);
    }
    return f1_sum / (float)NUM_CLASSES;
}

static uint32_t _to_milli(float value) {
    if (value <= 0.0f) {
        return 0U;
    }
    return (uint32_t)(value * 1000.0f + 0.5f);
}

static void _print_fixed3(uint32_t milli_value) {
    printf("%lu.%03lu", (unsigned long)(milli_value / 1000U), (unsigned long)(milli_value % 1000U));
}

static void _print_metric_line(const char *name, const binary_metrics_t *metrics) {
    uint32_t acc_milli = _to_milli(_metrics_accuracy(metrics));
    uint32_t f1_milli = _to_milli(_metrics_f1_macro(metrics));
    printf("%s acc=", name);
    _print_fixed3(acc_milli);
    printf(" f1_macro=");
    _print_fixed3(f1_milli);
    printf("\r\n");
}

static void _print_batch_episode_line(
    uint32_t record_id,
    uint32_t support_count,
    uint32_t query_count,
    const binary_metrics_t *pre_metrics,
    const binary_metrics_t *linear_metrics,
    const binary_metrics_t *prototype_metrics
) {
    printf(
        "BATCH rec=%lu support=%lu query=%lu pre_f1_macro=",
        (unsigned long)record_id,
        (unsigned long)support_count,
        (unsigned long)query_count
    );
    _print_fixed3(_to_milli(_metrics_f1_macro(pre_metrics)));
    printf(" sgd_f1_macro=");
    _print_fixed3(_to_milli(_metrics_f1_macro(linear_metrics)));
    printf(" proto_f1_macro=");
    _print_fixed3(_to_milli(_metrics_f1_macro(prototype_metrics)));
    printf("\r\n");
}

static void _evaluate_linear(
    const linear_head_t *head,
    const float *query_embeddings,
    const uint8_t *query_labels,
    uint32_t query_count,
    binary_metrics_t *metrics
) {
    for (uint32_t i = 0U; i < query_count; ++i) {
        int pred = linear_head_predict(head, &query_embeddings[i * EMBEDDING_DIM]);
        _metrics_update(metrics, pred, (int)query_labels[i]);
    }
}

static void _evaluate_prototypes(
    const prototype_head_t *head,
    const float *query_embeddings,
    const uint8_t *query_labels,
    uint32_t query_count,
    binary_metrics_t *metrics
) {
    for (uint32_t i = 0U; i < query_count; ++i) {
        int pred = prototype_head_predict(head, &query_embeddings[i * EMBEDDING_DIM]);
        _metrics_update(metrics, pred, (int)query_labels[i]);
    }
}

/* -----------------------------------------------------------------------
 * DWT Profiling
 * Cycles are the primary metric. Convert to time off-target using the
 * validated hardware clock, not a CMSIS bookkeeping variable.
 * Inference stats: additional min=/max=/n= fields.
 * ----------------------------------------------------------------------- */
#define PROF_INFER_REPS  200U
#define PROF_K_COUNT     3U

extern uint32_t SystemCoreClock;
extern void SystemCoreClockUpdate(void);

static void _prof_print(const char *tag, uint32_t cycles) {
    printf("PROF %-36s cyc=%lu\r\n", tag, (unsigned long)cycles);
}

static ECG_UNUSED_FN void _run_profiling(void) {
    static float s_emb[20U * EMBEDDING_DIM];
    static const uint8_t s_lbl[20U] = {
        0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1
    };
    static const uint32_t k_vals[PROF_K_COUNT] = {1U, 5U, 10U};

    uint32_t t0, t1;
    uint32_t sum_c, mn_c, mx_c, elapsed;

    g_prof_results.magic = 0U;
    g_prof_results.configured_hf0_hz = 0U;
    g_prof_results.infer_reps = PROF_INFER_REPS;
    g_prof_results.dwt_ok = 0U;
    SystemCoreClockUpdate();
    g_prof_results.api_hf0_hz = Cy_SysClk_ClkHfGetFrequency(0U);
    g_prof_results.system_core_clock_hz = SystemCoreClock;
    g_prof_results.dwt_spin_cyc = 0U;

    printf("\r\n===PROFILE_START===\r\n");
    printf(
        "PROF cfg_hf0_hz=%lu infer_reps=%u api_hf0_hz=%lu syscore_hz=%lu\r\n",
        (unsigned long)g_prof_results.configured_hf0_hz,
        PROF_INFER_REPS,
        (unsigned long)g_prof_results.api_hf0_hz,
        (unsigned long)g_prof_results.system_core_clock_hz
    );
    dwt_init();
    {
        uint32_t c0 = dwt_count();
        volatile uint32_t spin = 1000U;
        while (spin--) {
            (void)spin;
        }
        {
            uint32_t c1 = dwt_count();
            if (c1 == c0) {
                printf("PROF dwt=UNAVAILABLE\r\n");
                printf("===PROFILE_END===\r\n");
                return;
            }
            g_prof_results.dwt_ok = 1U;
            g_prof_results.dwt_spin_cyc = c1 - c0;
            printf("PROF dwt=OK sanity_delta=%lu\r\n", (unsigned long)(c1 - c0));
        }
    }

    for (uint32_t i = 0U; i < 5U; ++i) {
        ecg_backbone_infer(ecg_demo_support, s_emb);
    }
    sum_c = 0U;
    mn_c = 0xFFFFFFFFU;
    mx_c = 0U;
    for (uint32_t i = 0U; i < PROF_INFER_REPS; ++i) {
        t0 = dwt_count();
        ecg_backbone_infer(ecg_demo_support, s_emb);
        t1 = dwt_count();
        elapsed = t1 - t0;
        sum_c += elapsed;
        if (elapsed < mn_c) {
            mn_c = elapsed;
        }
        if (elapsed > mx_c) {
            mx_c = elapsed;
        }
    }
    {
        uint32_t mean_c = sum_c / PROF_INFER_REPS;
        g_prof_results.infer_mean_cyc = mean_c;
        g_prof_results.infer_min_cyc = mn_c;
        g_prof_results.infer_max_cyc = mx_c;
        printf(
            "PROF %-36s cyc=%lu min_cyc=%lu max_cyc=%lu n=%u\r\n",
            "infer_single_beat",
            (unsigned long)mean_c,
            (unsigned long)mn_c,
            (unsigned long)mx_c,
            PROF_INFER_REPS
        );
    }

    {
        volatile uint32_t *proto_slots[PROF_K_COUNT] = {
            &g_prof_results.proto_K1_cyc,
            &g_prof_results.proto_K5_cyc,
            &g_prof_results.proto_K10_cyc
        };
        for (uint32_t ki = 0U; ki < PROF_K_COUNT; ++ki) {
            uint32_t K = k_vals[ki];
            uint32_t n_sup = K * 2U;
            char tag[48];
            prototype_head_t ph;
            uint32_t ti = 0U;
            const char *prefix = "proto_adapt_K";
            const char *suffix = "_total";

            for (; prefix[ti]; ++ti) {
                tag[ti] = prefix[ti];
            }
            if (K == 1U) {
                tag[ti++] = '1';
            } else if (K == 5U) {
                tag[ti++] = '5';
            } else {
                tag[ti++] = '1';
                tag[ti++] = '0';
            }
            for (uint32_t si = 0U; suffix[si]; ++si) {
                tag[ti++] = suffix[si];
            }
            tag[ti] = '\0';

            t0 = dwt_count();
            prototype_head_reset(&ph);
            for (uint32_t i = 0U; i < n_sup; ++i) {
                uint32_t src = i % (uint32_t)ECG_DEMO_SUPPORT_SAMPLES;
                ecg_backbone_infer(
                    &ecg_demo_support[src * ECG_DEMO_SIGNAL_LENGTH],
                    &s_emb[i * EMBEDDING_DIM]
                );
                prototype_head_update(&ph, &s_emb[i * EMBEDDING_DIM], (int)s_lbl[i]);
            }
            t1 = dwt_count();
            *proto_slots[ki] = t1 - t0;
            _prof_print(tag, t1 - t0);
        }
    }

    {
        volatile uint32_t *sgd_slots[PROF_K_COUNT] = {
            &g_prof_results.sgd_K1_cyc,
            &g_prof_results.sgd_K5_cyc,
            &g_prof_results.sgd_K10_cyc
        };
        for (uint32_t ki = 0U; ki < PROF_K_COUNT; ++ki) {
            uint32_t K = k_vals[ki];
            uint32_t n_sup = K * 2U;
            char tag[48];
            linear_head_t lh;
            uint32_t ti = 0U;
            const char *prefix2 = "sgd_adapt_K";
            const char *suffix2 = "_total";

            for (; prefix2[ti]; ++ti) {
                tag[ti] = prefix2[ti];
            }
            if (K == 1U) {
                tag[ti++] = '1';
            } else if (K == 5U) {
                tag[ti++] = '5';
            } else {
                tag[ti++] = '1';
                tag[ti++] = '0';
            }
            for (uint32_t si = 0U; suffix2[si]; ++si) {
                tag[ti++] = suffix2[si];
            }
            tag[ti] = '\0';

            linear_head_load(&lh, ecg_head_weight, ecg_head_bias);
            t0 = dwt_count();
            for (uint32_t i = 0U; i < n_sup; ++i) {
                uint32_t src = i % (uint32_t)ECG_DEMO_SUPPORT_SAMPLES;
                ecg_backbone_infer(
                    &ecg_demo_support[src * ECG_DEMO_SIGNAL_LENGTH],
                    &s_emb[i * EMBEDDING_DIM]
                );
            }
            linear_head_adapt(&lh, s_emb, s_lbl, n_sup, 50U, 0.05f);
            t1 = dwt_count();
            *sgd_slots[ki] = t1 - t0;
            _prof_print(tag, t1 - t0);
        }
    }

    g_prof_results.magic = 0xDEADBEEFU;
    printf("===PROFILE_END===\r\n");
}

static ECG_UNUSED_FN void _run_demo_episode(void) {
    static float support_embeddings[ECG_DEMO_SUPPORT_SAMPLES * EMBEDDING_DIM];
    static float query_embeddings[ECG_DEMO_QUERY_SAMPLES * EMBEDDING_DIM];
    linear_head_t pretrained_head;
    linear_head_t adapted_head;
    prototype_head_t prototype_head;
    binary_metrics_t pre_metrics;
    binary_metrics_t linear_metrics;
    binary_metrics_t prototype_metrics;

    _metrics_reset(&pre_metrics);
    _metrics_reset(&linear_metrics);
    _metrics_reset(&prototype_metrics);

    linear_head_load(&pretrained_head, ecg_head_weight, ecg_head_bias);
    linear_head_copy(&adapted_head, &pretrained_head);
    prototype_head_reset(&prototype_head);

    ecg_backbone_batch_infer((const float *)ecg_demo_support, ECG_DEMO_SUPPORT_SAMPLES, support_embeddings);
    ecg_backbone_batch_infer((const float *)ecg_demo_query, ECG_DEMO_QUERY_SAMPLES, query_embeddings);

    _evaluate_linear(
        &pretrained_head,
        query_embeddings,
        ecg_demo_query_labels,
        ECG_DEMO_QUERY_SAMPLES,
        &pre_metrics
    );

    for (uint32_t i = 0U; i < ECG_DEMO_SUPPORT_SAMPLES; ++i) {
        prototype_head_update(
            &prototype_head,
            &support_embeddings[i * EMBEDDING_DIM],
            (int)ecg_demo_support_labels[i]
        );
    }
    linear_head_adapt(
        &adapted_head,
        support_embeddings,
        ecg_demo_support_labels,
        ECG_DEMO_SUPPORT_SAMPLES,
        50U,
        0.05f
    );

    _evaluate_linear(
        &adapted_head,
        query_embeddings,
        ecg_demo_query_labels,
        ECG_DEMO_QUERY_SAMPLES,
        &linear_metrics
    );
    _evaluate_prototypes(
        &prototype_head,
        query_embeddings,
        ecg_demo_query_labels,
        ECG_DEMO_QUERY_SAMPLES,
        &prototype_metrics
    );

    printf("\r\nECG head adaptation demo\r\n");
    _print_metric_line("pre_linear", &pre_metrics);
    _print_metric_line("post_linear_sgd", &linear_metrics);
    _print_metric_line("post_prototype", &prototype_metrics);
}

static void _run_batch_benchmark(void) {
    static float support_embeddings[ECG_BENCH_MAX_SUPPORT_SAMPLES * EMBEDDING_DIM];
    static float query_embedding[EMBEDDING_DIM];
    binary_metrics_t agg_pre;
    binary_metrics_t agg_linear;
    binary_metrics_t agg_proto;
    uint32_t sum_pre_acc_milli = 0U;
    uint32_t sum_pre_f1_milli = 0U;
    uint32_t sum_sgd_acc_milli = 0U;
    uint32_t sum_sgd_f1_milli = 0U;
    uint32_t sum_proto_acc_milli = 0U;
    uint32_t sum_proto_f1_milli = 0U;
    uint32_t total_support = 0U;
    uint32_t total_query = 0U;
    uint32_t batch_start = 0U;
    uint32_t batch_end = 0U;
    uint32_t batch_total_cyc = 0U;

    _metrics_reset(&agg_pre);
    _metrics_reset(&agg_linear);
    _metrics_reset(&agg_proto);
    g_batch_results.magic = 0U;
    g_batch_results.episodes = 0U;
    g_batch_results.total_support = 0U;
    g_batch_results.total_query = 0U;
    g_batch_results.batch_total_cyc = 0U;
    g_batch_results.avg_episode_cyc = 0U;
    g_batch_results.pre_acc_milli = 0U;
    g_batch_results.pre_f1_macro_milli = 0U;
    g_batch_results.sgd_acc_milli = 0U;
    g_batch_results.sgd_f1_macro_milli = 0U;
    g_batch_results.proto_acc_milli = 0U;
    g_batch_results.proto_f1_macro_milli = 0U;

    dwt_init();
    batch_start = dwt_count();

    printf("\r\nECG on-device batch benchmark\r\n");
    printf(
        "BATCH_CFG shots=%u episodes=%u total_support=%u total_query=%u max_query=%u\r\n",
        (unsigned int)ECG_BENCH_SHOTS,
        (unsigned int)ECG_BENCH_EPISODES,
        (unsigned int)ECG_BENCH_TOTAL_SUPPORT_SAMPLES,
        (unsigned int)ECG_BENCH_TOTAL_QUERY_SAMPLES,
        (unsigned int)ECG_BENCH_MAX_QUERY_SAMPLES
    );

    for (uint32_t episode = 0U; episode < (uint32_t)ECG_BENCH_EPISODES; ++episode) {
        uint32_t support_offset = ecg_bench_support_offsets[episode];
        uint32_t support_count = ecg_bench_support_counts[episode];
        uint32_t query_offset = ecg_bench_query_offsets[episode];
        uint32_t query_count = ecg_bench_query_counts[episode];
        linear_head_t pretrained_head;
        linear_head_t adapted_head;
        prototype_head_t prototype_head;
        binary_metrics_t ep_pre;
        binary_metrics_t ep_linear;
        binary_metrics_t ep_proto;

        _metrics_reset(&ep_pre);
        _metrics_reset(&ep_linear);
        _metrics_reset(&ep_proto);

        linear_head_load(&pretrained_head, ecg_head_weight, ecg_head_bias);
        linear_head_copy(&adapted_head, &pretrained_head);
        prototype_head_reset(&prototype_head);

        for (uint32_t i = 0U; i < support_count; ++i) {
            uint32_t global_idx = support_offset + i;
            ecg_backbone_infer(
                &ecg_bench_support[global_idx * ECG_BENCH_SIGNAL_LENGTH],
                &support_embeddings[i * EMBEDDING_DIM]
            );
            prototype_head_update(
                &prototype_head,
                &support_embeddings[i * EMBEDDING_DIM],
                (int)ecg_bench_support_labels[global_idx]
            );
        }

        linear_head_adapt(
            &adapted_head,
            support_embeddings,
            &ecg_bench_support_labels[support_offset],
            support_count,
            50U,
            0.05f
        );

        for (uint32_t i = 0U; i < query_count; ++i) {
            uint32_t global_idx = query_offset + i;
            uint8_t label = ecg_bench_query_labels[global_idx];
            ecg_backbone_infer(
                &ecg_bench_query[global_idx * ECG_BENCH_SIGNAL_LENGTH],
                query_embedding
            );
            _metrics_update(&ep_pre, linear_head_predict(&pretrained_head, query_embedding), (int)label);
            _metrics_update(&ep_linear, linear_head_predict(&adapted_head, query_embedding), (int)label);
            _metrics_update(&ep_proto, prototype_head_predict(&prototype_head, query_embedding), (int)label);
        }

        _metrics_merge(&agg_pre, &ep_pre);
        _metrics_merge(&agg_linear, &ep_linear);
        _metrics_merge(&agg_proto, &ep_proto);
        sum_pre_acc_milli += _to_milli(_metrics_accuracy(&ep_pre));
        sum_pre_f1_milli += _to_milli(_metrics_f1_macro(&ep_pre));
        sum_sgd_acc_milli += _to_milli(_metrics_accuracy(&ep_linear));
        sum_sgd_f1_milli += _to_milli(_metrics_f1_macro(&ep_linear));
        sum_proto_acc_milli += _to_milli(_metrics_accuracy(&ep_proto));
        sum_proto_f1_milli += _to_milli(_metrics_f1_macro(&ep_proto));
        total_support += support_count;
        total_query += query_count;

        _print_batch_episode_line(
            ecg_bench_record_ids[episode],
            support_count,
            query_count,
            &ep_pre,
            &ep_linear,
            &ep_proto
        );
    }

    batch_end = dwt_count();
    batch_total_cyc = batch_end - batch_start;

    g_batch_results.episodes = (uint32_t)ECG_BENCH_EPISODES;
    g_batch_results.total_support = total_support;
    g_batch_results.total_query = total_query;
    g_batch_results.batch_total_cyc = batch_total_cyc;
    g_batch_results.avg_episode_cyc = batch_total_cyc / (uint32_t)ECG_BENCH_EPISODES;
    g_batch_results.pre_acc_milli = sum_pre_acc_milli / (uint32_t)ECG_BENCH_EPISODES;
    g_batch_results.pre_f1_macro_milli = sum_pre_f1_milli / (uint32_t)ECG_BENCH_EPISODES;
    g_batch_results.sgd_acc_milli = sum_sgd_acc_milli / (uint32_t)ECG_BENCH_EPISODES;
    g_batch_results.sgd_f1_macro_milli = sum_sgd_f1_milli / (uint32_t)ECG_BENCH_EPISODES;
    g_batch_results.proto_acc_milli = sum_proto_acc_milli / (uint32_t)ECG_BENCH_EPISODES;
    g_batch_results.proto_f1_macro_milli = sum_proto_f1_milli / (uint32_t)ECG_BENCH_EPISODES;
    g_batch_results.magic = 0xBA7C0001U;

    printf(
        "BATCH_AGG episodes=%lu total_support=%lu total_query=%lu total_cyc=%lu avg_episode_cyc=%lu\r\n",
        (unsigned long)ECG_BENCH_EPISODES,
        (unsigned long)total_support,
        (unsigned long)total_query,
        (unsigned long)batch_total_cyc,
        (unsigned long)(batch_total_cyc / (uint32_t)ECG_BENCH_EPISODES)
    );
    _print_metric_line("batch_pre_linear", &agg_pre);
    _print_metric_line("batch_post_linear_sgd", &agg_linear);
    _print_metric_line("batch_post_prototype", &agg_proto);
}

void ecg_head_adaptation_app_run(void) {
    _uart_init();
    _run_profiling();
    _run_batch_benchmark();
}
