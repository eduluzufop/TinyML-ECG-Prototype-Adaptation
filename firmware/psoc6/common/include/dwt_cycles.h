#pragma once
#include <stdint.h>

/*
 * Cortex-M4 DWT cycle counter utility.
 * Registers are memory-mapped at fixed ARM architecture addresses.
 */

#define DWT_DEMCR_REG   (*(volatile uint32_t *)0xE000EDFC)
#define DWT_CYCCNT_REG  (*(volatile uint32_t *)0xE0001004)
#define DWT_CTRL_REG    (*(volatile uint32_t *)0xE0001000)

/* Use CMSIS definitions if available, else fall back to literal values */
#ifndef DWT_DEMCR_TRCENA_Msk
#define DWT_DEMCR_TRCENA_Msk    (1UL << 24)
#endif
#ifndef DWT_CTRL_CYCCNTENA_Msk
#define DWT_CTRL_CYCCNTENA_Msk  (1UL << 0)
#endif

static inline void dwt_init(void) {
    DWT_DEMCR_REG  |= DWT_DEMCR_TRCENA_Msk;
    DWT_CYCCNT_REG  = 0U;
    DWT_CTRL_REG   |= DWT_CTRL_CYCCNTENA_Msk;
}

static inline uint32_t dwt_count(void) {
    return DWT_CYCCNT_REG;
}
