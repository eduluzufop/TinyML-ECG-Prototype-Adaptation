# PSoC6 CY8CPROTO-063-BLE — Knowledge Base

Bootstrap reference for desenvolvimento com **ModusToolbox 3.x** no **CY8CPROTO-063-BLE** em Linux.
Inclui tudo que foi necessário para fazer UART, boot dual-core e memória compartilhada funcionar de verdade.

---

## Hardware

| Item | Detalhe |
|------|---------|
| Kit | CY8CPROTO-063-BLE |
| MCU | CYBLE-416045-02 (PSoC 63) |
| CM4 | Cortex-M4 @ 150 MHz |
| CM0+ | Cortex-M0+ @ 100 MHz |
| Flash | 1 MB (`0x10000000`) |
| SRAM | 288 KB — CM0+: `0x08000000`, CM4: `0x08003000` |
| Porta de programação | **J10 (KitProg3)** — micro-USB próxima ao chip programador |
| UART de debug | P5_0 (RX) / P5_1 (TX) → SCB5 → KitProg3 virtual COM |

---

## Ambiente

| Software | Versão testada | Caminho padrão |
|----------|---------------|----------------|
| ModusToolbox | 3.8 | `/opt/Tools/ModusToolbox/tools_3.8/` |
| GCC ARM (incluso no MTB) | 14.2.1 | `/opt/Tools/mtb-gcc-arm-eabi/14.2.1/` |
| fw-loader | 3.12.0 | Download manual (ver seção abaixo) |
| OpenOCD (incluso no MTB) | 0.12.0 | via `make program` |

---

## 1. Atualizar KitProg2 → KitProg3 (obrigatório uma vez)

Kits novos saem com KitProg2 (`04b4:f147`). O MTB 3.x exige KitProg3 (`04b4:f155`).

```bash
# Verificar versão atual
./fw-loader/bin/fw-loader --device-list

# Atualizar
./fw-loader/bin/fw-loader --update-kp3

# Instalar regras udev (se necessário)
sudo bash ./fw-loader/udev_rules/install_rules.sh
```

Após upgrade confirmar:
```bash
lsusb | grep -i cypress
# Deve mostrar: KitProg3 CMSIS-DAP (04b4:f155)
ls /dev/ttyACM0
```

---

## 2. Criar projeto dual-core MTB

```bash
/opt/Tools/ModusToolbox/tools_3.8/project-creator/project-creator-cli \
  --board-id APP_CY8CPROTO-063-BLE \
  --app-id mtb-example-psoc6-dual-cpu-empty-app \
  --user-app-name psoc6_dual_core_ecg \
  --target-path firmware/psoc6/mtb/build_workspace
```

Estrutura gerada:
```
psoc6_dual_core_ecg/
├── proj_cm0p/      ← CM0+ (boot + producer)
├── proj_cm4/       ← CM4 (consumer/inference)
├── shared_inc/     ← headers compartilhados
├── shared_src/     ← fontes compartilhados
└── bsps/           ← BSP gerado (não editar diretamente)
```

---

## 3. Boot dual-core: CM0+ deve acordar o CM4

**O CM4 não inicializa sozinho.** O CM0+ é o core de boot e precisa:
1. Inicializar o BSP
2. Chamar `Cy_SysEnableCM4()` com o endereço de início do CM4

O endereço do CM4 está no linker script do CM0+ (`FLASH_CM0P_SIZE = 0x4400` → CM4 começa em `0x10004400`).

```c
/* CM0+ main.c */
#include "cybsp.h"
#include "system_psoc6.h"
#include <string.h>

#define CM4_APPL_ADDR  (0x10004400u)  /* CM4 flash origin — confirmar no linker map */

int main(void) {
    cybsp_init();                          /* clocks, power, pinos */
    memset(&g_shared_memory, 0, sizeof(g_shared_memory));  /* ver seção 5 */
    Cy_SysEnableCM4(CM4_APPL_ADDR);       /* acorda o CM4 */
    /* loop do CM0+ aqui */
}
```

```c
/* CM4 main.c */
#include "cybsp.h"

int main(void) {
    cybsp_init();
    /* inicializar UART (ver seção 4) */
    /* loop do CM4 aqui */
}
```

---

## 4. UART via HAL (abordagem recomendada)

Use o **MTB HAL** (`cyhal_uart_init`) — ele gerencia o clock do SCB automaticamente. Abordagem PDL direta falhou silenciosamente por falta de configuração do divisor de clock.

```c
#include "cybsp.h"
#include "cyhal_uart.h"
#include <stdio.h>
#include <unistd.h>

static cyhal_uart_t g_uart;

static void uart_init(void) {
    static const cyhal_uart_cfg_t cfg = {
        .data_bits      = 8U,
        .stop_bits      = 1U,
        .parity         = CYHAL_UART_PARITY_NONE,
        .rx_buffer      = NULL,
        .rx_buffer_size = 0U,
    };
    /* CYBSP_DEBUG_UART_TX = P5_1, CYBSP_DEBUG_UART_RX = P5_0 (BSP do CY8CPROTO-063-BLE) */
    cyhal_uart_init(&g_uart, CYBSP_DEBUG_UART_TX, CYBSP_DEBUG_UART_RX,
                    NC, NC, NULL, &cfg);
    cyhal_uart_set_baud(&g_uart, 115200U, NULL);
}

/* Retarget printf → UART */
int _write(int fd, const char *buf, int count) {
    (void)fd;
    size_t len = (size_t)count;
    cyhal_uart_write(&g_uart, (void *)buf, &len);
    return (int)len;
}

int main(void) {
    cybsp_init();
    uart_init();
    printf("\r\nHello from CM4\r\n");
}
```

**Monitor UART:**
```bash
python -m serial.tools.miniterm /dev/ttyACM0 115200
# Sair: Ctrl+]
```

### Por que PDL falhou

Ao usar `Cy_SCB_UART_Init()` diretamente, é necessário configurar o divisor de clock do periférico manualmente antes:
```c
/* PeriClk = 100 MHz; baud 115200; oversample 12 → divisor = 71 */
Cy_SysClk_PeriphAssignDivider(PCLK_SCB5_CLOCK, CY_SYSCLK_DIV_8_BIT, 0U);
Cy_SysClk_PeriphSetDivider(CY_SYSCLK_DIV_8_BIT, 0U, 71U);
Cy_SysClk_PeriphEnableDivider(CY_SYSCLK_DIV_8_BIT, 0U);
```
O HAL faz isso automaticamente — prefira o HAL.

---

## 5. Memória compartilhada entre CM0+ e CM4

O SRAM do PSoC 6 é particionado: CM0+ usa `0x08000000` e CM4 usa `0x08003000`. **Cada core tem sua própria cópia de variáveis globais** — não compartilham por padrão.

### Solução: seção `.ecg_shared` em endereço fixo

**Passo 1 — Adicionar região `shared_ram` nos dois linker scripts:**

`COMPONENT_CM0P/TOOLCHAIN_GCC_ARM/linker.ld`:
```ld
ram        (rwx) : ORIGIN = 0x08000000, LENGTH = 0x02000  /* reduzir de 0x3000 */
shared_ram (rwx) : ORIGIN = 0x08002000, LENGTH = 0x01000  /* 4 KB compartilhados */
flash      (rx)  : ORIGIN = 0x10000000, LENGTH = 0x4400
```

`COMPONENT_CM4/TOOLCHAIN_GCC_ARM/linker.ld`:
```ld
ram        (rwx) : ORIGIN = 0x08003000, LENGTH = 0x044800
shared_ram (rwx) : ORIGIN = 0x08002000, LENGTH = 0x01000  /* mesmo endereço */
flash      (rx)  : ORIGIN = 0x10000000, LENGTH = 0x100000
```

**Passo 2 — Adicionar seção em endereço absoluto nos dois linker scripts:**
```ld
/* Após as seções existentes, em AMBOS os linker scripts */
.ecg_shared 0x08002100 (NOLOAD):
{
    KEEP(*(.ecg_shared))
} > shared_ram
```

> O offset `0x08002100` evita os símbolos IPC do BSP que ocupam `0x08002000–0x080020FF`.

**Passo 3 — Marcar a variável compartilhada:**
```c
/* shared_protocol.c */
shared_memory_t g_shared_memory __attribute__((section(".ecg_shared")));
```

**Passo 4 — Zeragem explícita no CM0+ (seção NOLOAD não é zerada pelo startup):**
```c
memset(&g_shared_memory, 0, sizeof(g_shared_memory));
```

**Verificar se o endereço está igual nos dois ELFs:**
```bash
arm-none-eabi-nm proj_cm0p.elf | grep g_shared_memory
arm-none-eabi-nm proj_cm4.elf  | grep g_shared_memory
# Ambos devem mostrar: 08002100 B g_shared_memory
```

### Por que `.cy_sharedmem` do BSP não funcionou diretamente

O BSP já usa `.cy_sharedmem` para seus próprios símbolos IPC. O número de símbolos difere entre CM0+ e CM4, então `g_shared_memory` cai em offsets diferentes nos dois cores. Criar uma seção própria com endereço absoluto é a solução correta.

---

## 6. Build e flash

```bash
cd firmware/psoc6/mtb/build_workspace/psoc6_dual_core_ecg

make build      # compila CM0+ e CM4, gera app_combined.hex
make program    # erase + write + verify + reset
make qprogram   # write + verify + reset (sem erase completo, mais rápido)
```

Saída esperada do build:
```
Total FLASH (Available): 1.048.576
Total FLASH (Utilized) : ~43.000
Build complete
```

Saída esperada do flash:
```
** Verified OK **
** Resetting Target **
```

---

## 7. Checklist de debug

| Sintoma | Causa provável | Solução |
|---------|---------------|---------|
| `unable to find a matching CMSIS-DAP device` | KitProg2 não atualizado | `fw-loader --update-kp3` |
| UART abre mas nada aparece | CM4 não inicializado | CM0+ precisa chamar `Cy_SysEnableCM4()` |
| UART silenciosa mesmo com CM4 vivo | Clock do SCB não configurado | Usar `cyhal_uart_init` (HAL) |
| Dados fluem mas lógica parece errada | CM0+/CM4 com cópias separadas da struct | Usar seção `.ecg_shared` em endereço fixo (ver seção 5) |
| `seq=N` repetindo sem incrementar | Shared memory não está realmente compartilhada | Confirmar endereço igual nos dois ELFs com `nm` |
| Output garbled no miniterm | `\r` sem `\n` equivalente no terminal | Usar `\r\n` no printf ou `--eol LF` no miniterm |

---

## 8. Referências rápidas

```bash
# Verificar kit conectado
lsusb | grep -i cypress

# Verificar endereços da shared memory
/opt/Tools/mtb-gcc-arm-eabi/14.2.1/gcc/bin/arm-none-eabi-nm \
  build/APP_CY8CPROTO-063-BLE/Debug/proj_cm0p.elf | grep shared
/opt/Tools/mtb-gcc-arm-eabi/14.2.1/gcc/bin/arm-none-eabi-nm \
  build/APP_CY8CPROTO-063-BLE/Debug/proj_cm4.elf | grep shared

# Monitor UART
python -m serial.tools.miniterm /dev/ttyACM0 115200

# Instalar pyserial se necessário
pip install pyserial
```
