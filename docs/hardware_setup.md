# Hardware Setup

> Para referência completa de bootstrap (UART, dual-core, shared memory), consulte [`docs/PSOC6-063-BLE.md`](PSOC6-063-BLE.md).

## Placa utilizada

- **CY8CPROTO-063-BLE** (PSoC 6 BLE Prototyping Kit)
  - MCU: CYBLE-416045-02 (PSoC 63 — Cortex-M4 @ 150 MHz + Cortex-M0+ @ 100 MHz)
  - 1 MB Flash, 288 KB SRAM
  - KitProg3 integrado (programador/debugger on-board)
  - BSP ModusToolbox: `APP_CY8CPROTO-063-BLE`

---

## Pré-requisitos

| Software | Versão testada |
|----------|----------------|
| ModusToolbox | 3.8 |
| GCC ARM Embedded (incluso no MTB) | — |
| fw-loader (Infineon Firmware Loader) | 3.12.0 |
| Python | 3.9+ |

**ModusToolbox** deve estar instalado em `/opt/Tools/ModusToolbox/tools_3.8/`.
O `project-creator-cli`, `make`, `ninja` e `openocd` são providos pelo próprio MTB — não é necessário instalá-los separadamente.

---

## 1. Conexão física

1. Use um **cabo USB com dados** (não apenas carga).
2. Conecte na porta **J10 (KitProg3)** — é a micro-USB próxima ao chip programador, **não** a J2 (Target USB).
3. O LED do KitProg deve acender (âmbar ou verde dependendo do modo).

Verifique no Linux:

```bash
lsusb | grep -i cypress
# Esperado: Cypress KitProg3 CMSIS-DAP (04b4:f155)

ls /dev/ttyACM*
# Esperado: /dev/ttyACM0
```

---

## 2. Atualizar firmware do KitProg (KitProg2 → KitProg3)

Kits que saem de fábrica com KitProg2 (`04b4:f147`) precisam ser atualizados.
O ModusToolbox exige KitProg3 (`04b4:f155`) para programação via CMSIS-DAP.

Baixe o fw-loader em: https://github.com/Infineon/Firmware-loader/releases

```bash
# Verificar versão atual do programador
./fw-loader/bin/fw-loader --device-list

# Atualizar para KitProg3
./fw-loader/bin/fw-loader --update-kp3
```

Após o upgrade, confirme:

```bash
lsusb | grep -i cypress
# Deve mostrar: Cypress Semiconductor Corp. KitProg3 CMSIS-DAP (04b4:f155)
```

### Regras udev (se o dispositivo aparecer sem permissão de acesso)

```bash
sudo bash ./fw-loader/udev_rules/install_rules.sh
# Reconecte o cabo após instalar
```

---

## 3. Projeto ModusToolbox

O projeto dual-core já está criado em:

```
firmware/psoc6/mtb/build_workspace/psoc6_dual_core_ecg/
├── proj_cm0p/   ← CM0+ (replay/producer)
├── proj_cm4/    ← CM4 (inference/consumer)
├── shared_inc/  ← shared_protocol.h
└── shared_src/  ← shared_protocol.c
```

Se precisar recriar o projeto do zero:

```bash
/opt/Tools/ModusToolbox/tools_3.8/project-creator/project-creator-cli \
  --board-id APP_CY8CPROTO-063-BLE \
  --app-id mtb-example-psoc6-dual-cpu-empty-app \
  --user-app-name psoc6_dual_core_ecg \
  --target-path firmware/psoc6/mtb/build_workspace
```

---

## 4. Build

```bash
cd firmware/psoc6/mtb/build_workspace/psoc6_dual_core_ecg
make build
```

Compila CM0+ e CM4 e gera `app_combined.hex` com o mapa de memória consolidado.

Saída esperada ao final:

```
Total FLASH (Available): 1.048.576
Total FLASH (Utilized) : 18.384
Build complete
```

---

## 5. Gravar o firmware

```bash
# Programação completa (erase + write + verify + reset)
make program

# Re-programação rápida (sem erase completo — para iteração)
make qprogram
```

O OpenOCD grava `app_combined.hex` em ambos os núcleos em uma única operação.

Saída esperada ao final:

```
** Verified OK **
** Resetting Target **
```

---

## 6. Monitorar saída UART

O CM4 envia métricas por UART virtual (115200 8N1) via KitProg3:

```bash
python -m serial.tools.miniterm /dev/ttyACM0 115200
```

Formato de cada linha de log:

```
seq=<N> label=<0|1> pred=<0|1> e2e_us=<us> infer_us=<us>
```

---

## Observações práticas

- Registre versão do MTB, BSP e toolchain nos logs de experimento.
- Para medição de energia: preparar shunt externo (fase futura).
- O projeto está configurado em modo **Debug** por padrão. Para release: `make build CONFIG=Release`.
