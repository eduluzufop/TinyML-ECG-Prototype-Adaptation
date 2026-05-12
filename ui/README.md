# UI demoECG

Ferramenta gráfica em Python para demonstrar o firmware `demoECG` no `CY8CPROTO-063-BLE`.

O firmware transmite pela UART uma sequência de batimentos em `JSON Lines`:
- fase `support`: batimentos usados para adaptação
- fase `query`: batimentos avaliados depois da adaptação
- rótulo verdadeiro: `normal` ou `arrhythmic`
- saídas on-device:
  - `Pre`
  - `Linear SGD`
  - `Prototype`

## Dependências

```bash
python3 -m pip install pyserial matplotlib numpy
```

## Rodar a UI

```bash
python3 ui/demo_ecg_monitor.py \
  --port /dev/serial/by-id/usb-Cypress_Semiconductor_KitProg3_CMSIS-DAP_0F1902F302098400-if02
```

## Captura headless

Útil para validar serial e gerar um frame estático da demo:

```bash
python3 ui/demo_ecg_monitor.py \
  --headless-capture \
  --max-beats 6 \
  --timeout 20 \
  --out-png ui/artifacts/demo_capture.png \
  --out-json ui/artifacts/demo_capture.json
```

## Firmware demoECG

O workspace MTB foi configurado para compilar a demo gráfica com:
- [`firmware/psoc6/app/source/demo_ecg_app.c`](../firmware/psoc6/app/source/demo_ecg_app.c)
- [`firmware/psoc6/generated/demo_replay.h`](../firmware/psoc6/generated/demo_replay.h)

Build e gravação:

```bash
cd firmware/psoc6/mtb/build_workspace/psoc6_dual_core_ecg/proj_cm4
make build_proj CONFIG=Release
make program_proj CONFIG=Release
```
