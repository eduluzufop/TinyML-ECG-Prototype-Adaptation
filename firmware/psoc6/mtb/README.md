# Notas ModusToolbox

Esta pasta agora contém o workspace MTB real em `build_workspace/psoc6_dual_core_ecg`.

Resumo da integração atual:
- `CM0+`: boot e wake-up do `CM4`
- `CM4`: nesta configuração atual executa a demo gráfica `demoECG` em `firmware/psoc6/app/source/demo_ecg_app.c`
- `Backbone`: inferência congelada em C usando `firmware/psoc6/generated/model_params.h`
- `Head adaptation on-device`: protótipos e `SGD` leve em `W,b`
- `ml-tflite-micro`: asset registrado no projeto `proj_cm4` para a migração do backend do backbone para o runtime oficial Infineon

## demoECG para UI

O firmware `demoECG` transmite pela UART uma sessão contínua em `JSON Lines`, adequada para a UI em `./ui`:
- `session`: metadados do replay
- `beat`: janela do ECG, fase (`support` ou `query`), rótulo verdadeiro e saídas `pre` / `sgd` / `proto`
- `summary`: acurácia agregada da sessão demo

Fluxo:
```bash
cd firmware/psoc6/mtb/build_workspace/psoc6_dual_core_ecg/proj_cm4
make build_proj CONFIG=Release
make program_proj CONFIG=Release
python3 ../../../../../ui/demo_ecg_monitor.py
```

Semântica dos resultados impressos na UART:
- `pre_linear`: inferência on-device com a cabeça global, sem adaptação local.
- `post_linear_sgd`: adaptação on-device da última camada com `SGD` apenas em `W,b`.
- `post_prototype`: adaptação on-device por protótipos no espaço de embeddings.

Validação em hardware já executada no `CY8CPROTO-063-BLE`:
- `pre_linear acc=0.938 f1=0.000`
- `post_linear_sgd acc=0.500 f1=0.200`
- `post_prototype acc=0.375 f1=0.167`

Esses números validam o fluxo embarcado fim a fim, mas não devem ser tratados como resultado final do artigo. Eles vêm do `demo_replay.h` exportado para teste rápido de integração.

Artefatos gerados esperados:
- `firmware/psoc6/generated/model_params.h`
- `firmware/psoc6/generated/demo_replay.h`
- `firmware/psoc6/generated/benchmark_replay.h`

## Benchmark on-device em lote

O firmware também suporta um benchmark agregado embarcado com múltiplos registros `DS2`.

Fluxo reproduzível:
```bash
PYTHONPATH=. python3 scripts/export_firmware_benchmark_data.py
PYTHONPATH=. python3 scripts/evaluate_firmware_benchmark_reference.py
cd firmware/psoc6/mtb/build_workspace/psoc6_dual_core_ecg/proj_cm4
make build_proj CONFIG=Release
make program_proj CONFIG=Release
```

Resumo do cenário atualmente usado no paper:
- `1-shot`
- `18` episódios `DS2`
- `32` queries por registro
- `576` query beats no total

Arquivos associados:
- [`firmware/psoc6/generated/benchmark_replay.h`](../../../generated/benchmark_replay.h)
- [`results/firmware/ondevice_batch_1shot.json`](../../../../results/firmware/ondevice_batch_1shot.json)
- [`results/firmware/ondevice_batch_1shot_reference.json`](../../../../results/firmware/ondevice_batch_1shot_reference.json)
- [`results/firmware/ondevice_batch_1shot_device_summary.json`](../../../../results/firmware/ondevice_batch_1shot_device_summary.json)

## Profiling DWT da arquitetura medium

Para levar a variante `medium` ao hardware:
```bash
PYTHONPATH=. python3 scripts/export_model.py --checkpoint artifacts/checkpoints/architecture_sweep/medium.pt --out_dir firmware/psoc6/generated
cd firmware/psoc6/mtb/build_workspace/psoc6_dual_core_ecg/proj_cm4
make build_proj CONFIG=Release
make program_proj CONFIG=Release
```

O firmware atual executa `_run_profiling()` antes do benchmark em lote e preenche `g_prof_results`.

Resumo já medido no `CY8CPROTO-063-BLE`:
- inferência por beat: `24.21 ms`
- protótipos `K=1/5/10`: `48.47 / 242.24 / 484.45 ms`
- `SGD` `K=1/5/10`: `50.52 / 252.54 / 505.06 ms`

Artefato:
- [`results/firmware/medium_profile.json`](../../../../results/firmware/medium_profile.json)
