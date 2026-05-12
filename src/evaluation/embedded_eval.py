from __future__ import annotations


def parse_uart_log(log_text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in log_text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            try:
                out[k.strip()] = float(v.strip())
            except ValueError:
                pass
    return out
