#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import serial


NORMAL_COLOR = "#157f6b"
ARR_COLOR = "#c44536"
PRE_COLOR = "#6c757d"
SGD_COLOR = "#1f77b4"
PROTO_COLOR = "#f28e2b"
BG_COLOR = "#f6f3ea"
GRID_COLOR = "#d9d2c3"
TEXT_COLOR = "#1d1d1b"


@dataclass
class BeatEvent:
    session: int
    beat_index: int
    phase: str
    true_label: int
    true_name: str
    samples: np.ndarray
    pre_pred: int | None
    sgd_pred: int | None
    proto_pred: int | None
    pre_name: str | None
    sgd_name: str | None
    proto_name: str | None


@dataclass
class SummaryEvent:
    session: int
    pre_acc: float
    sgd_acc: float
    proto_acc: float


def label_color(label: int | None) -> str:
    if label is None:
        return PRE_COLOR
    return NORMAL_COLOR if int(label) == 0 else ARR_COLOR


def label_text(name: str | None) -> str:
    if not name:
        return "--"
    return "Normal" if name == "normal" else "Arrhythmic"


def parse_event(payload: dict[str, Any]) -> BeatEvent | SummaryEvent | None:
    event_type = payload.get("type")
    if event_type == "beat":
        if "samples_milli" in payload:
            samples = np.asarray(payload["samples_milli"], dtype=np.float32) / 1000.0
        else:
            samples = np.asarray(payload["samples"], dtype=np.float32)
        return BeatEvent(
            session=int(payload["session"]),
            beat_index=int(payload["beat_index"]),
            phase=str(payload["phase"]),
            true_label=int(payload["true_label"]),
            true_name=str(payload["true_name"]),
            samples=samples,
            pre_pred=payload.get("pre_pred"),
            sgd_pred=payload.get("sgd_pred"),
            proto_pred=payload.get("proto_pred"),
            pre_name=payload.get("pre_name"),
            sgd_name=payload.get("sgd_name"),
            proto_name=payload.get("proto_name"),
        )
    if event_type == "summary":
        return SummaryEvent(
            session=int(payload["session"]),
            pre_acc=float(payload["pre_acc"]),
            sgd_acc=float(payload["sgd_acc"]),
            proto_acc=float(payload["proto_acc"]),
        )
    return None


class SerialBeatReader(threading.Thread):
    def __init__(self, port: str, baudrate: int, out_queue: queue.Queue[Any]) -> None:
        super().__init__(daemon=True)
        self._port = port
        self._baudrate = baudrate
        self._out_queue = out_queue
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        pending_meta: dict[tuple[int, int], dict[str, Any]] = {}
        with serial.Serial(self._port, self._baudrate, timeout=0.5) as ser:
            buffer = ""
            while not self._stop.is_set():
                raw = ser.read(ser.in_waiting or 1)
                if not raw:
                    continue
                buffer += raw.decode("utf-8", errors="ignore")
                while True:
                    split_at = -1
                    for delimiter in ("\n", "\r"):
                        idx = buffer.find(delimiter)
                        if idx >= 0 and (split_at < 0 or idx < split_at):
                            split_at = idx
                    if split_at < 0:
                        break
                    line = buffer[:split_at]
                    buffer = buffer[split_at + 1 :]
                    line = line.strip()
                    if not line:
                        continue
                    brace_at = line.find("{")
                    if brace_at >= 0:
                        line = line[brace_at:]
                    elif line.startswith("\"type\""):
                        line = "{" + line
                    else:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    event_type = payload.get("type")
                    if event_type == "beat_meta":
                        key = (int(payload["session"]), int(payload["beat_index"]))
                        pending_meta[key] = payload
                        continue
                    if event_type == "beat_wave":
                        key = (int(payload["session"]), int(payload["beat_index"]))
                        meta = pending_meta.pop(key, None)
                        if meta is None:
                            continue
                        merged = dict(meta)
                        merged["type"] = "beat"
                        merged["samples_milli"] = payload.get("samples_milli", [])
                        event = parse_event(merged)
                    else:
                        event = parse_event(payload)
                    if event is not None:
                        self._out_queue.put(event)


class DemoDashboard:
    def __init__(self) -> None:
        self.last_beat: BeatEvent | None = None
        self.last_summary: SummaryEvent | None = None
        self.history: deque[tuple[str, int]] = deque(maxlen=18)
        self.start_time = time.time()

        self.fig = plt.figure(figsize=(14, 8), facecolor=BG_COLOR)
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1.2, 3.2], width_ratios=[3.2, 1.2])
        self.ax_title = self.fig.add_subplot(gs[0, 0])
        self.ax_preds = self.fig.add_subplot(gs[0, 1])
        self.ax_wave = self.fig.add_subplot(gs[1, 0])
        self.ax_hist = self.fig.add_subplot(gs[1, 1])

        for ax in (self.ax_title, self.ax_preds, self.ax_hist):
            ax.set_axis_off()

        self.ax_wave.set_facecolor("#fffdf7")
        self.ax_wave.grid(True, color=GRID_COLOR, linewidth=0.8, alpha=0.9)
        self.ax_wave.set_xlabel("Sample")
        self.ax_wave.set_ylabel("Amplitude (normalized)")
        self.line, = self.ax_wave.plot([], [], color="#111827", linewidth=2.4)
        self.fill = None

        self.fig.suptitle("PSoC 6 ECG demo monitor", x=0.02, y=0.98, ha="left", va="top", fontsize=20, color=TEXT_COLOR)
        self.fig.tight_layout(rect=(0, 0, 1, 0.96))

    def update(self, beat: BeatEvent | None = None, summary: SummaryEvent | None = None) -> None:
        if beat is not None:
            self.last_beat = beat
            self.history.append((beat.phase, beat.true_label))
        if summary is not None:
            self.last_summary = summary
        self._render()

    def _render(self) -> None:
        self.ax_title.clear()
        self.ax_preds.clear()
        self.ax_hist.clear()
        self.ax_title.set_axis_off()
        self.ax_preds.set_axis_off()
        self.ax_hist.set_axis_off()

        if self.last_beat is None:
            self.ax_title.text(0.0, 0.8, "Waiting for serial data...", fontsize=18, color=TEXT_COLOR, weight="bold")
            self.fig.canvas.draw_idle()
            return

        beat = self.last_beat
        x = np.arange(len(beat.samples))
        self.line.set_data(x, beat.samples)
        self.ax_wave.set_xlim(0, len(beat.samples) - 1)
        ymin = float(np.min(beat.samples)) - 0.5
        ymax = float(np.max(beat.samples)) + 0.5
        self.ax_wave.set_ylim(ymin, ymax)
        if self.fill is not None:
            self.fill.remove()
        self.fill = self.ax_wave.fill_between(
            x,
            beat.samples,
            np.zeros_like(beat.samples),
            color=label_color(beat.true_label),
            alpha=0.12,
        )

        phase_title = "Support beat" if beat.phase == "support" else "Query beat"
        self.ax_title.text(0.0, 0.82, phase_title, fontsize=20, color=TEXT_COLOR, weight="bold")
        self.ax_title.text(
            0.0,
            0.48,
            f"Session {beat.session}  •  Beat {beat.beat_index}",
            fontsize=13,
            color="#4b5563",
        )
        self._pill(self.ax_title, 0.0, 0.12, "Ground truth", label_text(beat.true_name), label_color(beat.true_label))
        elapsed = time.time() - self.start_time
        self.ax_title.text(0.78, 0.82, f"uptime {elapsed:0.1f}s", fontsize=12, color="#6b7280")

        self.ax_preds.text(0.0, 0.85, "On-device outputs", fontsize=16, color=TEXT_COLOR, weight="bold")
        self._pill(self.ax_preds, 0.0, 0.52, "Linear SGD", label_text(beat.sgd_name), SGD_COLOR)
        self._pill(self.ax_preds, 0.0, 0.26, "Prototype", label_text(beat.proto_name), PROTO_COLOR)

        self.ax_hist.text(0.0, 0.95, "Recent beat stream", fontsize=16, color=TEXT_COLOR, weight="bold")
        for idx, (phase, label) in enumerate(self.history):
            y = 0.82 - idx * 0.045
            if y < 0.05:
                break
            marker = "S" if phase == "support" else "Q"
            name = "Normal" if label == 0 else "Arrhythmic"
            self.ax_hist.text(0.02, y, marker, fontsize=12, color="#6b7280", weight="bold")
            self.ax_hist.add_patch(
                mpatches.Rectangle((0.14, y - 0.02), 0.08, 0.032, color=label_color(label), transform=self.ax_hist.transAxes)
            )
            self.ax_hist.text(0.27, y, name, fontsize=11, color=TEXT_COLOR)

        if self.last_summary is not None:
            s = self.last_summary
            self.ax_hist.text(0.0, 0.08, "Last session accuracy", fontsize=14, color=TEXT_COLOR, weight="bold")
            self.ax_hist.text(0.02, -0.02, f"Pre {s.pre_acc:.3f}  |  SGD {s.sgd_acc:.3f}  |  Proto {s.proto_acc:.3f}",
                              fontsize=11, color="#374151")

        self.fig.canvas.draw_idle()

    @staticmethod
    def _pill(ax: plt.Axes, x: float, y: float, title: str, value: str, color: str) -> None:
        ax.text(x, y + 0.10, title, fontsize=11, color="#6b7280", transform=ax.transAxes)
        ax.text(
            x,
            y,
            value,
            fontsize=14,
            color="white",
            transform=ax.transAxes,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": color, "edgecolor": "none"},
        )


def consume_events(event_queue: queue.Queue[Any], dashboard: DemoDashboard) -> None:
    updated = False
    while True:
        try:
            event = event_queue.get_nowait()
        except queue.Empty:
            break
        if isinstance(event, BeatEvent):
            dashboard.update(beat=event)
            updated = True
        elif isinstance(event, SummaryEvent):
            dashboard.update(summary=event)
            updated = True
    if updated:
        dashboard.fig.canvas.draw_idle()


def run_interactive(port: str, baudrate: int) -> None:
    q: queue.Queue[Any] = queue.Queue()
    reader = SerialBeatReader(port, baudrate, q)
    dashboard = DemoDashboard()
    reader.start()

    timer = dashboard.fig.canvas.new_timer(interval=100)

    def on_timer() -> None:
        consume_events(q, dashboard)
        timer.start()

    timer.add_callback(on_timer)
    timer.start()
    plt.show()
    reader.stop()


def run_headless_capture(port: str, baudrate: int, max_beats: int, timeout_s: float, out_png: Path, out_json: Path) -> None:
    plt.switch_backend("Agg")
    q: queue.Queue[Any] = queue.Queue()
    reader = SerialBeatReader(port, baudrate, q)
    dashboard = DemoDashboard()
    reader.start()

    captured_beats: list[dict[str, Any]] = []
    deadline = time.time() + timeout_s
    try:
        while time.time() < deadline and len(captured_beats) < max_beats:
            try:
                event = q.get(timeout=0.5)
            except queue.Empty:
                continue
            if isinstance(event, BeatEvent):
                dashboard.update(beat=event)
                captured_beats.append(
                    {
                        "session": event.session,
                        "beat_index": event.beat_index,
                        "phase": event.phase,
                        "true_name": event.true_name,
                        "pre_name": event.pre_name,
                        "sgd_name": event.sgd_name,
                        "proto_name": event.proto_name,
                    }
                )
            elif isinstance(event, SummaryEvent):
                dashboard.update(summary=event)
        dashboard.fig.savefig(out_png, dpi=160, bbox_inches="tight")
        out_json.write_text(json.dumps({"captured_beats": captured_beats}, indent=2), encoding="utf-8")
    finally:
        reader.stop()


def main() -> None:
    ap = argparse.ArgumentParser(description="Live demo UI for the PSoC 6 ECG replay firmware.")
    ap.add_argument("--port", default="/dev/serial/by-id/usb-Cypress_Semiconductor_KitProg3_CMSIS-DAP_0F1902F302098400-if02")
    ap.add_argument("--baudrate", type=int, default=115200)
    ap.add_argument("--headless-capture", action="store_true")
    ap.add_argument("--max-beats", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--out-png", default="ui/artifacts/demo_capture.png")
    ap.add_argument("--out-json", default="ui/artifacts/demo_capture.json")
    args = ap.parse_args()

    if args.headless_capture:
        out_png = Path(args.out_png)
        out_json = Path(args.out_json)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        run_headless_capture(args.port, args.baudrate, args.max_beats, args.timeout, out_png, out_json)
        return

    run_interactive(args.port, args.baudrate)


if __name__ == "__main__":
    main()
