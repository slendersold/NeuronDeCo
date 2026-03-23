#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mne
import numpy as np
from pylsl import (
    StreamInfo,
    StreamInlet,
    StreamOutlet,
    local_clock,
    resolve_byprop,
)


def _edf_field(text: str, width: int) -> bytes:
    return text[:width].ljust(width).encode("ascii", errors="ignore")


def write_edf_manual(*, data: np.ndarray, ch_names: list[str], sfreq: int, out_path: Path) -> None:
    """
    Minimal EDF writer (16-bit) without external EDF libraries.
    data shape: (n_channels, n_samples), values in Volts.
    """
    n_channels, n_samples = data.shape
    if len(ch_names) != n_channels:
        raise ValueError("len(ch_names) must match number of channels")
    if n_samples % sfreq != 0:
        raise ValueError("n_samples must be divisible by sfreq")

    n_records = n_samples // sfreq
    dmin, dmax = -32768, 32767
    pmins, pmaxs = [], []
    for ch in range(n_channels):
        x = data[ch]
        m = max(abs(float(np.min(x))), abs(float(np.max(x))), 1e-6)
        pmins.append(-1.2 * m)
        pmaxs.append(1.2 * m)

    header_bytes = 256 + n_channels * 256
    h = bytearray()
    h += _edf_field("0", 8)
    h += _edf_field("MOCK_PATIENT", 80)
    h += _edf_field("MOCK_LSL_RECORDING", 80)
    h += _edf_field("01.01.26", 8)
    h += _edf_field("01.01.01", 8)
    h += _edf_field(str(header_bytes), 8)
    h += _edf_field("", 44)
    h += _edf_field(str(n_records), 8)
    h += _edf_field("1", 8)  # duration of a data record in seconds
    h += _edf_field(str(n_channels), 4)

    for name in ch_names:
        h += _edf_field(name, 16)
    for _ in ch_names:
        h += _edf_field("", 80)
    for _ in ch_names:
        h += _edf_field("V", 8)
    for v in pmins:
        h += _edf_field(f"{v:.8g}", 8)
    for v in pmaxs:
        h += _edf_field(f"{v:.8g}", 8)
    for _ in ch_names:
        h += _edf_field(str(dmin), 8)
    for _ in ch_names:
        h += _edf_field(str(dmax), 8)
    for _ in ch_names:
        h += _edf_field("", 80)
    for _ in ch_names:
        h += _edf_field(str(sfreq), 8)
    for _ in ch_names:
        h += _edf_field("", 32)

    if len(h) != header_bytes:
        raise RuntimeError(f"Bad EDF header size: {len(h)} != {header_bytes}")

    with open(out_path, "wb") as f:
        f.write(h)
        for r in range(n_records):
            s0 = r * sfreq
            s1 = s0 + sfreq
            for ch in range(n_channels):
                x = data[ch, s0:s1]
                pmin, pmax = pmins[ch], pmaxs[ch]
                scale = (dmax - dmin) / (pmax - pmin)
                dig = np.rint((x - pmin) * scale + dmin).astype(np.int32)
                dig = np.clip(dig, dmin, dmax).astype("<i2")
                f.write(dig.tobytes(order="C"))


def build_signal_sample(t_s: float, ch: int) -> float:
    # Per-channel sinusoid with channel-specific frequency/amplitude.
    f = 6.0 + 4.5 * ch
    a = (12.0 + 2.5 * ch) * 1e-6  # Volts
    return float(a * np.sin(2.0 * np.pi * f * t_s))


def produce_and_capture_lsl(
    *,
    sfreq: int = 2000,
    duration_sec: int = 60,
    n_channels: int = 7,
    marker_every_sec: float = 3.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce synthetic EEG + marker streams over LSL and capture them with inlets.

    Returns:
      eeg_data: (n_channels, n_samples)
      eeg_ts:   (n_samples,)
      marker_events: (n_events, 2) with columns [timestamp, code]
    """
    rng = np.random.default_rng(seed)

    # Outlets
    eeg_info = StreamInfo("MockEEG", "EEG", n_channels, sfreq, "float32", "mock-eeg-uid-001")
    marker_info = StreamInfo("MockMarkers", "Markers", 1, 0.0, "string", "mock-markers-uid-001")
    eeg_out = StreamOutlet(eeg_info, chunk_size=1, max_buffered=360)
    marker_out = StreamOutlet(marker_info, chunk_size=1, max_buffered=360)

    # Resolve + inlets
    eeg_streams = resolve_byprop("name", "MockEEG", timeout=3.0)
    marker_streams = resolve_byprop("name", "MockMarkers", timeout=3.0)
    if not eeg_streams or not marker_streams:
        raise RuntimeError("Could not resolve LSL streams (MockEEG/MockMarkers).")
    eeg_in = StreamInlet(eeg_streams[0], max_buflen=360)
    marker_in = StreamInlet(marker_streams[0], max_buflen=360)
    eeg_in.open_stream(timeout=3.0)
    marker_in.open_stream(timeout=3.0)

    # Ensure outlets see connected consumers before publishing.
    t_wait = time.time() + 3.0
    while (not eeg_out.have_consumers() or not marker_out.have_consumers()) and time.time() < t_wait:
        time.sleep(0.01)

    n_samples = sfreq * duration_sec
    t0 = local_clock()

    # Publish quickly with explicit timestamps (virtual real-time timeline).
    for i in range(n_samples):
        t_rel = i / sfreq
        ts = t0 + t_rel

        sample = [build_signal_sample(t_rel, ch) + 2e-6 * rng.standard_normal() for ch in range(n_channels)]
        eeg_out.push_sample(sample, ts)

        # Marker pulse on alternating codes 1/9 every marker_every_sec.
        if i % int(marker_every_sec * sfreq) == 0 and i > int(1.0 * sfreq):
            k = i // int(marker_every_sec * sfreq)
            code = "1" if (k % 2 == 0) else "9"
            marker_out.push_sample([code], ts)

    # Give inlets a short moment to drain network buffer.
    time.sleep(0.2)

    # Pull EEG chunks
    eeg_samples = []
    eeg_ts = []
    while True:
        chunk, ts = eeg_in.pull_chunk(timeout=0.05, max_samples=4096)
        if not ts:
            break
        eeg_samples.extend(chunk)
        eeg_ts.extend(ts)

    # Pull markers
    marker_events = []
    while True:
        smp, ts = marker_in.pull_sample(timeout=0.02)
        if ts is None:
            break
        try:
            code = int(smp[0])
        except Exception:
            continue
        if code in (1, 9):
            marker_events.append((ts, code))

    if len(eeg_samples) == 0:
        raise RuntimeError("No EEG samples captured from LSL inlet.")
    if len(marker_events) == 0:
        raise RuntimeError("No marker events captured from LSL inlet.")

    eeg_arr = np.asarray(eeg_samples, dtype=np.float64).T  # (C, T)
    eeg_ts_arr = np.asarray(eeg_ts, dtype=np.float64)
    marker_arr = np.asarray(marker_events, dtype=np.float64)  # (N,2): ts, code
    return eeg_arr, eeg_ts_arr, marker_arr


def build_raw_and_events(
    eeg_data: np.ndarray,
    eeg_ts: np.ndarray,
    marker_events: np.ndarray,
    sfreq: int,
) -> tuple[mne.io.RawArray, np.ndarray, dict[str, int]]:
    # Convert marker timestamps to nearest sample indices in captured EEG timeline.
    marker_ts = marker_events[:, 0]
    marker_codes = marker_events[:, 1].astype(int)
    sample_idx = np.searchsorted(eeg_ts, marker_ts, side="left")
    sample_idx = np.clip(sample_idx, 0, eeg_data.shape[1] - 1)

    events = np.column_stack([sample_idx, np.zeros_like(sample_idx), marker_codes]).astype(int)
    # Keep only unique sample indices (first marker if duplicate index appears)
    _, uniq_pos = np.unique(events[:, 0], return_index=True)
    events = events[np.sort(uniq_pos)]

    ch_names = [f"EEG{i:02d}" for i in range(1, eeg_data.shape[0] + 1)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * len(ch_names))
    raw = mne.io.RawArray(eeg_data, info, verbose=False)
    event_id = {"negative": 1, "positive": 9}
    return raw, events, event_id


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic LSL EEG+markers and build EDF+TFR.")
    ap.add_argument("--out-dir", default="/home/user/Desktop/neurondeco_mock_data", help="Output dir outside repo")
    ap.add_argument("--subject", default="s01")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sfreq", type=int, default=2000)
    ap.add_argument("--duration-sec", type=int, default=60)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    eeg_data, eeg_ts, marker_events = produce_and_capture_lsl(
        sfreq=args.sfreq, duration_sec=args.duration_sec, seed=args.seed
    )
    raw, events, event_id = build_raw_and_events(eeg_data, eeg_ts, marker_events, sfreq=args.sfreq)

    edf_path = out_dir / f"mock_{args.subject}.edf"
    tfr_path = out_dir / f"tfr_{args.subject}.fif"

    ch_names = [f"EEG{i:02d}" for i in range(1, eeg_data.shape[0] + 1)]
    write_edf_manual(data=eeg_data, ch_names=ch_names, sfreq=args.sfreq, out_path=edf_path)

    # Re-read from EDF and use the same events mapped from LSL timestamps.
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=-0.2,
        tmax=1.9,
        baseline=(0.0, 0.1),
        preload=True,
        reject=None,
        verbose=False,
    )
    freqs = np.linspace(1.0, 60.0, 80)
    n_cycles = freqs / 2.0
    tfr = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=False,
        average=False,
        decim=10,
        verbose=False,
    )
    tfr.save(tfr_path, overwrite=True)

    print(f"Saved EDF: {edf_path}")
    print(f"Saved TFR: {tfr_path}")
    print(f"EEG samples: {eeg_data.shape[1]}, channels: {eeg_data.shape[0]}, sfreq: {args.sfreq}")
    print(f"Events: {events.shape[0]}, unique codes: {np.unique(events[:,2])}")
    print(f"TFR shape: {tfr.data.shape}")


if __name__ == "__main__":
    main()

