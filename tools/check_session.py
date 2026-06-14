#!/usr/bin/env python3
"""Sanity-check a recorded session (events.h5 + mocap.h5) for correctness and time alignment.

Checks performed:
  Structural:
    - files exist; attrs sane
    - /events/t and /mocap/t are non-decreasing (no time travel)
    - event rate, mocap rate fall in expected range
    - quaternion norms ≈ 1, positions not all-zero/NaN
    - tracking_valid coverage
  Time alignment:
    - first / last sample offsets between two streams
    - overlap window
    - cross-correlation of event-rate vs mocap-speed (only meaningful if there is motion)
  Optional --plot: writes PNG plots of (event rate, mocap speed, mocap xyz) vs time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def section(title):
    print(f"\n=== {title} ===")


def fmt_ok(label, ok, detail=""):
    mark = "OK " if ok else "BAD"
    print(f"  [{mark}] {label}" + (f"  ({detail})" if detail else ""))


def check_monotonic(arr, name):
    if len(arr) < 2:
        return True, "len<2"
    diffs = np.diff(arr.astype(np.int64))
    n_back = int(np.sum(diffs < 0))
    if n_back == 0:
        return True, f"{len(arr)} samples, dt range [{int(diffs.min())}, {int(diffs.max())}] µs"
    return False, f"{n_back} backward steps! min step={int(diffs.min())} µs"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session_dir", type=Path)
    p.add_argument("--rb-id", type=int, default=None,
                   help="Optional NatNet ID to focus checks on; defaults to most-common id.")
    p.add_argument("--plot", action="store_true",
                   help="Save PNG plots in session_dir.")
    p.add_argument("--motion-thresh", type=float, default=0.005,
                   help="Position-stddev threshold (m) above which we run cross-correlation.")
    args = p.parse_args()

    sdir = args.session_dir
    ev_h5 = sdir / "events.h5"
    mc_h5 = sdir / "mocap.h5"

    section("FILES")
    fmt_ok("events.h5 exists", ev_h5.exists(), str(ev_h5))
    fmt_ok("mocap.h5 exists",  mc_h5.exists(),  str(mc_h5))
    if not (ev_h5.exists() and mc_h5.exists()):
        sys.exit("missing files; abort")

    ev = h5py.File(ev_h5, "r")
    mc = h5py.File(mc_h5, "r")

    section("EVENTS.H5 ATTRS")
    for k, v in dict(ev.attrs).items():
        print(f"  {k} = {v}")

    section("MOCAP.H5 ATTRS")
    for k, v in dict(mc.attrs).items():
        print(f"  {k} = {v}")

    # ---------- structural: events ----------
    section("EVENTS STRUCTURE")
    ev_t = ev["events/t"][:]
    ev_x = ev["events/x"][:]
    ev_y = ev["events/y"][:]
    ev_p = ev["events/p"][:]
    fmt_ok("count > 0", len(ev_t) > 0, f"{len(ev_t):,} events")
    ok, det = check_monotonic(ev_t, "events/t")
    fmt_ok("events/t non-decreasing", ok, det)
    dur_us = int(ev_t[-1] - ev_t[0]) if len(ev_t) > 1 else 0
    rate = len(ev_t) / (dur_us / 1e6) if dur_us > 0 else 0
    fmt_ok("event rate", 1e2 < rate < 1e8, f"{rate:,.0f} eps over {dur_us/1e6:.3f}s")
    w, h = int(ev.attrs.get("resolution_width", 640)), int(ev.attrs.get("resolution_height", 480))
    in_x = (0 <= ev_x.min()) and (ev_x.max() < w)
    in_y = (0 <= ev_y.min()) and (ev_y.max() < h)
    fmt_ok("x in [0, W)", in_x, f"x in [{ev_x.min()},{ev_x.max()}] vs W={w}")
    fmt_ok("y in [0, H)", in_y, f"y in [{ev_y.min()},{ev_y.max()}] vs H={h}")
    p_set = set(np.unique(ev_p).tolist())
    fmt_ok("polarity in {0,1} or {-1,1}", p_set <= {0, 1, -1}, f"unique={sorted(p_set)}")

    # ---------- structural: mocap ----------
    section("MOCAP STRUCTURE")
    mc_t   = mc["mocap/t"][:]
    rb_id  = mc["mocap/rb_id"][:]
    rb_t   = mc["mocap/rb_t_idx"][:]
    rb_x   = mc["mocap/rb_x"][:]
    rb_y_  = mc["mocap/rb_y"][:]
    rb_z   = mc["mocap/rb_z"][:]
    rb_qx  = mc["mocap/rb_qx"][:]
    rb_qy  = mc["mocap/rb_qy"][:]
    rb_qz  = mc["mocap/rb_qz"][:]
    rb_qw  = mc["mocap/rb_qw"][:]
    rb_tv  = mc["mocap/rb_tracking_valid"][:]

    fmt_ok("packet count > 0", len(mc_t) > 0, f"{len(mc_t):,} packets")
    ok, det = check_monotonic(mc_t, "mocap/t")
    fmt_ok("mocap/t non-decreasing", ok, det)
    mdur_us = int(mc_t[-1] - mc_t[0]) if len(mc_t) > 1 else 0
    mrate = len(mc_t) / (mdur_us / 1e6) if mdur_us > 0 else 0
    fmt_ok("mocap rate 60–500 Hz", 60 < mrate < 500, f"{mrate:.1f} Hz over {mdur_us/1e6:.3f}s")

    ids_unique = np.unique(rb_id)
    print(f"  rigid body ids present: {ids_unique.tolist()}")
    target = args.rb_id if args.rb_id is not None else int(np.bincount(rb_id).argmax())
    mask = rb_id == target
    print(f"  using rb_id={target}  ({int(mask.sum())} observations)")

    qn = np.sqrt(rb_qx[mask]**2 + rb_qy[mask]**2 + rb_qz[mask]**2 + rb_qw[mask]**2)
    fmt_ok("quaternion ‖q‖ ≈ 1",
           bool(np.all(np.abs(qn - 1) < 0.05)),
           f"min={qn.min():.4f} max={qn.max():.4f}")
    pos_finite = np.all(np.isfinite(rb_x[mask])) and np.all(np.isfinite(rb_y_[mask])) and np.all(np.isfinite(rb_z[mask]))
    fmt_ok("position finite (no NaN/Inf)", pos_finite)
    pos_nonzero = (rb_x[mask] != 0).any() or (rb_y_[mask] != 0).any() or (rb_z[mask] != 0).any()
    fmt_ok("position not all zeros", pos_nonzero)
    tv_frac = float(rb_tv[mask].mean()) if mask.sum() else 0.0
    fmt_ok("tracking_valid fraction > 0.9", tv_frac > 0.9, f"{tv_frac*100:.1f}%")

    pos_std = float(np.linalg.norm([rb_x[mask].std(), rb_y_[mask].std(), rb_z[mask].std()]))
    print(f"  position spread (‖σ‖): {pos_std*1000:.1f} mm")

    # ---------- time alignment ----------
    section("TIME ALIGNMENT")
    print(f"  events/t  range : [{ev_t[0]:>18d}, {ev_t[-1]:>18d}] µs")
    print(f"  mocap/t   range : [{mc_t[0]:>18d}, {mc_t[-1]:>18d}] µs")
    overlap_start = max(ev_t[0],  mc_t[0])
    overlap_end   = min(ev_t[-1], mc_t[-1])
    overlap_s = max(0, overlap_end - overlap_start) / 1e6
    fmt_ok("overlap window > 0",
           overlap_s > 0,
           f"{overlap_s:.3f}s in common")
    head_offset_ms = (mc_t[0] - ev_t[0]) / 1000.0
    tail_offset_ms = (ev_t[-1] - mc_t[-1]) / 1000.0
    print(f"  mocap starts {head_offset_ms:+.1f} ms after events first event")
    print(f"  events end   {tail_offset_ms:+.1f} ms after mocap last packet")
    fmt_ok("startup offset < 500 ms", abs(head_offset_ms) < 500)

    # ---------- cross-correlation only if there's motion ----------
    section("MOTION-BASED CROSS-CHECK")
    target_t = mc_t[rb_t[mask]]
    target_x = rb_x[mask]; target_y = rb_y_[mask]; target_z = rb_z[mask]
    if pos_std < args.motion_thresh:
        print(f"  position spread {pos_std*1000:.1f} mm < threshold {args.motion_thresh*1000:.0f} mm")
        print("  -> body is essentially stationary; can't cross-correlate.")
        print("     re-record while WAVING the rigid body in front of the camera to verify sync.")
    else:
        bin_us = 10_000  # 10 ms bins
        t0 = max(ev_t[0], target_t[0])
        t1 = min(ev_t[-1], target_t[-1])
        n_bins = max(1, int((t1 - t0) // bin_us))
        edges = t0 + np.arange(n_bins + 1) * bin_us

        ev_rate, _ = np.histogram(ev_t, bins=edges)
        # mocap speed via numerical derivative
        dt = np.diff(target_t.astype(np.float64)) / 1e6
        dx = np.diff(target_x); dy = np.diff(target_y); dz = np.diff(target_z)
        spd = np.sqrt(dx*dx + dy*dy + dz*dz) / np.maximum(dt, 1e-6)
        spd_t = (target_t[:-1] + target_t[1:]) / 2
        # resample speed to bins
        spd_binned = np.interp((edges[:-1] + edges[1:]) / 2, spd_t, spd, left=0, right=0)

        a = (ev_rate - ev_rate.mean()) / (ev_rate.std() + 1e-9)
        b = (spd_binned - spd_binned.mean()) / (spd_binned.std() + 1e-9)
        corr = np.correlate(a, b, mode="full")
        lags = np.arange(-len(b) + 1, len(a)) * bin_us / 1000.0  # ms
        best = int(np.argmax(corr))
        best_lag_ms = float(lags[best])
        peak = corr[best] / len(a)
        print(f"  bin = {bin_us/1000:.0f} ms,  best lag = {best_lag_ms:+.1f} ms,  normalized peak = {peak:.3f}")
        fmt_ok("|best lag| < 30 ms", abs(best_lag_ms) < 30)
        fmt_ok("correlation peak > 0.3", peak > 0.3,
               "low peak can mean weak motion or wrong sync")

    # ---------- optional plot ----------
    if args.plot:
        section("PLOTS")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"  matplotlib not available: {e}")
        else:
            t_ref = min(ev_t[0], mc_t[0])
            bin_us = 10_000
            edges = np.arange(t_ref, max(ev_t[-1], mc_t[-1]) + bin_us, bin_us)
            ev_rate, _ = np.histogram(ev_t, bins=edges)
            tc = (edges[:-1] - t_ref) / 1e6

            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            axes[0].plot(tc, ev_rate)
            axes[0].set_ylabel("events / 10 ms"); axes[0].set_title("event rate")
            target_t_s = (mc_t[rb_t[mask]] - t_ref) / 1e6
            axes[1].plot(target_t_s, target_x, label="x")
            axes[1].plot(target_t_s, target_y, label="y")
            axes[1].plot(target_t_s, target_z, label="z")
            axes[1].set_ylabel("position (m)"); axes[1].legend(loc="best")
            axes[1].set_title(f"mocap xyz for rb_id={target}")
            if len(target_t_s) > 1:
                dt = np.diff(target_t_s)
                dpos = np.diff(np.stack([target_x, target_y, target_z], axis=1), axis=0)
                spd = np.linalg.norm(dpos, axis=1) / np.maximum(dt, 1e-6)
                axes[2].plot((target_t_s[:-1] + target_t_s[1:]) / 2, spd)
            axes[2].set_ylabel("|v| (m/s)"); axes[2].set_xlabel("time since start (s)")
            axes[2].set_title("mocap speed")
            fig.tight_layout()
            out = sdir / "check_session.png"
            fig.savefig(out, dpi=120)
            print(f"  saved {out}")

    section("DONE")


if __name__ == "__main__":
    main()
