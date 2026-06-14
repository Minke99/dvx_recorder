#!/usr/bin/env python3
"""Depack a recorded session: render events to MP4 + plot mocap xyz/yaw.

Usage:
    python tools/depack_h5_data.py recordings/<session_dir>
        [--mode time|count] [--fps 30] [--events-per-frame 5000]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def render_event_video(events_h5: Path, out_mp4: Path, mode: str,
                       fps: float, events_per_frame: int, background: int):
    with h5py.File(events_h5, "r") as handle:
        width = int(handle.attrs.get("resolution_width", 640))
        height = int(handle.attrs.get("resolution_height", 480))
        n = min(handle["events/x"].shape[0],
                handle["events/y"].shape[0],
                handle["events/t"].shape[0],
                handle["events/p"].shape[0])
        x = handle["events/x"][:n].astype(np.int64)
        y = handle["events/y"][:n].astype(np.int64)
        t = handle["events/t"][:n].astype(np.int64)
        p = handle["events/p"][:n].astype(np.int64)

    if len(t) == 0:
        print("  no events to render")
        return 0

    t = t - t[0]
    span_s = t[-1] / 1e6

    if mode == "time":
        frame_us = int(round(1e6 / fps))
        num_frames = int(t[-1] // frame_us) + 1
        frame_idx = (t // frame_us).astype(np.int64)
        starts = np.searchsorted(frame_idx, np.arange(num_frames), side="left")
        ends = np.searchsorted(frame_idx, np.arange(num_frames), side="right")
        print(f"  mode=time, events={len(t):,}, span={span_s:.2f}s, "
              f"{width}x{height}, fps={fps}, frames={num_frames}, {frame_us}us/frame")
    else:
        n_per = events_per_frame
        starts = np.arange(0, len(t), n_per, dtype=np.int64)
        ends = np.minimum(starts + n_per, len(t))
        num_frames = len(starts)
        print(f"  mode=count, events={len(t):,}, span={span_s:.2f}s, "
              f"{width}x{height}, fps={fps}, frames={num_frames}, {n_per} events/frame")

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter failed to open " + str(out_mp4))

    for fi in range(num_frames):
        img = np.full((height, width, 3), background, dtype=np.uint8)
        s, e = int(starts[fi]), int(ends[fi])
        if e > s:
            fx = x[s:e]; fy = y[s:e]; fp = p[s:e]
            pos = fp == 1
            neg = ~pos
            # OpenCV 是 BGR: ON(变亮)=绿, OFF(变暗)=红 —— 和实时预览一致
            img[fy[pos], fx[pos]] = (0, 255, 0)
            img[fy[neg], fx[neg]] = (0, 0, 255)
        writer.write(img)
        if fi % 50 == 0:
            print(f"  frame {fi}/{num_frames}")

    writer.release()
    print(f"  saved {out_mp4}")
    return num_frames


def quat_to_yaw(qx, qy, qz, qw):
    return np.arctan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def plot_mocap(mocap_h5: Path, out_png: Path, rb_id: int | None, show: bool):
    with h5py.File(mocap_h5, "r") as h:
        g = h["mocap"]
        mc_t = g["t"][:]
        if len(mc_t) == 0:
            print("  mocap.h5 has 0 packets; skipping plot")
            return False
        rb_keys = ["rb_t_idx", "rb_id", "rb_x", "rb_y", "rb_z",
                   "rb_qx", "rb_qy", "rb_qz", "rb_qw"]
        n = min(g[k].shape[0] for k in rb_keys)
        rb_t_idx = g["rb_t_idx"][:n]
        ids = g["rb_id"][:n]
        rb_x = g["rb_x"][:n]; rb_y = g["rb_y"][:n]; rb_z = g["rb_z"][:n]
        rb_qx = g["rb_qx"][:n]; rb_qy = g["rb_qy"][:n]
        rb_qz = g["rb_qz"][:n]; rb_qw = g["rb_qw"][:n]

    if len(ids) == 0:
        print("  no rigid-body observations; skipping plot")
        return False

    target = rb_id if rb_id is not None else int(np.bincount(ids).argmax())
    mask = ids == target
    print(f"  plotting rb_id={target}  ({int(mask.sum())} observations)")

    t = mc_t[rb_t_idx[mask]]
    t_s = (t - t[0]) / 1e6
    x = rb_x[mask]; y = rb_y[mask]; z = rb_z[mask]
    yaw = quat_to_yaw(rb_qx[mask], rb_qy[mask], rb_qz[mask], rb_qw[mask])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t_s, x, label="x")
    axes[0].plot(t_s, y, label="y")
    axes[0].plot(t_s, z, label="z")
    axes[0].set_ylabel("position (m)")
    axes[0].legend(loc="best")
    axes[0].set_title(f"mocap xyz / yaw   rb_id={target}")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_s, np.degrees(yaw))
    axes[1].set_ylabel("yaw (deg)")
    axes[1].set_xlabel("time since first sample (s)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"  saved {out_png}")
    if show:
        plt.show()
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--mode", choices=["time", "count"], default="time",
                    help="time = fixed time interval per frame (1/fps); count = fixed event count.")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="Playback fps. In time mode also sets the time window (1/fps).")
    ap.add_argument("--events-per-frame", type=int, default=5000,
                    help="Events per frame when --mode count.")
    ap.add_argument("--background", type=int, default=0, help="Background gray level 0-255.")
    ap.add_argument("--rb-id", type=int, default=None)
    ap.add_argument("--show", action="store_true",
                    help="Open the mocap plot interactively after saving.")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    sdir = args.session_dir
    ev_h5 = sdir / "events.h5"
    mc_h5 = sdir / "mocap.h5"

    if ev_h5.exists():
        print(f"=== events: {ev_h5}")
        out_name = "events_time.mp4" if args.mode == "time" else "events_count.mp4"
        render_event_video(ev_h5, sdir / out_name, args.mode, args.fps,
                           args.events_per_frame, args.background)
    else:
        print(f"missing {ev_h5}")

    if mc_h5.exists():
        print(f"=== mocap: {mc_h5}")
        try:
            plot_mocap(mc_h5, sdir / "mocap_xyz_yaw.png", args.rb_id, args.show)
        except Exception as e:
            print(f"  mocap plot failed: {e}")
    else:
        print(f"missing {mc_h5}")


if __name__ == "__main__":
    main()
