#!/usr/bin/env python3
"""
离线把【原始 events.h5】去噪(BackgroundActivityNoiseFilter)+ 压缩,生成新文件。

配合 Jetson 工作流:
  录制时用 record_session.py --compress none --no-denoise(最轻、不丢事件),
  事后(不赶时间时,在笔记本或 Jetson 上)用本脚本补上去噪和压缩。

用法:
    python tools/denoise_h5.py recordings/<session>                 # -> events_denoised.h5
    python tools/denoise_h5.py recordings/<session>/events.h5 --ba-ms 3 --compress gzip
    python tools/denoise_h5.py in.h5 --output out.h5
"""
import argparse
import datetime
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import dv_processing as dv

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import record_session as rs   # 复用 make_event_datasets / write_event_batch / 压缩


def main():
    ap = argparse.ArgumentParser(description="离线给 events.h5 去噪 + 压缩")
    ap.add_argument("path", type=Path, help="session 目录 或 events.h5")
    ap.add_argument("--output", default=None, help="输出 h5(默认 同目录 events_denoised.h5)")
    ap.add_argument("--ba-ms", type=float, default=3.0, dest="ba_ms", help="去噪窗口(ms),越小越狠")
    ap.add_argument("--compress", choices=["gzip", "lzf", "none"], default="gzip",
                    help="输出压缩(离线不赶时间,默认 gzip 最省盘)")
    ap.add_argument("--chunk", type=int, default=2_000_000, help="每块事件数")
    args = ap.parse_args()

    in_path = args.path / "events.h5" if args.path.is_dir() else args.path
    out_path = Path(args.output) if args.output else in_path.with_name("events_denoised.h5")

    with h5py.File(in_path, "r") as f:
        n = f["events/t"].shape[0]
        w = int(f.attrs.get("resolution_width", 640))
        h = int(f.attrs.get("resolution_height", 480))
        src_attrs = {k: f.attrs[k] for k in f.attrs}
    print(f"输入: {in_path}  {n:,} 事件  {w}x{h}  ba-ms={args.ba_ms}  compress={args.compress}")

    noise = dv.noise.BackgroundActivityNoiseFilter(
        (w, h), backgroundActivityDuration=datetime.timedelta(milliseconds=max(0.1, args.ba_ms)))

    written = 0
    t0 = time.time()
    with h5py.File(in_path, "r") as fin, h5py.File(out_path, "w") as fout:
        ds = rs.make_event_datasets(fout, args.compress)
        for k, v in src_attrs.items():
            fout.attrs[k] = v
        fout.attrs["denoised"] = True
        fout.attrs["denoise_ba_ms"] = float(args.ba_ms)
        fout.attrs["compress"] = args.compress
        xt, yt, tt, pt = fin["events/x"], fin["events/y"], fin["events/t"], fin["events/p"]
        for s in range(0, n, args.chunk):
            e = min(s + args.chunk, n)
            x = xt[s:e].tolist(); y = yt[s:e].tolist()
            t = tt[s:e].tolist(); p = pt[s:e].tolist()
            store = dv.EventStore()
            for ti, xi, yi, pi in zip(t, x, y, p):
                store.push_back(int(ti), int(xi), int(yi), bool(pi))
            noise.accept(store)
            written += rs.write_event_batch(ds, noise.generateEvents())
            print(f"  {e:,}/{n:,}  保留 {written:,}", end="\r", flush=True)
        fout.attrs["num_events"] = int(written)

    print(f"\n完成: {out_path}")
    print(f"  {n:,} -> {written:,}  (去掉 {(1-written/max(n,1))*100:.0f}%)  用时 {time.time()-t0:.1f}s")
    print("  (mocap.h5 不受影响,沿用原 session 里的;事件时间轴未变,对齐仍成立)")


if __name__ == "__main__":
    main()
