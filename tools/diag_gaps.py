#!/usr/bin/env python3
"""
诊断 events.h5 的时间空洞,判断"黑帧"到底是什么原因:
  - 空洞【周期性】(比如每隔约 1.0s 一个) -> 相机/USB 层面的断流(跟 CPU 无关)
  - 空洞【随机、和高事件率相关】           -> 录制跟不上、丢事件(CPU/IO 瓶颈)
  - 本来就【很多帧事件少】                 -> 场景太静,不是 bug

用法:
    python tools/diag_gaps.py recordings/<session>      # 读里面的 events.h5
    python tools/diag_gaps.py path/to/events.h5
    python tools/diag_gaps.py recordings/<session> --fps 30
"""
import argparse
from pathlib import Path

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="session 目录 或 events.h5")
    ap.add_argument("--fps", type=float, default=30.0, help="按多少 fps 分帧统计空帧")
    args = ap.parse_args()

    ev = args.path / "events.h5" if args.path.is_dir() else args.path
    with h5py.File(ev, "r") as f:
        t = f["events/t"][:].astype(np.int64)
        attrs = {k: f.attrs[k] for k in f.attrs}

    n = len(t)
    if n < 2:
        print("事件太少,无法分析")
        return
    dur = (t[-1] - t[0]) / 1e6
    print(f"文件: {ev}")
    print(f"events: {n:,}   时长 {dur:.2f}s   平均 {n/dur/1e6:.2f} Meps")
    print(f"attrs: denoised={attrs.get('denoised')}  compress={attrs.get('compress')}  "
          f"num_packets={attrs.get('num_packets')}")

    dt = np.diff(t)  # 相邻事件间隔(us)
    print("\n== 时间空洞(相邻事件间隔)==")
    for thr_ms in (5, 20, 50, 100, 200):
        cnt = int(np.sum(dt > thr_ms * 1000))
        print(f"  间隔 > {thr_ms:3d} ms : {cnt} 处")
    print(f"  最大间隔: {dt.max()/1000:.1f} ms")

    # 最大的若干空洞,看发生时刻是否等间隔(周期性)
    k = min(12, len(dt))
    big = np.argsort(dt)[-k:]
    big = big[np.argsort(t[big])]
    print(f"\n== 最大的 {k} 个空洞(发生时刻 / 时长 / 距上一个空洞)==")
    prev = None
    for i in big:
        at = (t[i] - t[0]) / 1e6
        since = f"{at - prev:6.2f}s" if prev is not None else "   -  "
        print(f"    @{at:7.2f}s   {dt[i]/1000:8.1f} ms   间隔上一个 {since}")
        prev = at

    # 按 fps 分帧:多少帧是空的(=黑帧)
    win = int(1e6 / args.fps)
    edges = np.arange(t[0], t[-1] + win, win)
    cnt, _ = np.histogram(t, bins=edges)
    empty = int(np.sum(cnt == 0))
    sparse = int(np.sum(cnt < 50))
    print(f"\n== 按 {args.fps:g}fps 分帧 ==")
    print(f"  共 {len(cnt)} 帧;空帧(0 事件) {empty} ({empty/len(cnt)*100:.1f}%);"
          f"  很稀(<50事件) {sparse} ({sparse/len(cnt)*100:.1f}%)")
    print(f"  每帧事件数: 中位 {int(np.median(cnt)):,}  最大 {int(cnt.max()):,}")

    print("\n== 判读 ==")
    if empty == 0:
        print("  没有空帧 —— 黑帧多半不是数据问题(可能是渲染/编码或显示)。")
    elif dt.max() > 50_000:
        gaps_s = (t[big] - t[0]) / 1e6
        diffs = np.diff(gaps_s)
        regular = len(diffs) >= 3 and (np.std(diffs) / (np.mean(diffs) + 1e-9) < 0.25)
        if regular:
            print(f"  大空洞近似【等间隔】(约每 {np.mean(diffs):.2f}s 一个)"
                  " -> 像相机/USB 周期性断流,跟 CPU 无关。")
        else:
            print("  大空洞【不规则】 -> 更像录制跟不上丢事件(试 --compress none / --no-denoise"
                  " / record_video);或 USB 不稳。")
    else:
        print("  没有明显大空洞,空帧来自场景太静(事件本来就少)。")


if __name__ == "__main__":
    main()
