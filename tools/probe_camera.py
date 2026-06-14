#!/usr/bin/env python3
"""
极简相机探针:把"相机/USB 层"和"我们的录制代码层"分开,定位周期性空洞到底来自哪。

它【只做一件事】:在一个最紧的循环里不停 getNextEventBatch(),把每批的
(到达墙钟, 事件最早t, 事件最晚t, 事件数) 记到内存。不去噪、不录 mocap、
不写盘、不压缩、不渲染。跑完分析事件时间线里的空洞。

判读:
  - 这样【还有】>50ms 的事件时间空洞  -> 是相机/USB/驱动层在丢(跟我们代码无关)
  - 这样【没有】空洞,但 record_session 有 -> 是我们的处理/写盘/mocap 拖的

用法:
    python tools/probe_camera.py                 # 默认录 10s
    python tools/probe_camera.py --duration 15
    python tools/probe_camera.py --serial <序列号>
"""
import argparse
import sys
import time

import numpy as np
import dv_processing as dv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--serial", default=None)
    ap.add_argument("--gap-ms", type=float, default=50.0, help="多大算空洞")
    args = ap.parse_args()

    try:
        cam = dv.io.camera.open(args.serial) if args.serial else dv.io.camera.open()
    except Exception as e:  # noqa: BLE001
        print(f"打开相机失败: {e}\n(DV-GUI 还开着?)", file=sys.stderr)
        sys.exit(1)
    print(f"相机: {cam.getCameraName()}  分辨率: {cam.getEventResolution()}")
    print(f"纯采集 {args.duration:g}s,什么都不做… (Ctrl+C 提前停)")

    wall, tlo, thi, sz = [], [], [], []
    t0 = time.time()
    try:
        while cam.isRunning() and time.time() - t0 < args.duration:
            ev = cam.getNextEventBatch()
            if ev is None or ev.size() == 0:
                continue
            wall.append(time.time())
            tlo.append(int(ev.getLowestTime()))
            thi.append(int(ev.getHighestTime()))
            sz.append(int(ev.size()))
    except KeyboardInterrupt:
        pass

    if len(wall) < 3:
        print("批次太少,没采到数据")
        return
    wall = np.array(wall); tlo = np.array(tlo); thi = np.array(thi); sz = np.array(sz)
    span = (thi[-1] - tlo[0]) / 1e6
    print(f"\n采到 {len(wall)} 批, {sz.sum():,} 事件, 事件时间跨度 {span:.2f}s, "
          f"平均 {sz.sum()/span/1e6:.2f} Meps")
    print(f"批速率 ~{len(wall)/(wall[-1]-wall[0]):.0f} 批/秒")

    # 事件时间空洞 = 下一批最早t - 本批最晚t
    ev_gap = (tlo[1:] - thi[:-1])
    # 墙钟到达间隔
    wall_gap = np.diff(wall) * 1e6  # us
    th = args.gap_ms * 1000
    big = np.where(ev_gap > th)[0]
    print(f"\n>{args.gap_ms:g}ms 的【事件时间空洞】: {len(big)} 个,"
          f" 总 {ev_gap[big].sum()/1e6:.2f}s ({ev_gap[big].sum()/(span*1e6)*100:.0f}%)")
    for i in big[:25]:
        print(f"  @{(thi[i]-tlo[0])/1e6:6.2f}s  事件空洞 {ev_gap[i]/1000:6.0f}ms"
              f"   同期墙钟也停了 {wall_gap[i]/1000:6.0f}ms")
    if len(big) > 3:
        per = np.median(np.diff((thi[big]) / 1e6))
        print(f"  空洞中位间隔: {per:.3f}s")

    print("\n== 判读 ==")
    if len(big) == 0:
        print("  纯采集没有空洞 -> 丢包是【我们的代码】拖的(去噪/写盘/压缩/mocap)。")
        print("     下一步:逐个关掉看是哪个(--no-denoise / 去掉 mocap / --compress none)。")
    else:
        wall_also = np.median(wall_gap[big])
        if wall_also > th * 0.5:
            print(f"  采集时事件空洞 + 同期墙钟也停 ~{wall_also/1000:.0f}ms")
            print("  -> 连最紧的纯采集循环都停 = 相机/USB/驱动/供电层周期性断流,跟我们代码无关。")
            print("     方向:换 USB 口/线、关 USB 自动挂起、Jetson 供电/功耗模式、libcaer 缓冲。")
        else:
            print("  -> 事件时间有空洞但墙钟没停,少见;把输出发我。")


if __name__ == "__main__":
    main()
