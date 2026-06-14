#!/usr/bin/env python3
"""
DVXplorer 实时预览 + 录制成 HDF5。

一边看画面(黑底、绿 ON / 红 OFF,可去噪),一边把事件存进 .h5 文件。

存的格式(和 record_dvx 的 record_raw.py 一致,可直接用它的 render_video.py 回放):
    events/x  int32
    events/y  int32
    events/t  int64   (时间戳, 单位 us)
    events/p  int8    (1 = ON/变亮, 0 = OFF/变暗)
    attrs: camera_name, resolution_width, resolution_height, time_unit="us",
           format="raw_dvx_events_v1", num_events, num_packets, duration_wall_s

依赖:
    pip install dv-processing opencv-python numpy h5py

用法:
    # 录到 recordings/dvx_<时间戳>.h5,实时预览带去噪(默认 ba-ms=3.0)
    python dvx_record.py

    # 录固定时长(秒)后自动停止
    python dvx_record.py --duration 10

    # 指定输出文件
    python dvx_record.py --output recordings/my_take.h5

    # 默认存的是【原始未去噪】事件(无损,去噪只作用于预览);
    # 想直接存【去噪后】的干净事件:
    python dvx_record.py --save-denoised

    # 预览不去噪 / 调去噪强度(越小越狠)
    python dvx_record.py --no-denoise
    python dvx_record.py --ba-ms 1.0

    其它参数同 dvx_live.py: --fps --scale --swap --serial --net

运行时快捷键:
    q / ESC   停止并保存
    r         暂停 / 继续录制(预览不中断)
    d         开/关预览去噪

画面左上角会显示 ● REC / PAUSE、已录时长和事件数。
"""

import argparse
import datetime
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import h5py
import dv_processing as dv


def open_source(args):
    if args.net:
        host, _, port = args.net.partition(":")
        return dv.io.NetworkReader(host, int(port or 7777))
    if args.serial:
        return dv.io.camera.open(args.serial)
    return dv.io.camera.open()


def create_ds(group, name, dtype):
    return group.create_dataset(name, shape=(0,), maxshape=(None,),
                                dtype=dtype, chunks=(1 << 16,))


def append_ds(ds, arr):
    n = ds.shape[0]
    ds.resize((n + len(arr),))
    ds[n:] = arr


def main():
    ap = argparse.ArgumentParser(description="DVXplorer 实时预览 + 录制 HDF5")
    ap.add_argument("--output", default=None, help="输出 .h5 路径(默认 recordings/dvx_<时间戳>.h5)")
    ap.add_argument("--duration", type=float, default=0.0, help="录制时长(秒),<=0 表示一直录到按 q")
    ap.add_argument("--save-denoised", action="store_true", help="存去噪后的事件(默认存原始无损)")
    ap.add_argument("--ba-ms", type=float, default=3.0, dest="ba_ms",
                    help="去噪相关窗口(毫秒),越小去噪越狠 (默认 3.0)")
    ap.add_argument("--no-denoise", action="store_true", help="预览不去噪")
    ap.add_argument("--fps", type=float, default=30.0, help="预览刷新率 (默认 30)")
    ap.add_argument("--scale", type=float, default=1.0, help="窗口放大倍数")
    ap.add_argument("--swap", action="store_true", help="交换颜色 (红=ON, 绿=OFF)")
    ap.add_argument("--serial", default=None, help="指定相机序列号")
    ap.add_argument("--net", default=None, metavar="HOST:PORT", help="改连 DV 网络流")
    args = ap.parse_args()

    # 输出路径
    if args.output:
        out_path = Path(args.output)
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("recordings") / f"dvx_{stamp}.h5"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        cam = open_source(args)
    except Exception as e:
        print(f"\n打开事件源失败: {e}", file=sys.stderr)
        if not args.net:
            print("最常见原因: DV-GUI 还开着,相机被占用。先关掉 DV 再跑。", file=sys.stderr)
        sys.exit(1)

    res = cam.getEventResolution()
    if res is None:
        print("该事件源没有事件流。", file=sys.stderr)
        sys.exit(1)
    width, height = res
    cam_name = cam.getCameraName()
    print(f"已连接: {cam_name}   分辨率: {width}x{height}")
    print(f"录制到: {out_path}   存{'去噪后' if args.save_denoised else '原始'}事件")

    # 渲染器
    vis = dv.visualization.EventVisualizer(res)
    vis.setBackgroundColor(dv.visualization.colors.black())
    on_c, off_c = dv.visualization.colors.lime(), dv.visualization.colors.red()
    if args.swap:
        on_c, off_c = off_c, on_c
    vis.setPositiveColor(on_c)
    vis.setNegativeColor(off_c)

    # 去噪滤波器(预览去噪 或 存去噪 时都需要)
    preview_denoise = not args.no_denoise
    ba_ms = max(0.1, args.ba_ms)
    noise = dv.noise.BackgroundActivityNoiseFilter(
        res, backgroundActivityDuration=datetime.timedelta(milliseconds=ba_ms))

    win = "DVXplorer Record"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, int(width * args.scale), int(height * args.scale))

    # 共享状态(给预览回调用)
    st = {"recording": True, "events": 0, "rec_seconds": 0.0}

    slicer = dv.EventStreamSlicer()
    interval = datetime.timedelta(milliseconds=1000.0 / args.fps)

    def render(events):
        img = vis.generateImage(events)
        if st["recording"]:
            txt = f"REC {st['rec_seconds']:5.1f}s  {st['events']:,} ev"
            cv2.circle(img, (16, 18), 7, (0, 0, 255), -1)
            color = (255, 255, 255)
        else:
            txt = "PAUSE  (r resume)"
            color = (0, 215, 255)
        cv2.putText(img, txt, (30, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        cv2.imshow(win, img)

    slicer.doEveryTimeInterval(interval, render)

    print("开始录制。 q/ESC 停止保存 | r 暂停/继续 | d 开关预览去噪")
    start = time.time()
    rec_accum = 0.0          # 实际处于录制状态的累计时长
    last_t = start
    packets = 0

    with h5py.File(out_path, "w") as h:
        g = h.create_group("events")
        x_ds = create_ds(g, "x", np.int32)
        y_ds = create_ds(g, "y", np.int32)
        t_ds = create_ds(g, "t", np.int64)
        p_ds = create_ds(g, "p", np.int8)
        h.attrs["camera_name"] = cam_name
        h.attrs["resolution_width"] = int(width)
        h.attrs["resolution_height"] = int(height)
        h.attrs["time_unit"] = "us"
        h.attrs["format"] = "raw_dvx_events_v1"

        try:
            while cam.isRunning():
                events = cam.getNextEventBatch()
                if events is not None and events.size() > 0:
                    # 决定存盘内容 / 预览内容
                    if args.save_denoised:
                        noise.accept(events)
                        to_disk = noise.generateEvents()
                        to_preview = to_disk
                    else:
                        to_disk = events
                        if preview_denoise:
                            noise.accept(events)
                            to_preview = noise.generateEvents()
                        else:
                            to_preview = events

                    if st["recording"] and to_disk.size() > 0:
                        a = to_disk.numpy()
                        append_ds(x_ds, a["x"].astype(np.int32))
                        append_ds(y_ds, a["y"].astype(np.int32))
                        append_ds(t_ds, a["timestamp"].astype(np.int64))
                        append_ds(p_ds, a["polarity"].astype(np.int8))
                        st["events"] += int(to_disk.size())
                        packets += 1

                    slicer.accept(to_preview)

                now = time.time()
                if st["recording"]:
                    rec_accum += now - last_t
                    st["rec_seconds"] = rec_accum
                last_t = now

                if args.duration > 0 and rec_accum >= args.duration:
                    print("\n到达设定时长,停止。")
                    break

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                elif key == ord("r"):
                    st["recording"] = not st["recording"]
                    print(f"\n录制 -> {'继续' if st['recording'] else '暂停'}")
                elif key == ord("d"):
                    preview_denoise = not preview_denoise
                    print(f"\n预览去噪 -> {'开' if preview_denoise else '关'}")

                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
        except KeyboardInterrupt:
            print("\n用户中断。")
        finally:
            h.attrs["num_events"] = int(st["events"])
            h.attrs["num_packets"] = int(packets)
            h.attrs["duration_wall_s"] = float(rec_accum)
            cv2.destroyAllWindows()

    print(f"已保存: {out_path}  共 {st['events']:,} 个事件 / {packets} 个批次"
          f"  录制时长 {rec_accum:.1f}s")


if __name__ == "__main__":
    main()
