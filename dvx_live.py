#!/usr/bin/env python3
"""
DVXplorer 实时事件预览 —— 复刻 DV viewer 的效果。

黑色背景，正极性(变亮)事件用绿色，负极性(变暗)事件用红色。

依赖:
    pip install dv-processing opencv-python numpy

用法:
    # 直接连相机(需先关闭 DV-GUI，USB 相机同一时刻只能被一个程序占用)
    python dvx_live.py

    # 调高/调低刷新率(默认 30fps)
    python dvx_live.py --fps 60

    # 换色:红=ON 绿=OFF
    python dvx_live.py --swap

    # 窗口放大 3 倍
    python dvx_live.py --scale 3

    # 不关 DV，改连 DV 的网络输出流(需在 DV-GUI 里加一个
    #   "output > net tcp server" 模块，假设监听 127.0.0.1:7777)
    python dvx_live.py --net 127.0.0.1:7777

按 q 或 ESC 退出。
"""

import argparse
import datetime
import sys

import cv2
import dv_processing as dv


def open_source(args):
    """打开事件源:网络流或本地相机。"""
    if args.net:
        host, _, port = args.net.partition(":")
        port = int(port or 7777)
        print(f"连接网络流 {host}:{port} ...")
        return dv.io.NetworkReader(host, port)

    # 本地 USB 相机
    if args.serial:
        return dv.io.camera.open(args.serial)
    return dv.io.camera.open()


def main():
    ap = argparse.ArgumentParser(description="DVXplorer 实时事件预览")
    ap.add_argument("--fps", type=float, default=30.0, help="画面刷新率 (默认 30)")
    ap.add_argument("--scale", type=float, default=1.0, help="窗口放大倍数 (默认 1)")
    ap.add_argument("--swap", action="store_true", help="交换颜色 (红=ON, 绿=OFF)")
    ap.add_argument("--serial", default=None, help="指定相机序列号(多台相机时)")
    ap.add_argument("--net", default=None, metavar="HOST:PORT",
                    help="改连 DV 的网络输出流，而不是直接占用相机")
    args = ap.parse_args()

    try:
        cam = open_source(args)
    except Exception as e:
        print(f"\n打开事件源失败: {e}\n", file=sys.stderr)
        if not args.net:
            print("最常见原因: DV-GUI 还开着，相机被它占用了。", file=sys.stderr)
            print("解决办法二选一:", file=sys.stderr)
            print("  1) 关闭 DV-GUI 后重新运行本脚本；或", file=sys.stderr)
            print("  2) 保持 DV 打开，在 DV 里加 net tcp server 输出，", file=sys.stderr)
            print("     再用 python dvx_live.py --net 127.0.0.1:7777", file=sys.stderr)
        sys.exit(1)

    res = cam.getEventResolution()  # (宽, 高)
    if res is None:
        print("该事件源没有事件流。", file=sys.stderr)
        sys.exit(1)
    width, height = res
    name = cam.getCameraName()
    print(f"已连接: {name}   分辨率: {width}x{height}   刷新率: {args.fps:g}fps")
    print("按 q 或 ESC 退出。")

    # 事件渲染器:黑底 + 绿(ON)/红(OFF)，和 DV 一致
    vis = dv.visualization.EventVisualizer(res)
    vis.setBackgroundColor(dv.visualization.colors.black())
    on_color = dv.visualization.colors.lime()   # 亮绿 (0,255,0) BGR
    off_color = dv.visualization.colors.red()    # 亮红 (0,0,255) BGR
    if args.swap:
        on_color, off_color = off_color, on_color
    vis.setPositiveColor(on_color)
    vis.setNegativeColor(off_color)

    win = "DVXplorer Live"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, int(width * args.scale), int(height * args.scale))

    # 按时间窗口把事件切成一帧帧来显示
    slicer = dv.EventStreamSlicer()
    interval = datetime.timedelta(milliseconds=1000.0 / args.fps)

    def render(events):
        cv2.imshow(win, vis.generateImage(events))

    slicer.doEveryTimeInterval(interval, render)

    try:
        while cam.isRunning():
            events = cam.getNextEventBatch()
            if events is not None:
                slicer.accept(events)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            # 用户点了窗口的关闭按钮
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
