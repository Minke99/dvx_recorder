#!/usr/bin/env python3
"""
DVXplorer 实时事件预览 —— 带去噪版(在 dvx_live.py 基础上加了噪声滤波)。

和 dvx_live.py 一样:黑底、绿(ON)/红(OFF)。
区别:画面送去显示之前先过一道 BackgroundActivityNoiseFilter 去噪。
原理:孤立的、周围一小段时间内没有邻近事件的点判为噪声丢掉;
      真实运动的边缘因为空间-时间相关性会被保留。

依赖:
    pip install dv-processing opencv-python numpy

用法:
    # 默认开去噪
    python dvx_live_denoise.py

    # 调去噪强度:相关窗口(毫秒),越小去得越狠(默认 1.0)
    python dvx_live_denoise.py --ba-ms 0.5     # 更干净,可能损失弱信号
    python dvx_live_denoise.py --ba-ms 3.0     # 更温和,保留更多

    # 关掉去噪(等价于原始 dvx_live.py)
    python dvx_live_denoise.py --no-denoise

    其它参数同 dvx_live.py: --fps --scale --swap --serial --net

运行时快捷键:
    q / ESC   退出
    d         临时开/关去噪(方便对比)
    [ / ]     实时调小 / 调大相关窗口(改变去噪强度)

按 q 或 ESC 退出。
"""

import argparse
import datetime
import sys
import time

import cv2
import dv_processing as dv


def open_source(args):
    """打开事件源:网络流或本地相机。"""
    if args.net:
        host, _, port = args.net.partition(":")
        port = int(port or 7777)
        print(f"连接网络流 {host}:{port} ...")
        return dv.io.NetworkReader(host, port)
    if args.serial:
        return dv.io.camera.open(args.serial)
    return dv.io.camera.open()


def main():
    ap = argparse.ArgumentParser(description="DVXplorer 实时事件预览(去噪版)")
    ap.add_argument("--fps", type=float, default=30.0, help="画面刷新率 (默认 30)")
    ap.add_argument("--scale", type=float, default=1.0, help="窗口放大倍数 (默认 1)")
    ap.add_argument("--swap", action="store_true", help="交换颜色 (红=ON, 绿=OFF)")
    ap.add_argument("--serial", default=None, help="指定相机序列号")
    ap.add_argument("--net", default=None, metavar="HOST:PORT",
                    help="改连 DV 的网络输出流")
    ap.add_argument("--ba-ms", type=float, default=1.0, dest="ba_ms",
                    help="去噪相关窗口(毫秒),越小去噪越狠 (默认 1.0)")
    ap.add_argument("--no-denoise", action="store_true", help="关闭去噪")
    args = ap.parse_args()

    try:
        cam = open_source(args)
    except Exception as e:
        print(f"\n打开事件源失败: {e}\n", file=sys.stderr)
        if not args.net:
            print("最常见原因: DV-GUI 还开着,相机被它占用了。先关掉 DV 再跑。",
                  file=sys.stderr)
        sys.exit(1)

    res = cam.getEventResolution()
    if res is None:
        print("该事件源没有事件流。", file=sys.stderr)
        sys.exit(1)
    width, height = res
    print(f"已连接: {cam.getCameraName()}   分辨率: {width}x{height}")

    # 渲染器:黑底 + 绿(ON)/红(OFF)
    vis = dv.visualization.EventVisualizer(res)
    vis.setBackgroundColor(dv.visualization.colors.black())
    on_color = dv.visualization.colors.lime()
    off_color = dv.visualization.colors.red()
    if args.swap:
        on_color, off_color = off_color, on_color
    vis.setPositiveColor(on_color)
    vis.setNegativeColor(off_color)

    # 去噪滤波器
    ba_ms = max(0.1, args.ba_ms)
    noise = dv.noise.BackgroundActivityNoiseFilter(
        res, backgroundActivityDuration=datetime.timedelta(milliseconds=ba_ms))
    denoise_on = not args.no_denoise

    win = "DVXplorer Live (denoise)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, int(width * args.scale), int(height * args.scale))

    slicer = dv.EventStreamSlicer()
    interval = datetime.timedelta(milliseconds=1000.0 / args.fps)

    def render(events):
        cv2.imshow(win, vis.generateImage(events))

    slicer.doEveryTimeInterval(interval, render)

    last_log = time.time()
    print(f"去噪: {'开' if denoise_on else '关'}   相关窗口: {ba_ms:.2f}ms"
          "   (运行时: d 开关去噪 / [ ] 调强度 / q 退出)")

    try:
        while cam.isRunning():
            events = cam.getNextEventBatch()
            if events is not None:
                if denoise_on:
                    noise.accept(events)
                    events = noise.generateEvents()
                slicer.accept(events)

            # 每 ~2 秒报一次去掉了多少
            if denoise_on and time.time() - last_log > 2.0:
                print(f"  去噪中: 去除比例 {noise.getReductionFactor()*100:.0f}%"
                      f"  (相关窗口 {ba_ms:.2f}ms)", end="\r")
                last_log = time.time()

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("d"):
                denoise_on = not denoise_on
                print(f"\n去噪 -> {'开' if denoise_on else '关'}")
            elif key == ord("["):
                ba_ms = max(0.1, ba_ms - 0.5)
                noise.setBackgroundActivityDuration(
                    datetime.timedelta(milliseconds=ba_ms))
                print(f"\n相关窗口 -> {ba_ms:.2f}ms (更狠)")
            elif key == ord("]"):
                ba_ms = ba_ms + 0.5
                noise.setBackgroundActivityDuration(
                    datetime.timedelta(milliseconds=ba_ms))
                print(f"\n相关窗口 -> {ba_ms:.2f}ms (更温和)")

            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
