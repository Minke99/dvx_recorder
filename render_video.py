#!/usr/bin/env python3
"""
把录好的事件 h5 (events/x,y,t,p) 回放渲染成 MP4。

搬自 record_dvx/render_video.py,做了两处改动:
  1) 去掉对 lib/ 的依赖,单文件即可运行;
  2) 配色改成和本目录预览脚本一致:ON(变亮)=绿,OFF(变暗)=红
     (原版是 ON=红、OFF=蓝,会和实时画面对不上)。

用法:
    # 不指定输入时,自动选 recordings/ 里最新的 .h5
    python render_video.py

    # 指定输入/输出
    python render_video.py --input recordings/take1.h5 --output recordings/take1.mp4

    # 固定时间间隔(默认,播放=真实时间)
    python render_video.py --fps 30

    # 固定事件数:每帧 5000 个事件
    python render_video.py --mode count --events-per-frame 5000

依赖: pip install opencv-python h5py numpy
"""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


def newest_recording():
    """没给 --input 时,挑 recordings/ 里最新修改的 .h5。"""
    rec = Path("recordings")
    files = sorted(rec.glob("*.h5"), key=lambda p: p.stat().st_mtime) if rec.is_dir() else []
    return files[-1] if files else None


def render(args):
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = newest_recording()
        if input_path is None:
            print("recordings/ 下没有 .h5 文件,请用 --input 指定。")
            return
        print("自动选用最新录制:", input_path)

    output_path = Path(args.output) if args.output else input_path.with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as handle:
        width = int(handle.attrs.get("resolution_width", 640))
        height = int(handle.attrs.get("resolution_height", 480))
        x = handle["events/x"][:].astype(np.int64)
        y = handle["events/y"][:].astype(np.int64)
        t = handle["events/t"][:].astype(np.int64)
        p = handle["events/p"][:].astype(np.int64)

    if len(t) == 0:
        print("no events to render")
        return

    t0 = t[0]
    t = t - t0
    span_s = t[-1] / 1e6

    if args.mode == "time":
        # 固定时间间隔:每帧覆盖 1/fps 秒
        frame_us = int(round(1e6 / args.fps))
        num_frames = int(t[-1] // frame_us) + 1
        frame_idx = (t // frame_us).astype(np.int64)
        starts = np.searchsorted(frame_idx, np.arange(num_frames), side="left")
        ends = np.searchsorted(frame_idx, np.arange(num_frames), side="right")
        print("mode=time, events={}, span={:.2f}s, {}x{}, fps={}, frames={}, {}us/frame".format(
            len(t), span_s, width, height, args.fps, num_frames, frame_us))
    else:
        # 固定事件数:每帧 events_per_frame 个事件
        n = args.events_per_frame
        starts = np.arange(0, len(t), n, dtype=np.int64)
        ends = np.minimum(starts + n, len(t))
        num_frames = len(starts)
        print("mode=count, events={}, span={:.2f}s, {}x{}, fps={}, frames={}, {} events/frame".format(
            len(t), span_s, width, height, args.fps, num_frames, n))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (width, height))

    bg = args.background
    for fi in range(num_frames):
        img = np.full((height, width, 3), bg, dtype=np.uint8)
        s, e = starts[fi], ends[fi]
        if e > s:
            fx = x[s:e]
            fy = y[s:e]
            fp = p[s:e]
            pos = fp == 1
            neg = ~pos
            # OpenCV 是 BGR: ON(变亮)=绿, OFF(变暗)=红 —— 和实时预览一致
            img[fy[pos], fx[pos]] = (0, 255, 0)
            img[fy[neg], fx[neg]] = (0, 0, 255)
        writer.write(img)
        if fi % 50 == 0:
            print("frame {}/{}".format(fi, num_frames))

    writer.release()
    print("saved", output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="把事件 h5 回放渲染成 MP4。")
    parser.add_argument("--input", default=None, help="输入 h5 路径(默认选 recordings/ 里最新的)。")
    parser.add_argument("--output", default=None, help="输出 MP4 路径(默认同名 .mp4)。")
    parser.add_argument("--mode", choices=["time", "count"], default="time",
                        help="time = 每帧固定时间窗;count = 每帧固定事件数。")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="播放帧率。time 模式下同时决定时间窗(1/fps)。")
    parser.add_argument("--events-per-frame", type=int, default=5000,
                        help="--mode count 时每帧的事件数。")
    parser.add_argument("--background", type=int, default=0, help="背景灰度 0-255(默认 0 黑)。")
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args())
