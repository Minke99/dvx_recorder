#!/usr/bin/env python3
"""
录【事件视频 mp4】+ mocap(时间对齐)—— 给 Jetson 等慢设备用的轻量版。

和 record_session.py 的区别:
  record_session 把【每个事件】写进 events.h5 —— 数据量/CPU 随事件率涨,
                  慢设备(Jetson)高事件率下可能跟不上、丢事件 -> 回放黑帧。
  record_video   把事件按 --fps 累积成帧、直接编码进 events.mp4 ——
                  数据量/CPU 由【帧率】决定,与事件率无关 -> 慢设备也不丢、文件还小。

代价(有损):只有 fps 帧(默认 30),没有微秒级原始事件;不能事后改帧率/事件数重渲;
mocap 对齐精度到帧(~1/fps),不是微秒。需要原始事件就用 record_session.py。

输出:
    recordings/<ts>/events.mp4   黑底/绿ON/红OFF,按 --fps
    recordings/<ts>/mocap.h5     和 record_session 同格式(time_unit=us_camera_relative)
    recordings/<ts>/sync.json    {camera_first_us, wall_first_us, video_fps}
  帧 i 对应相机时间 ≈ cam_first_us + i*(1e6/fps);mocap/t 在同一相机时间轴上,
  所以某个 mocap 时刻 t 对应帧 round((t - cam_first_us) / (1e6/fps))。

依赖: pip install dv-processing opencv-python numpy h5py tqdm

用法:
    python record_video.py --duration 10            # 只录(无窗口),Ctrl+C / q 停
    python record_video.py --fps 60                 # 60fps
    python record_video.py --no-denoise             # 不去噪(更省 CPU)
    python record_video.py --preview                # 同时看实时画面
"""

import argparse
import datetime
import json
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import h5py
import dv_processing as dv

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import record_session as rs          # 复用 mocap_worker / write_mocap_packet / make_mocap_datasets / open_camera
from lib.quit_key import QuitKey


def main():
    ap = argparse.ArgumentParser(description="录事件视频 mp4 + mocap(时间对齐),慢设备轻量版")
    ap.add_argument("--session-name", default=None, help="recordings/ 下子目录名(默认时间戳)")
    ap.add_argument("--session-root", default="recordings")
    ap.add_argument("--duration", type=float, default=0.0, help="录制秒数,<=0 录到停止")
    ap.add_argument("--multicast", default="239.255.42.99")
    ap.add_argument("--port", type=int, default=1511)
    ap.add_argument("--bind-ip", default="0.0.0.0")
    ap.add_argument("--fps", type=float, default=30.0, help="视频帧率(也是每帧的事件时间窗 1/fps)")
    ap.add_argument("--ba-ms", type=float, default=3.0, dest="ba_ms", help="去噪窗口(ms),越小越狠")
    ap.add_argument("--no-denoise", action="store_true", help="不去噪(更省 CPU)")
    ap.add_argument("--preview", action="store_true", help="同时开预览窗口")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--swap", action="store_true", help="红绿对调")
    ap.add_argument("--serial", default=None)
    ap.add_argument("--status-every", type=float, default=1.0)
    args = ap.parse_args()

    session_name = args.session_name or datetime.datetime.now().strftime("%Y%m%d%H%M")
    sdir = Path(args.session_root) / session_name
    sdir.mkdir(parents=True, exist_ok=True)
    video_path = sdir / "events.mp4"
    mocap_path = sdir / "mocap.h5"
    sync_path = sdir / "sync.json"

    try:
        cam = rs.open_camera(args)
    except Exception as e:  # noqa: BLE001
        print(f"\n打开相机失败: {e}\n常见原因: DV-GUI 还开着,先关掉。", file=sys.stderr)
        sys.exit(1)
    res = cam.getEventResolution()
    width, height = res
    cam_name = cam.getCameraName()
    print(f"相机: {cam_name}  {width}x{height}  -> {video_path}  ({args.fps:g}fps)")
    print(f"NatNet: {args.multicast}:{args.port}")

    # 渲染器 + 可选去噪
    vis = dv.visualization.EventVisualizer(res)
    vis.setBackgroundColor(dv.visualization.colors.black())
    on_c, off_c = dv.visualization.colors.lime(), dv.visualization.colors.red()
    if args.swap:
        on_c, off_c = off_c, on_c
    vis.setPositiveColor(on_c)
    vis.setNegativeColor(off_c)
    denoise = not args.no_denoise
    ba_ms = max(0.1, args.ba_ms)
    noise = dv.noise.BackgroundActivityNoiseFilter(
        res, backgroundActivityDuration=datetime.timedelta(milliseconds=ba_ms)) if denoise else None

    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(args.fps), (width, height))
    if not writer.isOpened():
        print("VideoWriter 打开失败(检查 opencv 的编码器)。", file=sys.stderr)
        sys.exit(1)

    show = args.preview
    win = "DVX video"
    if show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, int(width * args.scale), int(height * args.scale))

    st = {"frames": 0, "events": 0}

    def on_frame(events):
        img = vis.generateImage(events)
        writer.write(img)
        st["frames"] += 1
        st["events"] += events.size()
        if show:
            cv2.imshow(win, img)

    # 事件时间切片:每 1/fps(事件时间)出一帧
    slicer = dv.EventStreamSlicer()
    slicer.doEveryTimeInterval(datetime.timedelta(milliseconds=1000.0 / args.fps), on_frame)

    # mocap 后台线程(复用 record_session)
    q = queue.Queue()
    stop_evt = threading.Event()
    mstate = {"ok": None}
    mthread = threading.Thread(target=rs.mocap_worker,
                               args=(q, stop_evt, mstate, args.bind_ip, args.port, args.multicast),
                               daemon=True)
    mthread.start()

    anchor = None
    mocap_pending = []
    mocap_idx = 0
    mocap_bodies = 0
    start = time.time()
    rec_accum = 0.0
    last_t = start
    last_status = start
    quit_key = QuitKey() if not show else None

    print("滤波: " + (f"开 (ba-ms={ba_ms:g})" if denoise else "关"))
    print("开始录视频…  " + ("q/ESC 停止" if show else "Ctrl+C 或终端 q 停止"))

    with h5py.File(mocap_path, "w") as mc_h5:
        mc_ds = rs.make_mocap_datasets(mc_h5, "lzf")
        mc_h5.attrs["format"] = "mocap_natnet_v2"
        mc_h5.attrs["multicast_group"] = args.multicast
        mc_h5.attrs["udp_port"] = int(args.port)
        mc_h5.attrs["time_unit"] = "us_camera_relative"
        mc_h5.attrs["video_fps"] = float(args.fps)
        try:
            while cam.isRunning():
                drain_start = time.time()
                budget = (1.0 / args.fps) if show else 0.1
                got = False
                while True:
                    events = cam.getNextEventBatch()
                    if events is None or events.size() == 0:
                        break
                    got = True
                    if anchor is None:
                        cam_first = int(events.getLowestTime())
                        wall_first = int(time.time() * 1e6)
                        anchor = (cam_first, wall_first)
                        mc_h5.attrs["cam_first_us"] = cam_first
                        mc_h5.attrs["wall_first_us"] = wall_first
                        sync_path.write_text(json.dumps(
                            {"camera_first_us": cam_first, "wall_first_us": wall_first,
                             "video_fps": args.fps}))
                    if denoise:
                        noise.accept(events)
                        events = noise.generateEvents()
                    slicer.accept(events)        # 内部按帧触发 on_frame
                    if time.time() - drain_start >= budget:
                        break

                # mocap 排空 + 对齐写盘
                drained = []
                while True:
                    try:
                        drained.append(q.get_nowait())
                    except queue.Empty:
                        break
                if anchor is None:
                    mocap_pending.extend(drained)
                elif drained or mocap_pending:
                    cam_first, wall_first = anchor
                    for recv_wall_us, parsed in (mocap_pending + drained):
                        t_cam = cam_first + (recv_wall_us - wall_first)
                        mocap_bodies += rs.write_mocap_packet(mc_ds, t_cam, parsed, mocap_idx)
                        mocap_idx += 1
                    mocap_pending.clear()

                now = time.time()
                rec_accum += now - last_t
                last_t = now
                if args.duration > 0 and rec_accum >= args.duration:
                    print("\n到达设定时长,停止。")
                    break

                if show:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                        break
                else:
                    if quit_key is not None and quit_key.pressed():
                        print("\n收到 q,停止。")
                        break
                    if now - last_status >= args.status_every:
                        print(f"  rec {rec_accum:5.1f}s  frames={st['frames']}"
                              f"  events~{st['events']:,}  mocap={mocap_idx}", end="\r", flush=True)
                        last_status = now

                if not got and not drained:
                    time.sleep(0.0005)
        except KeyboardInterrupt:
            print("\n用户中断。")
        finally:
            stop_evt.set()
            if quit_key is not None:
                quit_key.restore()
            mc_h5.attrs["num_packets"] = int(mocap_idx)
            mc_h5.attrs["num_body_observations"] = int(mocap_bodies)
            mc_h5.attrs["duration_wall_s"] = float(rec_accum)
            writer.release()
            if show:
                cv2.destroyAllWindows()

    mthread.join(timeout=1.0)
    print(f"\n已保存: {video_path}  ({st['frames']} 帧)")
    print(f"        {mocap_path}  ({mocap_idx} 包, mocap_ok={mstate['ok']})")
    if mstate["ok"] is False:
        print("  ⚠️ 没收到 mocap:检查 Motive 是否在 streaming。")


if __name__ == "__main__":
    main()
