#!/usr/bin/env python3
"""
同步录制 DVX 事件 + Motive/OptiTrack NatNet mocap,时间对齐,存成一个 session 目录。

单进程:
  - 主线程:读相机事件 -> 实时预览(黑底/绿ON/红OFF,可去噪)-> 写 events.h5
  - 后台线程:收 mocap UDP(LIS.UdpReceiver)-> 丢进队列
  - 主线程把每包 mocap 用 cam_first + (recv_wall - wall_first) 换算到相机时间轴 -> 写 mocap.h5

时间对齐:
  以「第一批事件」为锚点:
      cam_first_us = 第一颗事件的相机时间戳
      wall_first_us = 收到第一批事件时的墙钟(us)
  之后 mocap 的每包到达墙钟 recv_wall_us 换算为相机时间:
      t_cam = cam_first_us + (recv_wall_us - wall_first_us)
  于是 events/t 与 mocap/t 落在同一条「相机相对微秒」时间轴上。

输出:
    recordings/<session>/events.h5     events/x,y,t,p
    recordings/<session>/mocap.h5      mocap/t,frame,num_bodies,rb_*
    recordings/<session>/sync.json     {cam_first_us, wall_first_us}
  可直接用 tools/check_session.py 校验对齐,tools/depack_h5_data.py 解包。

依赖: pip install dv-processing opencv-python numpy h5py tqdm

默认【不显示实时画面】,只录盘(高事件率下也稳)。加 --preview 才开窗口。
默认【对存盘的事件做去噪】(BackgroundActivityNoiseFilter,--ba-ms 默认 3.0);
加 --no-denoise 则存原始事件。
HDF5 压缩可选(--compress lzf/gzip/none,默认 lzf)+ x/y 用 int16。
注意:慢设备(如 Jetson)用 gzip 可能因 CPU 跟不上而丢事件(回放出现黑帧)
-> 改用 lzf(默认)或 none。

用法:
    python record_session.py                      # 只录盘(无窗口),Ctrl+C 或终端 q 停止
    python record_session.py --duration 10        # 录 10 秒后自动停
    python record_session.py --preview            # 开实时预览窗口(q/ESC 停, d 开关去噪)
    python record_session.py --session-name take1
    python record_session.py --port 1511 --multicast 239.255.42.99   # NatNet 网络参数

无预览时终端每秒打印 rec 时长 / 事件数 / Meps / mocap 包数;Ctrl+C 或按 q 停止。
"""

import argparse
import datetime
import json
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import h5py
import dv_processing as dv

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from LIS import UdpReceiver
from lib.quit_key import QuitKey


# --------------------------- h5 helpers (模块级,便于测试) ---------------------------

_COMPRESS = {
    # gzip: 最省盘但最吃 CPU(慢设备如 Jetson 可能跟不上 -> 丢事件)
    "gzip": dict(compression="gzip", compression_opts=1, shuffle=True),
    # lzf: 快很多、压缩约 3 倍,两边都能跑(默认)
    "lzf": dict(compression="lzf", shuffle=True),
    # none: 最快、最大,慢设备保命用
    "none": dict(),
}


def create_ds(group, name, dtype, compress="lzf"):
    return group.create_dataset(name, shape=(0,), maxshape=(None,),
                                dtype=dtype, chunks=(1 << 16,),
                                **_COMPRESS.get(compress, _COMPRESS["lzf"]))


def append_ds(ds, arr):
    n = ds.shape[0]
    arr = np.asarray(arr, dtype=ds.dtype)
    ds.resize((n + len(arr),))
    ds[n:] = arr


def make_event_datasets(h5, compress="lzf"):
    g = h5.create_group("events")
    return {
        # x,y 本来就是 int16(分辨率内),不必用 int32
        "x": create_ds(g, "x", np.int16, compress),
        "y": create_ds(g, "y", np.int16, compress),
        "t": create_ds(g, "t", np.int64, compress),
        "p": create_ds(g, "p", np.int8, compress),
    }


def make_mocap_datasets(h5, compress="lzf"):
    g = h5.create_group("mocap")
    names_i64 = ["t", "frame"]
    names_i32 = ["num_bodies", "rb_t_idx", "rb_id"]
    names_f32 = ["rb_x", "rb_y", "rb_z", "rb_qx", "rb_qy", "rb_qz", "rb_qw", "rb_mean_error"]
    d = {}
    for n in names_i64:
        d[n] = create_ds(g, n, np.int64, compress)
    for n in names_i32:
        d[n] = create_ds(g, n, np.int32, compress)
    for n in names_f32:
        d[n] = create_ds(g, n, np.float32, compress)
    d["rb_tracking_valid"] = create_ds(g, "rb_tracking_valid", np.int8, compress)
    return d


def write_event_batch(dsets, events):
    """把一批事件写进 events 数据集。返回写入条数。"""
    a = events.numpy()
    if len(a) == 0:
        return 0
    append_ds(dsets["x"], a["x"])
    append_ds(dsets["y"], a["y"])
    append_ds(dsets["t"], a["timestamp"])
    append_ds(dsets["p"], a["polarity"])
    return len(a)


def write_mocap_packet(dsets, t_cam, parsed, pkt_idx):
    """写一包 mocap(已换算到相机时间轴 t_cam)。返回该包刚体数。"""
    bodies = parsed["rigid_bodies"]
    append_ds(dsets["t"], [t_cam])
    append_ds(dsets["frame"], [parsed["frame"]])
    append_ds(dsets["num_bodies"], [len(bodies)])
    if bodies:
        n = len(bodies)
        append_ds(dsets["rb_t_idx"], [pkt_idx] * n)
        append_ds(dsets["rb_id"], [b["id"] for b in bodies])
        append_ds(dsets["rb_x"], [b["x"] for b in bodies])
        append_ds(dsets["rb_y"], [b["y"] for b in bodies])
        append_ds(dsets["rb_z"], [b["z"] for b in bodies])
        append_ds(dsets["rb_qx"], [b["qx"] for b in bodies])
        append_ds(dsets["rb_qy"], [b["qy"] for b in bodies])
        append_ds(dsets["rb_qz"], [b["qz"] for b in bodies])
        append_ds(dsets["rb_qw"], [b["qw"] for b in bodies])
        append_ds(dsets["rb_mean_error"], [b.get("mean_error", 0.0) for b in bodies])
        append_ds(dsets["rb_tracking_valid"], [1 if b.get("tracking_valid") else 0 for b in bodies])
    return len(bodies)


# --------------------------- mocap UDP 后台线程 ---------------------------

def mocap_worker(q, stop_evt, state, bind_ip, port, multicast):
    """后台:收 NatNet UDP,解析后把 (recv_wall_us, parsed) 放进队列。"""
    try:
        rx = UdpReceiver.UdpRigidBodies(udp_ip=bind_ip, udp_port=port, multicast_group=multicast)
    except Exception as e:  # noqa: BLE001
        print(f"\n[mocap] 初始化失败: {e}\n[mocap] 只录事件,不录 mocap。", file=sys.stderr)
        state["ok"] = False
        return
    state["ok"] = True
    rx._sock.settimeout(0.2)
    while not stop_evt.is_set():
        try:
            raw, _ = rx._sock.recvfrom(rx.len_data)
        except socket.timeout:
            continue
        except OSError:
            break
        recv_wall_us = int(time.time() * 1e6)
        parsed = rx._parse_frame_of_mocap_data(raw)
        if parsed is None:
            continue
        q.put((recv_wall_us, parsed))
    try:
        rx._sock.close()
    except OSError:
        pass


# --------------------------- 主流程 ---------------------------

def open_camera(args):
    if args.serial:
        return dv.io.camera.open(args.serial)
    return dv.io.camera.open()


def main():
    ap = argparse.ArgumentParser(description="同步录制 DVX 事件 + NatNet mocap(时间对齐)")
    ap.add_argument("--session-name", default=None, help="recordings/ 下的子目录名(默认时间戳)")
    ap.add_argument("--session-root", default="recordings", help="session 目录的父目录")
    ap.add_argument("--duration", type=float, default=0.0, help="录制秒数,<=0 表示录到按 q")
    # mocap / NatNet
    ap.add_argument("--multicast", default="239.255.42.99", help="NatNet 组播地址")
    ap.add_argument("--port", type=int, default=1511, help="NatNet 端口")
    ap.add_argument("--bind-ip", default="0.0.0.0", help="本地绑定 IP")
    ap.add_argument("--no-mocap", action="store_true", help="完全不录 mocap(只录事件;排查丢包用)")
    # 预览(默认关:只录盘;加 --preview 才开实时窗口)
    ap.add_argument("--preview", action="store_true",
                    help="开实时预览窗口(默认关:只录盘,避免高事件率下显示拖慢相机排空)")
    ap.add_argument("--status-every", type=float, default=1.0, help="无预览时每隔几秒打印一次状态")
    ap.add_argument("--compress", choices=["gzip", "lzf", "none"], default="lzf",
                    help="h5 压缩: lzf(默认,快+小,两边都能跑) / gzip(最小但吃CPU,慢设备会丢事件) / none(最快最大)")
    ap.add_argument("--fps", type=float, default=30.0, help="预览刷新率")
    ap.add_argument("--scale", type=float, default=1.0, help="窗口放大倍数")
    ap.add_argument("--swap", action="store_true", help="交换颜色(红=ON,绿=OFF)")
    ap.add_argument("--ba-ms", type=float, default=3.0, dest="ba_ms",
                    help="去噪相关窗口(ms),越小去得越狠 (默认 3.0)。作用于【存盘的事件】")
    ap.add_argument("--no-denoise", action="store_true",
                    help="不滤波,存原始事件(默认会先用 BackgroundActivityNoiseFilter 去噪再存)")
    ap.add_argument("--serial", default=None, help="指定相机序列号")
    args = ap.parse_args()

    session_name = args.session_name or datetime.datetime.now().strftime("%Y%m%d%H%M")
    session_dir = Path(args.session_root) / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    events_path = session_dir / "events.h5"
    mocap_path = session_dir / "mocap.h5"
    sync_path = session_dir / "sync.json"

    try:
        cam = open_camera(args)
    except Exception as e:  # noqa: BLE001
        print(f"\n打开相机失败: {e}\n最常见原因: DV-GUI 还开着,先关掉再跑。", file=sys.stderr)
        sys.exit(1)
    res = cam.getEventResolution()
    width, height = res
    cam_name = cam.getCameraName()
    print(f"相机: {cam_name}  分辨率: {width}x{height}")
    print(f"session 目录: {session_dir}")
    print(f"NatNet: {args.multicast}:{args.port} (bind {args.bind_ip})")

    show = args.preview
    win = "DVX + mocap session"
    # 去噪默认开,且作用于【存盘的事件】;--no-denoise 才存原始
    denoise = not args.no_denoise
    ba_ms = max(0.1, args.ba_ms)
    noise = None
    if denoise:
        noise = dv.noise.BackgroundActivityNoiseFilter(
            res, backgroundActivityDuration=datetime.timedelta(milliseconds=ba_ms))
    vis = None
    if show:
        vis = dv.visualization.EventVisualizer(res)
        vis.setBackgroundColor(dv.visualization.colors.black())
        on_c, off_c = dv.visualization.colors.lime(), dv.visualization.colors.red()
        if args.swap:
            on_c, off_c = off_c, on_c
        vis.setPositiveColor(on_c)
        vis.setNegativeColor(off_c)
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, int(width * args.scale), int(height * args.scale))
    print("滤波: " + (f"开 (ba-ms={ba_ms:g},存去噪后事件)" if denoise else "关 (存原始事件)"))
    print(f"压缩: {args.compress}" + ("  (慢设备如 Jetson 建议 lzf 或 none,避免丢事件)" if args.compress == "gzip" else ""))
    print("实时预览: " + ("开 (q/ESC 停止)" if show else "关 —— 只录盘。停止: 终端 q 或 Ctrl+C"))

    st = {"rec_s": 0.0, "events": 0, "mocap_pkts": 0, "mocap_bodies": 0, "mocap_ok": None, "eps": 0}

    def render(events):
        img = vis.generateImage(events)
        cv2.circle(img, (16, 18), 7, (0, 0, 255), -1)
        cv2.putText(img, f"REC {st['rec_s']:5.1f}s  {st['events']:,}ev  {st['eps']/1e6:.1f}Meps",
                    (30, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, img)

    # 预览按墙钟节流(不用 EventStreamSlicer,避免被事件时间牵着走累积延时)
    frame_dt = 1.0 / args.fps
    preview_buf = dv.EventStore()   # 累积「上次渲染以来」的事件,渲染时一次性处理

    # mocap 放到【独立子进程】里(各自 GIL,不会卡住本进程的相机采集),靠 sync.json 对齐。
    use_mocap = not args.no_mocap
    mocap_proc = None
    if use_mocap:
        mocap_cmd = [sys.executable, str(_REPO_ROOT / "record_mocap.py"),
                     "--sync-from", str(sync_path.resolve()),
                     "--output", str(mocap_path.resolve()),
                     "--multicast", args.multicast,
                     "--port", str(args.port),
                     "--bind-ip", args.bind_ip]
        if args.duration > 0:
            mocap_cmd += ["--duration", str(args.duration)]
        mocap_proc = subprocess.Popen(mocap_cmd, cwd=str(_REPO_ROOT), stdin=subprocess.DEVNULL)
        print(f"mocap: 独立子进程 PID={mocap_proc.pid}(等 sync.json 后开始录 {mocap_path.name})")
    else:
        print("mocap: 关(--no-mocap,只录事件)")

    anchor = None              # (cam_first_us, wall_first_us)
    start_wall = time.time()
    rec_accum = 0.0
    last_t = start_wall
    last_render = start_wall
    last_status = start_wall
    last_eps_t = start_wall
    last_ev_count = 0
    ev_packets = 0
    quit_key = QuitKey() if not show else None   # 无预览时用终端 q 停止

    print("开始录制…  (Ctrl+C 随时停止)")
    with h5py.File(events_path, "w") as ev_h5:
        ev_ds = make_event_datasets(ev_h5, args.compress)
        ev_h5.attrs["camera_name"] = cam_name
        ev_h5.attrs["resolution_width"] = int(width)
        ev_h5.attrs["resolution_height"] = int(height)
        ev_h5.attrs["time_unit"] = "us_camera_relative"
        ev_h5.attrs["format"] = "raw_dvx_events_v1"
        ev_h5.attrs["denoised"] = bool(denoise)
        ev_h5.attrs["compress"] = args.compress
        if denoise:
            ev_h5.attrs["denoise_ba_ms"] = float(ba_ms)

        try:
            while cam.isRunning():
                # 1) 尽量排空相机缓冲并写盘(中途不碰 GUI),避免高事件率下排不完导致延时累积
                got_any = False
                drain_start = time.time()
                budget = frame_dt if show else 0.1
                while True:
                    events = cam.getNextEventBatch()
                    if events is None or events.size() == 0:
                        break
                    got_any = True
                    if anchor is None:
                        cam_first = int(events.getLowestTime())
                        wall_first = int(time.time() * 1e6)
                        anchor = (cam_first, wall_first)
                        ev_h5.attrs["cam_first_us"] = cam_first
                        ev_h5.attrs["wall_first_us"] = wall_first
                        # 写 sync.json -> mocap 子进程读它对齐(用 .tmp 原子替换,避免读到半截)
                        tmp = sync_path.with_suffix(".json.tmp")
                        tmp.write_text(json.dumps(
                            {"camera_first_us": cam_first, "wall_first_us": wall_first}))
                        tmp.replace(sync_path)
                    if denoise:
                        noise.accept(events)
                        events = noise.generateEvents()   # 滤波后再存(预览同源)
                    st["events"] += write_event_batch(ev_ds, events)
                    ev_packets += 1
                    if show and events.size() > 0:
                        preview_buf.add(events)
                    if time.time() - drain_start >= budget:
                        break

                # 3) 计时 / 时长 / eps
                now = time.time()
                rec_accum += now - last_t
                last_t = now
                st["rec_s"] = rec_accum
                if now - last_eps_t >= 0.5:
                    st["eps"] = int((st["events"] - last_ev_count) / (now - last_eps_t))
                    last_ev_count = st["events"]
                    last_eps_t = now
                if args.duration > 0 and rec_accum >= args.duration:
                    print("\n到达设定时长,停止。")
                    break

                # 4) GUI 渲染(仅 --preview,墙钟节流)或 无预览时的停止键/状态
                if show:
                    if now - last_render >= frame_dt:
                        ev = preview_buf
                        preview_buf = dv.EventStore()
                        render(ev)   # preview_buf 已是滤波后的事件,直接画
                        last_render = now
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
                        mtag = "" if not use_mocap else (
                            " mocap:run" if (mocap_proc and mocap_proc.poll() is None) else " mocap:exit")
                        print(f"  rec {rec_accum:5.1f}s  events={st['events']:,}"
                              f"  {st['eps']/1e6:.2f}Meps{mtag}", end="\r", flush=True)
                        last_status = now

                if not got_any:
                    time.sleep(0.0005)
        except KeyboardInterrupt:
            print("\n用户中断。")
        finally:
            if quit_key is not None:
                quit_key.restore()
            ev_h5.attrs["num_events"] = int(st["events"])
            ev_h5.attrs["num_packets"] = int(ev_packets)
            ev_h5.attrs["duration_wall_s"] = float(rec_accum)
            if show:
                cv2.destroyAllWindows()

    # 停 mocap 子进程
    if mocap_proc is not None and mocap_proc.poll() is None:
        try:
            mocap_proc.send_signal(signal.SIGINT)   # record_mocap 收到会写完 attrs 再退出
            mocap_proc.wait(timeout=5)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            mocap_proc.kill()

    print(f"\n已保存 session: {session_dir}")
    print(f"  events.h5 : {st['events']:,} 事件 / {ev_packets} 批次")
    if use_mocap:
        if not mocap_path.exists():
            print("  ⚠️ mocap.h5 没生成:确认 Motive 在 streaming、组播/端口/网卡正确。")
        else:
            print(f"  mocap.h5  : 由子进程录制(独立进程,不抢事件采集)")
    print(f"  校验对齐: python tools/check_session.py {session_dir}")


if __name__ == "__main__":
    main()
