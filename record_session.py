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
HDF5 用 gzip+shuffle 压缩、x/y 用 int16,文件比未压缩小约 5 倍。

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
import socket
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

def create_ds(group, name, dtype):
    # gzip(level 1) + shuffle:对事件这种 (单调 t / 小坐标) 数据能压到约 1/5,基本无损且很快
    return group.create_dataset(name, shape=(0,), maxshape=(None,),
                                dtype=dtype, chunks=(1 << 16,),
                                compression="gzip", compression_opts=1, shuffle=True)


def append_ds(ds, arr):
    n = ds.shape[0]
    arr = np.asarray(arr, dtype=ds.dtype)
    ds.resize((n + len(arr),))
    ds[n:] = arr


def make_event_datasets(h5):
    g = h5.create_group("events")
    return {
        # x,y 本来就是 int16(分辨率内),不必用 int32
        "x": create_ds(g, "x", np.int16),
        "y": create_ds(g, "y", np.int16),
        "t": create_ds(g, "t", np.int64),
        "p": create_ds(g, "p", np.int8),
    }


def make_mocap_datasets(h5):
    g = h5.create_group("mocap")
    names_i64 = ["t", "frame"]
    names_i32 = ["num_bodies", "rb_t_idx", "rb_id"]
    names_f32 = ["rb_x", "rb_y", "rb_z", "rb_qx", "rb_qy", "rb_qz", "rb_qw", "rb_mean_error"]
    d = {}
    for n in names_i64:
        d[n] = create_ds(g, n, np.int64)
    for n in names_i32:
        d[n] = create_ds(g, n, np.int32)
    for n in names_f32:
        d[n] = create_ds(g, n, np.float32)
    d["rb_tracking_valid"] = create_ds(g, "rb_tracking_valid", np.int8)
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
    # 预览(默认关:只录盘;加 --preview 才开实时窗口)
    ap.add_argument("--preview", action="store_true",
                    help="开实时预览窗口(默认关:只录盘,避免高事件率下显示拖慢相机排空)")
    ap.add_argument("--status-every", type=float, default=1.0, help="无预览时每隔几秒打印一次状态")
    ap.add_argument("--fps", type=float, default=30.0, help="预览刷新率")
    ap.add_argument("--scale", type=float, default=1.0, help="窗口放大倍数")
    ap.add_argument("--swap", action="store_true", help="交换颜色(红=ON,绿=OFF)")
    ap.add_argument("--ba-ms", type=float, default=3.0, dest="ba_ms", help="预览去噪相关窗口(ms),越小越狠")
    ap.add_argument("--no-denoise", action="store_true", help="预览不去噪")
    ap.add_argument("--serial", default=None, help="指定相机序列号")
    args = ap.parse_args()

    session_name = args.session_name or datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
    preview_denoise = not args.no_denoise
    vis = None
    noise = None
    if show:
        # 预览渲染器 + 去噪(只有开预览时才需要)
        vis = dv.visualization.EventVisualizer(res)
        vis.setBackgroundColor(dv.visualization.colors.black())
        on_c, off_c = dv.visualization.colors.lime(), dv.visualization.colors.red()
        if args.swap:
            on_c, off_c = off_c, on_c
        vis.setPositiveColor(on_c)
        vis.setNegativeColor(off_c)
        noise = dv.noise.BackgroundActivityNoiseFilter(
            res, backgroundActivityDuration=datetime.timedelta(milliseconds=max(0.1, args.ba_ms)))
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, int(width * args.scale), int(height * args.scale))
        print("实时预览: 开(q/ESC 停止 | d 开关去噪)")
    else:
        print("实时预览: 关 —— 只录盘。停止: 终端按 q 或 Ctrl+C(或用 --duration)")

    st = {"rec_s": 0.0, "events": 0, "mocap_pkts": 0, "mocap_bodies": 0, "mocap_ok": None, "eps": 0}

    def render(events):
        img = vis.generateImage(events)
        m = "mocap:waiting" if st["mocap_ok"] is None else (
            f"mocap:{st['mocap_pkts']}" if st["mocap_ok"] else "mocap:OFF")
        cv2.circle(img, (16, 18), 7, (0, 0, 255), -1)
        cv2.putText(img, f"REC {st['rec_s']:5.1f}s  {st['events']:,}ev  {st['eps']/1e6:.1f}Meps  {m}",
                    (30, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, img)

    # 预览按墙钟节流(不用 EventStreamSlicer,避免被事件时间牵着走累积延时)
    frame_dt = 1.0 / args.fps
    preview_buf = dv.EventStore()   # 累积「上次渲染以来」的事件,渲染时一次性处理

    # 启动 mocap 线程
    q = queue.Queue()
    stop_evt = threading.Event()
    mstate = {"ok": None}
    mthread = threading.Thread(target=mocap_worker,
                               args=(q, stop_evt, mstate, args.bind_ip, args.port, args.multicast),
                               daemon=True)
    mthread.start()

    anchor = None              # (cam_first_us, wall_first_us)
    mocap_pending = []         # 锚点确立前先缓存的 mocap 包
    mocap_idx = 0
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
    with h5py.File(events_path, "w") as ev_h5, h5py.File(mocap_path, "w") as mc_h5:
        ev_ds = make_event_datasets(ev_h5)
        mc_ds = make_mocap_datasets(mc_h5)
        ev_h5.attrs["camera_name"] = cam_name
        ev_h5.attrs["resolution_width"] = int(width)
        ev_h5.attrs["resolution_height"] = int(height)
        ev_h5.attrs["time_unit"] = "us_camera_relative"
        ev_h5.attrs["format"] = "raw_dvx_events_v1"
        mc_h5.attrs["format"] = "mocap_natnet_v2"
        mc_h5.attrs["multicast_group"] = args.multicast
        mc_h5.attrs["udp_port"] = int(args.port)
        mc_h5.attrs["time_unit"] = "us_camera_relative"

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
                        for k, v in (("cam_first_us", cam_first), ("wall_first_us", wall_first)):
                            ev_h5.attrs[k] = v
                            mc_h5.attrs[k] = v
                        sync_path.write_text(json.dumps(
                            {"camera_first_us": cam_first, "wall_first_us": wall_first}))
                    st["events"] += write_event_batch(ev_ds, events)
                    ev_packets += 1
                    if show:
                        preview_buf.add(events)
                    if time.time() - drain_start >= budget:
                        break

                # 2) 排空 mocap 队列并写盘
                st["mocap_ok"] = mstate["ok"]
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
                        st["mocap_bodies"] += write_mocap_packet(mc_ds, t_cam, parsed, mocap_idx)
                        mocap_idx += 1
                    st["mocap_pkts"] = mocap_idx
                    mocap_pending.clear()

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
                        if preview_denoise and ev.size() > 0:
                            noise.accept(ev)
                            ev = noise.generateEvents()
                        render(ev)
                        last_render = now
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q")):
                            break
                        elif key == ord("d"):
                            preview_denoise = not preview_denoise
                            print(f"\n预览去噪 -> {'开' if preview_denoise else '关'}")
                        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                            break
                else:
                    if quit_key is not None and quit_key.pressed():
                        print("\n收到 q,停止。")
                        break
                    if now - last_status >= args.status_every:
                        print(f"  rec {rec_accum:5.1f}s  events={st['events']:,}"
                              f"  {st['eps']/1e6:.2f}Meps  mocap={mocap_idx}", end="\r", flush=True)
                        last_status = now

                if not got_any and not drained:
                    time.sleep(0.0005)
        except KeyboardInterrupt:
            print("\n用户中断。")
        finally:
            stop_evt.set()
            if quit_key is not None:
                quit_key.restore()
            ev_h5.attrs["num_events"] = int(st["events"])
            ev_h5.attrs["num_packets"] = int(ev_packets)
            ev_h5.attrs["duration_wall_s"] = float(rec_accum)
            mc_h5.attrs["num_packets"] = int(mocap_idx)
            mc_h5.attrs["num_body_observations"] = int(st["mocap_bodies"])
            mc_h5.attrs["duration_wall_s"] = float(rec_accum)
            if show:
                cv2.destroyAllWindows()

    mthread.join(timeout=1.0)
    print(f"\n已保存 session: {session_dir}")
    print(f"  events.h5 : {st['events']:,} 事件 / {ev_packets} 批次")
    print(f"  mocap.h5  : {mocap_idx:,} 包 / {st['mocap_bodies']:,} 刚体观测  (mocap_ok={mstate['ok']})")
    if mstate["ok"] is False:
        print("  ⚠️ 没收到 mocap:检查 Motive 是否在广播、组播/端口/网卡是否正确。")
    elif mocap_idx == 0:
        print("  ⚠️ mocap 包数为 0:确认 Motive 正在 streaming(NatNet)。")
    print(f"  校验对齐: python tools/check_session.py {session_dir}")


if __name__ == "__main__":
    main()
