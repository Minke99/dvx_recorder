#!/usr/bin/env python3
"""Record Motive NatNet rigid-body UDP stream to HDF5, timestamps aligned to camera time.

Time alignment:
  /mocap/t is in camera-relative microseconds, same axis as /events/t from record_raw.py.
  Conversion uses a sync file written by record_raw.py on its first event:
      camera_t_us = sync.camera_first_us + (recv_wall_us - sync.wall_first_us)
  Without --sync-from, falls back to wall-clock Unix microseconds in /mocap/t
  (set attr time_unit accordingly).
"""

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from LIS import UdpReceiver
from lib.h5_utils import append_dataset, create_resizable_dataset
from lib.paths import resolve_path
from lib.quit_key import QuitKey


def wait_for_sync(sync_path: Path, timeout_s: float) -> dict:
    """Block until sync_path exists and is parseable, or timeout."""
    deadline = time.time() + timeout_s
    print("waiting for sync file:", sync_path)
    while time.time() < deadline:
        if sync_path.exists():
            try:
                data = json.loads(sync_path.read_text())
                if "camera_first_us" in data and "wall_first_us" in data:
                    print("sync loaded: camera_first_us={}, wall_first_us={}".format(
                        data["camera_first_us"], data["wall_first_us"]))
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        time.sleep(0.05)
    sys.exit("error: sync file {} not available within {}s; "
             "is record_raw.py running with --sync-out?".format(sync_path, timeout_s))


def record(args):
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.sync_from:
        sync_path = resolve_path(args.sync_from)
        sync = wait_for_sync(sync_path, args.sync_wait_s)
        cam_first_us = int(sync["camera_first_us"])
        wall_first_us = int(sync["wall_first_us"])
        time_unit = "us_camera_relative"
    else:
        sync = None
        cam_first_us = 0
        wall_first_us = 0
        time_unit = "us_since_epoch"

    rx = UdpReceiver.UdpRigidBodies(
        udp_ip=args.bind_ip,
        udp_port=args.port,
        multicast_group=args.multicast,
    )

    print("NatNet {}:{}, bind={}".format(args.multicast, args.port, args.bind_ip))
    print("recording mocap to:", output_path)
    print("time axis:", time_unit)
    quit_key = QuitKey()
    print("stop by pressing 'q'" + ("" if quit_key.enabled else " (stdin not a TTY; use SIGTERM/SIGINT)"))
    # short recv timeout so we can poll the quit key
    rx._sock.settimeout(0.1)

    start = time.time()
    total_packets = 0
    total_bodies = 0

    with h5py.File(output_path, "w") as h:
        g = h.create_group("mocap")
        t_ds        = create_resizable_dataset(g, "t",                np.int64)
        frame_ds    = create_resizable_dataset(g, "frame",            np.int64)
        nbodies_ds  = create_resizable_dataset(g, "num_bodies",       np.int32)
        rb_tidx_ds  = create_resizable_dataset(g, "rb_t_idx",         np.int32)
        rb_id_ds    = create_resizable_dataset(g, "rb_id",            np.int32)
        rb_x_ds     = create_resizable_dataset(g, "rb_x",             np.float32)
        rb_y_ds     = create_resizable_dataset(g, "rb_y",             np.float32)
        rb_z_ds     = create_resizable_dataset(g, "rb_z",             np.float32)
        rb_qx_ds    = create_resizable_dataset(g, "rb_qx",            np.float32)
        rb_qy_ds    = create_resizable_dataset(g, "rb_qy",            np.float32)
        rb_qz_ds    = create_resizable_dataset(g, "rb_qz",            np.float32)
        rb_qw_ds    = create_resizable_dataset(g, "rb_qw",            np.float32)
        rb_track_ds = create_resizable_dataset(g, "rb_tracking_valid", np.int8)
        rb_err_ds   = create_resizable_dataset(g, "rb_mean_error",    np.float32)

        h.attrs["format"] = "mocap_natnet_v2"
        h.attrs["multicast_group"] = args.multicast
        h.attrs["udp_port"] = int(args.port)
        h.attrs["start_wall_s"] = float(start)
        h.attrs["time_unit"] = time_unit
        if sync is not None:
            h.attrs["sync_camera_first_us"] = int(cam_first_us)
            h.attrs["sync_wall_first_us"] = int(wall_first_us)

        # Convert SIGINT into a flag so it can't interrupt a packet mid-write
        # and leave datasets at inconsistent lengths.
        stop_requested = {"flag": False}

        def _on_sigint(signum, frame):
            stop_requested["flag"] = True

        prev_sigint = signal.signal(signal.SIGINT, _on_sigint)

        try:
            import socket as _socket
            while True:
                if stop_requested["flag"]:
                    print("\nrecording stopped by signal")
                    break
                if quit_key.pressed():
                    print("\nrecording stopped by 'q'")
                    break
                if args.duration > 0 and time.time() - start >= args.duration:
                    break
                try:
                    raw, _ = rx._sock.recvfrom(rx.len_data)
                except _socket.timeout:
                    continue
                recv_wall_us = int(time.time() * 1e6)
                parsed = rx._parse_frame_of_mocap_data(raw)
                if parsed is None:
                    continue

                if sync is not None:
                    t_stamp = cam_first_us + (recv_wall_us - wall_first_us)
                else:
                    t_stamp = recv_wall_us

                bodies = parsed["rigid_bodies"]
                append_dataset(t_ds,       [t_stamp])
                append_dataset(frame_ds,   [parsed["frame"]])
                append_dataset(nbodies_ds, [len(bodies)])
                if bodies:
                    n = len(bodies)
                    append_dataset(rb_tidx_ds, [total_packets] * n)
                    append_dataset(rb_id_ds,   [b["id"]   for b in bodies])
                    append_dataset(rb_x_ds,    [b["x"]    for b in bodies])
                    append_dataset(rb_y_ds,    [b["y"]    for b in bodies])
                    append_dataset(rb_z_ds,    [b["z"]    for b in bodies])
                    append_dataset(rb_qx_ds,   [b["qx"]   for b in bodies])
                    append_dataset(rb_qy_ds,   [b["qy"]   for b in bodies])
                    append_dataset(rb_qz_ds,   [b["qz"]   for b in bodies])
                    append_dataset(rb_qw_ds,   [b["qw"]   for b in bodies])
                    append_dataset(rb_track_ds, [1 if b.get("tracking_valid") else 0 for b in bodies])
                    append_dataset(rb_err_ds,  [b.get("mean_error", 0.0) for b in bodies])
                    total_bodies += n

                total_packets += 1
                if total_packets % args.status_every == 0:
                    print("packets={}, body_obs={}".format(total_packets, total_bodies))
        except KeyboardInterrupt:
            print("\nrecording stopped by user")
        finally:
            quit_key.restore()
            signal.signal(signal.SIGINT, prev_sigint)

        h.attrs["duration_wall_s"] = float(time.time() - start)
        h.attrs["num_packets"] = int(total_packets)
        h.attrs["num_body_observations"] = int(total_bodies)

    print("saved {}, packets={}, body_obs={}".format(output_path, total_packets, total_bodies))


def parse_args():
    parser = argparse.ArgumentParser(description="Record Motive NatNet mocap to HDF5, aligned to camera time.")
    parser.add_argument("--multicast", default="239.255.42.99")
    parser.add_argument("--port", type=int, default=1511)
    parser.add_argument("--bind-ip", default="0.0.0.0")
    parser.add_argument("--output", default="recordings/mocap.h5",
                        help="H5 output path (relative to this folder).")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Recording duration in seconds; <=0 runs until Ctrl+C.")
    parser.add_argument("--status-every", type=int, default=200,
                        help="Print progress every N packets.")
    parser.add_argument("--sync-from", default=None,
                        help="JSON sync file from record_raw.py --sync-out. "
                             "If set, /mocap/t is in camera-relative microseconds.")
    parser.add_argument("--sync-wait-s", type=float, default=30.0,
                        help="Seconds to wait for sync file before giving up.")
    return parser.parse_args()


if __name__ == "__main__":
    record(parse_args())
