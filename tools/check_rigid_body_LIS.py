#!/usr/bin/env python3
"""
监听 Motive / NatNet 广播的刚体 UDP 流，持续打印每个刚体的 NatNet ID 与位置。

默认网络参数与 LIS.UdpReceiver.UdpRigidBodies 相同。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from LIS import UdpReceiver


def main() -> None:
    parser = argparse.ArgumentParser(description="Print all mocap rigid-body poses from NatNet UDP.")
    parser.add_argument("--ip", default="0.0.0.0", help="Bind address (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=1511, help="UDP port (default 1511)")
    parser.add_argument(
        "--multicast",
        default="239.255.42.99",
        help="NatNet multicast group (default 239.255.42.99)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.15,
        help="Seconds between screen updates (default 0.15)",
    )
    parser.add_argument(
        "--only-tracked",
        action="store_true",
        help="If set, match UdpRigidBodies(only_tracked=True): untracked bodies show as zeros",
    )
    args = parser.parse_args()

    udp = UdpReceiver.UdpRigidBodies(
        udp_ip=args.ip,
        udp_port=args.port,
        multicast_group=args.multicast,
        rigid_body_ids=None,
        only_tracked=args.only_tracked,
    )
    udp.start_thread()

    print(
        "Receiving… Move one rigid body at a time to see which NatNet_id / data[i] moves.\n"
        "  data[i] = same index as process_data() dict key i in px4_quad_mocap_traj (sorted by id).\n"
        "Ctrl+C to exit.\n"
    )

    try:
        while True:
            bodies = udp._latest_rigid_bodies
            stamp = time.strftime("%H:%M:%S")
            print(f"\n--- {stamp}  bodies={len(bodies)} ---")
            if not bodies:
                print("  (no rigid bodies yet; check Motive streaming / firewall / port)")
            else:
                for i, rb in enumerate(bodies, start=1):
                    rid = rb.get("id", -1)
                    x, y, z = rb["x"], rb["y"], rb["z"]
                    ok = rb.get("tracking_valid", False)
                    err = rb.get("mean_error", 0.0)
                    print(
                        f"  data[{i}]  NatNet_id={rid:6d}  "
                        f"xyz=({x:8.3f}, {y:8.3f}, {z:8.3f}) m  "
                        f"tracked={ok!s:5}  mean_err={err:.5f}"
                    )
            time.sleep(max(0.02, args.interval))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
