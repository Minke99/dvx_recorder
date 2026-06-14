import socket
import struct

MCAST_GRP = "239.255.42.99"
PORT = 1511


def read_int(data, offset):
    value = struct.unpack_from("<i", data, offset)[0]
    return value, offset + 4


def read_float(data, offset):
    value = struct.unpack_from("<f", data, offset)[0]
    return value, offset + 4


def read_short(data, offset):
    value = struct.unpack_from("<h", data, offset)[0]
    return value, offset + 2


def parse_frame_of_mocap_data(data):
    offset = 0

    msg_id, packet_size = struct.unpack_from("<HH", data, offset)
    offset += 4

    print(f"\nmsg_id={msg_id}, packet_size={packet_size}, len={len(data)}")

    frame_number, offset = read_int(data, offset)
    print(f"frame={frame_number}")

    n_marker_sets, offset = read_int(data, offset)
    print(f"marker_sets={n_marker_sets}")

    # 这里你的包里是 0，所以直接跳过
    for _ in range(n_marker_sets):
        raise NotImplementedError("暂未处理 marker sets")

    n_other_markers, offset = read_int(data, offset)
    print(f"other_markers={n_other_markers}")

    # 每个 other marker 是 3 个 float
    offset += n_other_markers * 12

    n_rigid_bodies, offset = read_int(data, offset)
    print(f"rigid_bodies={n_rigid_bodies}")

    for i in range(n_rigid_bodies):
        rb_id, offset = read_int(data, offset)

        x, offset = read_float(data, offset)
        y, offset = read_float(data, offset)
        z, offset = read_float(data, offset)

        qx, offset = read_float(data, offset)
        qy, offset = read_float(data, offset)
        qz, offset = read_float(data, offset)
        qw, offset = read_float(data, offset)

        print(
            f"RB[{i}] id={rb_id} "
            f"pos=({x:.4f}, {y:.4f}, {z:.4f}) "
            f"quat=({qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f})"
        )

        # 旧版 NatNet rigid body 后面可能还有 marker 数据
        # 新版一般紧跟 mean_error 和 params
        mean_error, offset = read_float(data, offset)
        params, offset = read_short(data, offset)

        tracking_valid = bool(params & 0x01)

        print(
            f"       mean_error={mean_error:.6f}, "
            f"params={params}, tracking_valid={tracking_valid}"
        )

    return offset


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("", PORT))

mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

print(f"Listening multicast {MCAST_GRP}:{PORT} ...")

while True:
    data, addr = sock.recvfrom(65535)
    print(f"\nfrom={addr}")

    try:
        final_offset = parse_frame_of_mocap_data(data)
        print(f"parsed_bytes={final_offset}")
    except Exception as e:
        print("parse error:", e)
        print("raw first 64 bytes:", data[:64].hex())