import socket
import struct
import time
import threading
import numpy as np
from tqdm import tqdm


class UdpRigidBodies(object):
    """
    Compat layer:
    Receive OptiTrack NatNet UDP packets directly,
    repack rigid body data into the old custom UDP format:

    per rigid body:
        int16 x, y, z, qx, qy, qz, qw
    total 14 bytes per body

    Scaling matches the old DataProcessor:
        x,y,z  : float meters -> int16 by /0.0005
        qx..qw : float        -> int16 by /0.001
    """

    def __init__(
        self,
        udp_ip="0.0.0.0",
        udp_port=1511,
        multicast_group="239.255.42.99",
        rigid_body_ids=None,
        only_tracked=False,
    ):
        self.len_data = 65535
        self.udp_flag = 0
        self._udpStop = False
        self._udp_data = None
        self._udp_data_time = time.time()
        self._udpThread = None
        self._udpThread_on = False

        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.multicast_group = multicast_group

        # If provided, output order follows this list exactly.
        # Example: [1069, 1074, 1075]
        self.rigid_body_ids = rigid_body_ids

        # If True, invalid rigid bodies become zeros
        # If False, use whatever pose is in the packet
        self.only_tracked = only_tracked

        self.num_bodies = 0
        self._latest_rigid_bodies = []
        self._id_to_index = {}

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.udp_ip, self.udp_port))

        mreq = struct.pack("4sl", socket.inet_aton(self.multicast_group), socket.INADDR_ANY)
        self._sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

        self.sample_rate = -1
        self.sample_rate = self.get_sample_rate()
        self.sample_time = 1.0 / self.sample_rate if self.sample_rate > 0 else 0.01
        print("UDP receiver initialized")

    # -------------------------
    # basic binary readers
    # -------------------------
    def _read_int(self, data, offset):
        value = struct.unpack_from("<i", data, offset)[0]
        return value, offset + 4

    def _read_float(self, data, offset):
        value = struct.unpack_from("<f", data, offset)[0]
        return value, offset + 4

    def _read_short(self, data, offset):
        value = struct.unpack_from("<h", data, offset)[0]
        return value, offset + 2

    def _read_cstring(self, data, offset):
        end = data.index(b"\0", offset)
        value = data[offset:end].decode("utf-8", errors="ignore")
        return value, end + 1

    # -------------------------
    # helpers
    # -------------------------
    def _clamp_int16(self, x):
        return max(-32768, min(32767, int(round(x))))

    def _pack_old_body_format(self, rb):
        """
        Convert one rigid body dict into old 14-byte format:
        h h h h h h h
        """
        if self.only_tracked and (not rb["tracking_valid"]):
            vals = [0, 0, 0, 0, 0, 0, 0]
        else:
            vals = [
                self._clamp_int16(rb["x"] / 0.0005),
                self._clamp_int16(rb["y"] / 0.0005),
                self._clamp_int16(rb["z"] / 0.0005),
                self._clamp_int16(rb["qx"] / 0.001),
                self._clamp_int16(rb["qy"] / 0.001),
                self._clamp_int16(rb["qz"] / 0.001),
                self._clamp_int16(rb["qw"] / 0.001),
            ]
        return struct.pack("<hhhhhhh", *vals)

    # -------------------------
    # NatNet parsing
    # -------------------------
    def _parse_rigid_body(self, data, offset):
        rb_id, offset = self._read_int(data, offset)

        x, offset = self._read_float(data, offset)
        y, offset = self._read_float(data, offset)
        z, offset = self._read_float(data, offset)

        qx, offset = self._read_float(data, offset)
        qy, offset = self._read_float(data, offset)
        qz, offset = self._read_float(data, offset)
        qw, offset = self._read_float(data, offset)

        mean_error, offset = self._read_float(data, offset)
        params, offset = self._read_short(data, offset)
        tracking_valid = bool(params & 0x01)

        rb = {
            "id": rb_id,
            "x": x,
            "y": y,
            "z": z,
            "qx": qx,
            "qy": qy,
            "qz": qz,
            "qw": qw,
            "mean_error": mean_error,
            "params": params,
            "tracking_valid": tracking_valid,
        }
        return rb, offset

    def _parse_frame_of_mocap_data(self, data):
        """
        Parse enough of NatNet frame message to extract rigid bodies.
        Assumes NatNet frame packet (msg_id == 7).
        """
        offset = 0
        msg_id, packet_size = struct.unpack_from("<HH", data, offset)
        offset += 4

        if msg_id != 7:
            return None

        frame_number, offset = self._read_int(data, offset)

        # marker sets
        n_marker_sets, offset = self._read_int(data, offset)
        for _ in range(n_marker_sets):
            _, offset = self._read_cstring(data, offset)
            n_markers, offset = self._read_int(data, offset)
            offset += n_markers * 12  # 3 floats each

        # unlabeled markers
        n_other_markers, offset = self._read_int(data, offset)
        offset += n_other_markers * 12

        # rigid bodies
        n_rigid_bodies, offset = self._read_int(data, offset)
        rigid_bodies = []
        for _ in range(n_rigid_bodies):
            rb, offset = self._parse_rigid_body(data, offset)
            rigid_bodies.append(rb)

        return {
            "frame": frame_number,
            "n_rigid_bodies": n_rigid_bodies,
            "rigid_bodies": rigid_bodies,
            "packet_size": packet_size,
        }

    def _build_output_packet(self, rigid_bodies):
        """
        Output order:
        1) if rigid_body_ids provided -> follow that exact id order
        2) else -> sort by id, stable for old index-based access
        """
        if self.rigid_body_ids is not None:
            ordered_ids = list(self.rigid_body_ids)
            rb_map = {rb["id"]: rb for rb in rigid_bodies}

            out = bytearray()
            ordered_bodies = []
            for rb_id in ordered_ids:
                rb = rb_map.get(rb_id, {
                    "id": rb_id,
                    "x": 0.0, "y": 0.0, "z": 0.0,
                    "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 0.0,
                    "mean_error": 0.0,
                    "params": 0,
                    "tracking_valid": False,
                })
                ordered_bodies.append(rb)
                out.extend(self._pack_old_body_format(rb))
        else:
            ordered_bodies = sorted(rigid_bodies, key=lambda rb: rb["id"])
            out = bytearray()
            for rb in ordered_bodies:
                out.extend(self._pack_old_body_format(rb))

        self._latest_rigid_bodies = ordered_bodies
        self.num_bodies = len(ordered_bodies)
        self._id_to_index = {rb["id"]: i for i, rb in enumerate(ordered_bodies)}

        return bytes(out)

    # -------------------------
    # public API
    # -------------------------
    def get_sample_rate(self):
        if self.sample_rate != -1:
            return self.sample_rate

        print("Computing sample rate...")
        time_list = []

        for _ in tqdm(range(20), desc="Processing...", leave=True, position=0):
            raw, _ = self._sock.recvfrom(self.len_data)
            now = time.time()
            time_list.append(now)

            parsed = self._parse_frame_of_mocap_data(raw)
            if parsed is not None:
                self._udp_data = self._build_output_packet(parsed["rigid_bodies"])
                self._udp_data_time = now

        if len(time_list) < 2:
            print("Sample rate estimation failed, fallback to 100 Hz")
            return 100.0

        d_time = np.diff(time_list)
        sample_time = np.mean(d_time)
        sample_rate = 1.0 / sample_time if sample_time > 0 else 100.0

        print("Sample rate: %.2f Hz" % sample_rate)
        if self._udp_data is not None:
            print("Compat UDP data size: %d bytes" % len(self._udp_data))
            print("Number of rigid bodies: %d" % (len(self._udp_data) // 14))

        return sample_rate

    def start_thread(self):
        if not self._udpThread_on:
            self._udpStop = False
            self._udpThread = threading.Thread(target=self._udp_worker, args=(), daemon=True)
            self._udpThread.start()
            self._udpThread_on = True
            time.sleep(0.2)
            print("Upd thread start")
            print("Number of rigid bodies: %d" % self.num_bodies)
        else:
            print("New upd thread is not started")

    def _udp_worker(self):
        while not self._udpStop:
            self.udp_flag += 1
            raw, _ = self._sock.recvfrom(self.len_data)
            now = time.time()

            parsed = self._parse_frame_of_mocap_data(raw)
            if parsed is None:
                continue

            self._udp_data = self._build_output_packet(parsed["rigid_bodies"])
            self._udp_data_time = now

        print("upd thread stopped")

    def stop_thread(self):
        self._udpStop = True
        time.sleep(self.sample_time)
        self._udpThread_on = False

    def get_data(self):
        return self._udp_data

    def get_data_sync(self):
        self.udp_flag += 1
        raw, _ = self._sock.recvfrom(self.len_data)
        now = time.time()

        parsed = self._parse_frame_of_mocap_data(raw)
        if parsed is None:
            return self._udp_data

        self._udp_data = self._build_output_packet(parsed["rigid_bodies"])
        self._udp_data_time = now
        return self._udp_data
    def get_data_with_time(self):
        return self._udp_data, self._udp_data_time

    # optional helpers
    def get_body_ids(self):
        return [rb["id"] for rb in self._latest_rigid_bodies]

    def get_index_of_id(self, rigid_body_id):
        return self._id_to_index.get(rigid_body_id, None)
    

class DataProcessor(object):
    def __init__(self, num_bodies, sample_rate):
        self.num_bodies = num_bodies
        self.sample_rate = sample_rate
        body_name = [i for i in range(1, self.num_bodies + 1)]
        self.keys = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        self.data_list = {name: {key: 0.0 for key in self.keys} for name in body_name}

        self.save_list_name = []
        self.save_list_data = []

        for body in body_name:
            for key in self.keys:
                self.save_list_name.append('b' + str(body) + '_' + key)
                self.save_list_data.append(0.0)

    def process_data(self, udp_data):
        for i in range(1, self.num_bodies + 1):
            x, y, z, qx, qy, qz, qw = struct.unpack("hhhhhhh", udp_data[(i*14 - 14):i*14])
            self.data_list[i]['x'] = x * 0.0005
            self.data_list[i]['y'] = y * 0.0005
            self.data_list[i]['z'] = z * 0.0005
            self.data_list[i]['qx'] = qx * 0.001
            self.data_list[i]['qy'] = qy * 0.001
            self.data_list[i]['qz'] = qz * 0.001
            self.data_list[i]['qw'] = qw * 0.001

        i = 0
        for body in self.data_list:
            for key in self.keys:
                self.save_list_data[i] = self.data_list[body][key]
                i = i + 1
        return self.data_list, self.save_list_data