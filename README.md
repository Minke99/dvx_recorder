# DVXplorer 实时事件预览

复刻 iniVation DV viewer 的实时画面:**黑底,绿色 = 变亮(ON)事件,红色 = 变暗(OFF)事件**。
适用于 DVXplorer / DVXplorer Micro 等事件相机。

## 环境依赖

```bash
# 事件相机预览 / 录制
pip install dv-processing opencv-python numpy h5py
# 额外:mocap UDP 接收 + tools 分析脚本
pip install tqdm matplotlib
```

## 前提

USB 相机同一时刻只能被一个程序打开。**运行前请先关闭 DV-GUI**,否则会报
`LIBUSB_ERROR_BUSY`。(或者保持 DV 打开、在 DV 里加一个 `output > net tcp server`
模块,再用 `--net` 连过去,见下文。)

## 两个脚本

| 脚本 | 说明 |
|------|------|
| `dvx_live.py` | 基础版,纯实时预览(保底使用) |
| `dvx_live_denoise.py` | 去噪版,推荐日常看画面 |
| `dvx_record.py` | 实时预览 + 同时录制成 `.h5` 文件(只录事件) |
| `record_session.py` | **同步录制事件 + mocap,时间对齐**(预览 + events.h5 + mocap.h5) |

### 基础版

```bash
python dvx_live.py
```

### 去噪版(推荐)

```bash
python dvx_live_denoise.py --ba-ms 3.0
```

## 去噪参数 `--ba-ms`(重点)

控制去噪强度的就是这一个参数:**去噪相关时间窗口,单位毫秒。**

> ⚠️ **数字越小,滤得越狠**(越多孤立点被当成噪声丢掉);数字越大越温和、保留越多细节。

原理:一个事件如果在 `ba-ms` 这段时间内、周围邻近像素都没有别的事件,就判定为噪声丢弃。
真实运动的边缘有空间-时间相关性会被保留,背景随机散点会被清掉。

| `--ba-ms` | 效果 |
|-----------|------|
| 0.5 | 很狠,画面最干净,可能损失弱信号 |
| 1.0 | 默认,平衡 |
| **3.0** | **温和,保留细节多 —— 实测这个效果最好 ✅** |
| `--no-denoise` | 完全关闭去噪(等价于基础版) |

```bash
python dvx_live_denoise.py --ba-ms 3.0    # 当前首选
python dvx_live_denoise.py --ba-ms 0.5    # 想更干净时
python dvx_live_denoise.py --no-denoise   # 关掉去噪
```

### 录制版(存 h5)

一边看画面、一边把事件存进 HDF5 文件:

```bash
python dvx_record.py                       # 录到 recordings/dvx_<时间戳>.h5
python dvx_record.py --duration 10         # 录 10 秒后自动停止
python dvx_record.py --output recordings/take1.h5
```

- **默认存的是原始无损事件**,去噪只作用于预览画面;想直接存去噪后的干净数据用 `--save-denoised`。
- 录制时快捷键:`r` 暂停/继续、`d` 开关预览去噪、`q`/`ESC` 停止并保存。画面左上角有 `● REC / PAUSE`、已录时长和事件数。
- 去噪强度同样用 `--ba-ms`(默认 3.0)。

**H5 格式**(和 `record_dvx/record_raw.py` 一致,可直接拿那边的 `render_video.py` 回放):

```
events/x  int32      events/y  int32
events/t  int64 (us) events/p  int8  (1=ON 变亮, 0=OFF 变暗)
attrs: camera_name, resolution_width/height, time_unit="us",
       format="raw_dvx_events_v1", num_events, num_packets, duration_wall_s
```

读取示例:

```python
import h5py
with h5py.File("recordings/take1.h5") as f:
    x = f["events/x"][:]; y = f["events/y"][:]
    t = f["events/t"][:]; p = f["events/p"][:]   # t 单位 us, p∈{0,1}
```

## Mocap 录制(NatNet / OptiTrack,UDP)

从动捕(Motive / OptiTrack 的 NatNet 广播)接收刚体位姿,存成 HDF5。相关文件搬自 `record_dvx`:

| 文件 | 说明 |
|------|------|
| `LIS/UdpReceiver.py` | NatNet UDP 接收 + 解析刚体(`UdpRigidBodies` 类) |
| `record_mocap.py` | 把刚体流录成 `recordings/mocap.h5` |
| `tools/check_rigid_body_LIS.py` | 实时打印收到的刚体 ID / 位置,**先用它确认 UDP 收得到数据** |
| `tools/check_session.py` | 校验录好的 events.h5 + mocap.h5(结构 / 时间对齐) |
| `tools/depack_h5_data.py` | 把一次 session 解包:事件渲染成 MP4 + 画 mocap 轨迹 |

### 同步录制(事件 + mocap,时间对齐)—— 推荐

`record_session.py` 在**同一个进程**里同时收 DVX 事件和 mocap UDP,自动把两者放到同一条
「相机相对微秒」时间轴上,输出一个 session 目录:

```bash
# 0) 确认 Motive 正在 streaming(NatNet),先用它看一眼收不收得到
python tools/check_rigid_body_LIS.py

# 1) 同步录 —— 默认【不开实时画面】,只录盘(高事件率下也稳)
python record_session.py --duration 10
#   -> recordings/<时间戳>/events.h5 + mocap.h5 + sync.json
#   终端会每秒打印 rec时长 / 事件数 / Meps / mocap包数;Ctrl+C 或按 q 停止

# 1b) 想看实时画面再加 --preview(高事件率下可能略有延时,不影响存盘)
python record_session.py --preview

# 2) 校验对齐(结构 + 时间对齐 + 运动互相关)
python tools/check_session.py recordings/<时间戳>

# 3) 解包:事件渲染 MP4 + 画 mocap 轨迹
python tools/depack_h5_data.py recordings/<时间戳>
```

对齐原理:以第一批事件为锚点 `cam_first_us / wall_first_us`,每个 mocap 包按
`t_cam = cam_first_us + (到达墙钟 − wall_first_us)` 换算到相机时间轴。这样 `events/t` 与
`mocap/t` 共用同一时间轴(`time_unit="us_camera_relative"`)。已用合成数据跑通官方
`check_session.py`:overlap、startup offset < 500ms 全部 OK。

> 想验证对齐准不准:录的时候**在相机前挥动刚体**,`check_session.py` 会做事件率 vs 刚体速度的互相关,给出最佳 lag(应 < 30ms)。

**文件大小 / 压缩**:x/y 存 int16,并可选压缩 `--compress`:

| `--compress` | 速度 | 体积 | 适用 |
|---|---|---|---|
| `lzf`(默认) | 快 | 小(~3x) | **两边都推荐**,笔记本/Jetson 都能跑 |
| `gzip` | 慢(吃 CPU) | 最小(~5x) | 只在**快机器**(笔记本)上用 |
| `none` | 最快 | 最大 | 慢设备保命 / 离线再压 |

> ⚠️ **Jetson / 慢设备注意**:用 `gzip` 时单核 CPU 可能跟不上录制 →
> 相机缓冲溢出**丢事件** → 回放视频里出现**黑帧**。在 Jetson 上请用默认 `lzf`,
> 若仍丢就 `--compress none`(必要时再 `--no-denoise` 去掉去噪的 CPU 开销)。

事件数据量本质由**事件率**决定(运动越多/越近/纹理越密 → 事件越多 → 文件越大),
压缩只是把同样的事件存得更省。

### 只录 mocap(不带事件)

```bash
python record_mocap.py --output recordings/mocap.h5 --duration 10
```

`mocap.h5` 里 `mocap/` 组含 `t, frame, num_bodies` 及每个刚体的 `rb_x/y/z`、`rb_qx/qy/qz/qw`、
`rb_tracking_valid`、`rb_mean_error` 等;单独跑时时间戳是墙钟 Unix 微秒(`time_unit="us_since_epoch"`)。
要和事件对齐请直接用上面的 `record_session.py`。

## 其它常用参数(脚本通用)

| 参数 | 作用 |
|------|------|
| `--fps 60` | 画面刷新率(默认 30) |
| `--scale 2` | 窗口放大 2 倍(Micro 分辨率较低时好用) |
| `--swap` | 红绿对调(红 = ON,绿 = OFF) |
| `--serial <序列号>` | 多台相机时指定哪一台 |
| `--net HOST:PORT` | 改连 DV 的网络输出流,而不是直接占用相机 |

## 运行时快捷键

| 按键 | 作用 |
|------|------|
| `q` / `ESC` | 退出 |
| `d` | 临时开/关去噪(方便和原始画面对比,仅去噪版) |
| `[` | 去噪调狠一点(`ba-ms` 减小,仅去噪版) |
| `]` | 去噪调温和一点(`ba-ms` 增大,仅去噪版) |

## 备注

- 若实时窗口弹不出来、并一直刷 `QObject::moveToThread` 警告,那是 conda 自带 OpenCV
  的 Qt 后端和环境里另一套 Qt 插件冲突,和相机/去噪本身无关 —— 需要时再单独处理。
- `compare_denoise.png` 是一张去噪前/后的实拍对比图(左 RAW 满屏散点,右去噪后基本全黑)。
