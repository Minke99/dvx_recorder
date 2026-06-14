# DVXplorer 实时事件预览

复刻 iniVation DV viewer 的实时画面:**黑底,绿色 = 变亮(ON)事件,红色 = 变暗(OFF)事件**。
适用于 DVXplorer / DVXplorer Micro 等事件相机。

## 环境依赖

```bash
pip install dv-processing opencv-python numpy h5py
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
| `dvx_record.py` | 实时预览 + 同时录制成 `.h5` 文件 |

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
