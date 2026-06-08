# 视频四路同步处理流水线

> 1 路原始视频 → 4 路 MP4 输出,帧序列 **完全同步**

## 概览

| 输出文件 | 处理 | 编码 |
|---------|------|------|
| `<name>_1_original.mp4` | 原样透传 | `-c:v copy`(完全无损) |
| `<name>_2_blur.mp4` | `boxblur` 高斯模糊 | libx264 crf=18 |
| `<name>_3_compressed.mp4` | 降采样 32× + 像素化回放 + 极限码率 | libx264 crf=51 / 50 kbps |
| `<name>_4_combined.mp4` | `vstack` 上下拼接(原图 + 模糊) | libx264 crf=18 |

四路音频统一为 aac 128k。

## 为什么四路"完全同步"?

单 `ffmpeg` 进程 + 单 `filter_complex`,所有输出共享:
- 同一个 demuxer(读源)
- 同一个 filter graph(逐帧处理)
- 同一个时间轴(PTS 由解封装器统一打)

不会出现 Python 多进程方案中常见的帧调度漂移 / buffer 阻塞 / 启动时差。

实测:输入 5s / 30fps / 150 帧的视频,4 路输出全部 `nb_frames=150`,`duration=5.000000`。

## 关键 filter 图

```
[0:v] ── split=3 ──► [v0][v1][v2]
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
   [v0] (原图)     [v1] ─boxblur─► [blur_raw] ─ split=2 ─► [blur_a][blur_b]
                                                       │
                                                       └► 配合 [v0] 走 vstack ─► [combined]
       │
       └► [v2] ─ scale↓ ─ scale↑ ─► [compressed]
```

> 提示:filter 图里每个输出 pad 只能被消费一次。`[blur]` 同时被 `-map` 和 `vstack` 用,必须先 `split=2` 拆成两份。这是踩过的一个坑,代码里已经处理了。

## 使用方法

### 1. 准备环境
需要 `ffmpeg` (>= 4.x) 和 `python3` (>= 3.8)。
```bash
ffmpeg -version   # 验证
python3 --version
```

### 2. 跑默认参数
```bash
python3 process_video.py /path/to/input.mp4
```
默认输出到 `./output/`,模糊半径 15,压缩倍数 32。

### 3. 自定义参数
```bash
python3 process_video.py input.mp4 \
  -o ./out \
  --blur 20 \
  --compress 24
```

| 参数 | 含义 | 默认 |
|------|------|------|
| `input` | 输入视频路径(位置参数) | 必填 |
| `-o / --output` | 输出目录 | `./output` |
| `--blur` | 高斯模糊半径(像素) | `15` |
| `--compress` | 降采样倍数(值越大像素块越大) | `32` |
| `--show-cmd` | 打印完整 ffmpeg 命令(debug 用) | `false` |

## 输出示例

```text
$ python3 process_video.py samples/test.mp4
🚀 启动 ffmpeg ...

✅ 处理完成,输出文件:
   • test_1_original.mp4                           0.54 MB
   • test_2_blur.mp4                               0.25 MB
   • test_3_compressed.mp4                         0.07 MB
   • test_4_combined.mp4                           0.88 MB
```

## 扩展点

- **想换模糊算法**:`boxblur` → `gblur=sigma=20`(更柔)或 `smartblur`(细节保留)
- **想换"极限压缩"定义**:把 `scale + neighbor` 那段去掉,只保留 crf=51 即可(分辨率不变,仅码率极限)
- **想要 RTSP/RTMP 实时流输出**:把最后的输出文件路径替换成 `rtsp://...` / `rtmp://...`,其余不动(单进程同步依然成立)
- **想要更多处理支路**:`split=N` 调大,各支路分别走不同 filter,最后 `xstack` / `vstack` / `overlay` 组合

## 性能

- 单进程零拷贝,FFmpeg 内部以帧为单位在 filter 图里扇出。
- 性能瓶颈通常在编码(尤其是 crf=18 的两条)。可调 `-preset ultrafast` 提速,代价是文件稍大。
- 4K / 长视频毫无压力,前提是机器内存够(逐帧处理不会把整段加载到内存)。

## 目录结构

```
video-pipeline/
├── README.md              ← 本文件
├── process_video.py       ← 主脚本(Python 包装 ffmpeg)
├── samples/
│   └── test.mp4           ← 测试视频(5s / 640x360 / 30fps)
└── output/                ← 默认输出目录
    ├── test_1_original.mp4
    ├── test_2_blur.mp4
    ├── test_3_compressed.mp4
    ├── test_4_combined.mp4
    └── preview_4streams.png   ← 四路同帧预览图
```
