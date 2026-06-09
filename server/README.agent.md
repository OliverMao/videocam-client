# TeleAI 展厅监控智能体架构

## 系统总览

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI (main.py)                       │
│                       port: 28001                            │
├──────────────────┬──────────────────┬────────────────────────┤
│   /api/qa/*      │ /api/vectordb/*  │   /api/show-client     │
│   QA 智能体      │  矢量数据库       │   推理结果（stub）      │
├──────────────────┴──────────────────┴────────────────────────┤
│                                                              │
│  ┌─── agent/ ────────────────┐  ┌─── vectordb/ ──────────┐  │
│  │  prompt.py   SYSTEM_PROMPT│  │  embeddings.py  288-d  │  │
│  │  tools.py    5 tools      │  │  db.py          numpy  │  │
│  │  memory.py   持久记忆     │  │  capture.py     1fps   │  │
│  │  agent.py    QAAgent      │  └─────────────────────────┘  │
│  └───────────────────────────┘                               │
│                                                              │
│  ┌─── storage ──────────────────────────────────────────┐    │
│  │  .memory/   Markdown + YAML frontmatter + MEMORY.md  │    │
│  │  .vectordb/ vectors.npy + meta.json + thumbs/        │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
         │                        │
         ▼                        ▼
  ┌──────────────┐       ┌──────────────┐
  │  vLLM API    │       │  RTSP Camera │
  │  :30630      │       │  192.168.x.x │
  │  Qwen3.6-35B │       └──────────────┘
  └──────────────┘
```

---

## Agent 模块 (`server/agent/`)

### 文件结构

| 文件 | 职责 |
|------|------|
| `prompt.py` | `SYSTEM_PROMPT` 常量 — 角色定义、范围校验、工作流程、回答格式模板 |
| `tools.py` | `TOOLS` 列表（5 个 OpenAI function-calling 工具）、`TOOL_DESCRIPTIONS`、`MAX_TOOL_ITERATIONS=10` |
| `memory.py` | `MemoryManager` — 跨会话持久记忆系统 |
| `agent.py` | `QAAgent` — 主智能体，编排 prompt + tools + memory |

### 工具链

```
todo_write  →  query_database  →  sort_by_relevance  →  summarize  →  最终回答
   │                │                   │                   │
 列出计划       按时间+关键词       按匹配度排序         总结生成
 (必须第一步)     查询历史数据
```

每次回答必须完整走完全链，即使查询结果为空。

| 工具 | 参数 | 说明 |
|------|------|------|
| `todo_write` | `todos: [{content, status}]` | 任务规划与状态管理 |
| `query_database` | `start_time, end_time, keywords?` | 历史监控数据查询 |
| `sort_by_relevance` | `data, query` | 结果相关性排序 |
| `summarize` | `content, style?` | 结果总结（简洁/详细/报告） |
| `memory_write` | `content, type, tags?` | 保存长期记忆 |

### 回答格式模板

SYSTEM_PROMPT 定义了 4 种输出格式，由 LLM 根据问题类型自动选择：

| 格式 | 适用场景 | 核心结构 |
|------|---------|---------|
| 事件查询 | 异常事件、行为记录 | 表格（时间/位置/类型/详情）+ 统计 + 建议 |
| 状态分析 | 设备状态、安全态势 | 正常/异常列表 + 风险提示 + 结论 |
| 趋势对比 | 时间段对比、数据变化 | 对比表格 + 趋势判断 |
| 简洁回答 | 简单问答、无数据 | 2-3 句直接回答 |

---

## Memory 系统 (`agent/memory.py`)

### 架构

```
                    ┌─── MEMORY.md（索引，常驻 SYSTEM prompt）
.memory/  ────┤
                    └─── *.md（记忆文件，YAML frontmatter + 正文）
```

### 记忆类型

| 类型 | 用途 | 示例 |
|------|------|------|
| `user` | 用户偏好 | "用户偏好简洁风格回答" |
| `feedback` | 用户反馈/纠正 | "吸烟检测误报率高需说明" |
| `project` | 项目/设备信息 | "3楼摄像头已更换" |
| `reference` | 参考知识 | "展厅开放时间 9:00-18:00" |

### 数据流

```
用户提问
    │
    ▼
┌─────────────────┐
│  load_relevant   │  从索引中选相关记忆（LLM side-query，关键词降级）
│  (最多 5 条)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ build_system_    │  SYSTEM_PROMPT + MEMORY_INSTRUCTION
│ prompt           │  + 记忆索引表 + 相关记忆正文
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  _call_ai        │  带记忆注入的 system prompt 发给 LLM
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ extract_from_    │  后台 fire-and-forget，LLM 隐式提取
│ conversation     │  （最多 3 条/轮）
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  consolidate     │  文件数 ≥ 10 时触发，按类型分组合并去重
└─────────────────┘
```

### 关键参数

| 参数 | 值 | 说明 |
|------|----|------|
| `CONSOLIDATION_THRESHOLD` | 10 | 触发合并整理的文件数 |
| `MAX_MEMORIES_PER_QUERY` | 5 | 每次查询最多注入的记忆条数 |
| `MAX_MEMORY_CHARS` | 2000 | 注入记忆的总字符预算 |
| `MAX_IMPLICIT_EXTRACTS` | 3 | 每轮对话最多隐式提取的记忆条数 |

### 容错设计

所有记忆操作均 try/except 包裹，失败不影响主 QA 流程：
- LLM 选择失败 → 关键词匹配降级
- 关键词无结果 → 不注入记忆，正常使用
- 提取/合并失败 → 静默日志，无用户影响

---

## VectorDB 模块 (`server/vectordb/`)

### 文件结构

| 文件 | 职责 |
|------|------|
| `embeddings.py` | `compute_embedding(frame) → np.ndarray[288]` — OpenCV 特征提取 |
| `db.py` | `VectorDB` — numpy 存储 + 余弦相似度搜索 |
| `capture.py` | `FrameCaptureService` — RTSP 1fps 帧采集后台服务 |

### Embedding 特征（288 维）

| 特征 | 维度 | 方法 |
|------|------|------|
| HSV 颜色直方图 | 192 (64×3) | H/S/V 通道各 64 bins |
| 灰度直方图 | 64 | 单通道 64 bins |
| 空间边缘方向 | 32 (8×4) | Canny + Sobel，2×2 网格各 8 方向 |
| **合计** | **288** | L2 归一化 |

输入帧先缩放至 320px 宽，纯 OpenCV + numpy 实现，零额外依赖。

### 存储格式

```
.vectordb/
├── vectors.npy          # float32, shape: (N, 288)
├── meta.json            # [{id, timestamp, thumb}, ...]
└── thumbs/              # JPEG 缩略图 (160px 宽, quality 60)
    └── 2026-06-09_10-00-00.jpg
```

### 搜索流程

```
上传图片
    │
    ▼
compute_embedding()  →  288-d vector
    │
    ▼
VectorDB.search()    →  cosine similarity (点积，向量已 L2 归一化)
    │
    ▼
返回 top-K 结果      →  [{id, timestamp, thumb, score}, ...]
```

### 采集流程

```
FrameCaptureService (daemon thread)
    │
    ├── cv2.VideoCapture(RTSP_URL)
    │   └── buffer_size = 1 (低延迟)
    │
    ├── 每秒 1 帧
    │   ├── compute_embedding(frame)
    │   ├── 保存缩略图 (thumbs/)
    │   └── db.add(vector, timestamp, thumb_path)
    │
    ├── 每 30 秒持久化 db.save()
    │
    └── 断流自动重连 (3 秒间隔)
```

---

## API 接口

### QA 智能体

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/qa/ask` | 非流式问答，返回 `{answer, steps}` |
| `POST` | `/api/qa/ask-stream` | SSE 流式问答，事件流 |

请求体: `{"question": "...", "history": [...]}`

SSE 事件类型: `step_add`, `step_update`, `step_status`, `answer_chunk`, `done`, `error`

### 矢量数据库

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/vectordb/stats` | 数据库统计（帧数、维度、时间范围） |
| `POST` | `/api/vectordb/search` | 上传图片搜索相似帧（multipart, top_k 参数） |
| `GET` | `/api/vectordb/frames?start=...&end=...` | 时间范围查询 |
| `GET` | `/api/vectordb/thumb/{path}` | 获取缩略图 JPEG |

### 推理结果

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/show-client` | 最新推理结果（当前 stub） |

---

## 生命周期

```
FastAPI lifespan
    │
    ├── startup
    │   └── FrameCaptureService.start()   # 启动 RTSP 1fps 采集
    │
    └── shutdown
        ├── FrameCaptureService.stop()    # 停止采集 + db.save()
        └── inference_client.stop()       # 停止推理客户端
```

`QAAgent` 和 `VectorDB` 在模块级实例化，无需 lifespan 管理。`MemoryManager` 在首次查询时 lazy init（`ensure_ready()`）。

---

## 配置

| 参数 | 默认值 | 位置 |
|------|--------|------|
| LLM API | `http://116.238.240.2:30630` | `main.py` |
| 模型 | `Qwen3.6-35B-A3B` | `main.py` |
| API Key | `vllm` | `main.py` |
| RTSP URL | `rtsp://admin:qazwsx168@192.168.158.195:554/...` | `main.py` |
| 服务端口 | `28001` | `main.py` |
| 工具迭代上限 | `10` | `agent/tools.py` |
| 记忆合并阈值 | `10 文件` | `agent/memory.py` |
| 帧采集频率 | `1 fps` | `vectordb/capture.py` |
| 缩略图宽度 | `160px` | `vectordb/capture.py` |
| 向量维度 | `288` | `vectordb/embeddings.py` |

---

## 目录结构

```
server/
├── agent/                    # QA 智能体
│   ├── __init__.py           # exports QAAgent
│   ├── prompt.py             # SYSTEM_PROMPT (角色/流程/模板)
│   ├── tools.py              # 5 工具定义 + 常量
│   ├── memory.py             # MemoryManager (持久记忆)
│   └── agent.py              # QAAgent (主编排)
│
├── vectordb/                 # 矢量数据库
│   ├── __init__.py           # exports
│   ├── embeddings.py         # 帧 → 288-d 向量
│   ├── db.py                 # VectorDB (numpy + cosine)
│   └── capture.py            # 1fps RTSP 采集
│
├── main.py                   # FastAPI 入口 + 路由 + 生命周期
├── .memory/                  # 记忆文件 (runtime)
│   ├── MEMORY.md             # 索引
│   └── *.md                  # 记忆条目
├── .vectordb/                # 矢量数据 (runtime)
│   ├── vectors.npy
│   ├── meta.json
│   └── thumbs/
│
├── requirements.txt
└── Dockerfile
```
