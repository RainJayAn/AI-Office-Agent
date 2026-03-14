# AI Office Agent

一个按“先能跑，再完善”的思路实现的 AI 办公助手示例项目。当前版本已经包含：

- FastAPI 服务入口
- 支持多轮对话的聊天链路
- 模型原生 Function Calling 工具调用
- 本地 `md/txt` 文档 RAG
- OpenAI-compatible 模型接入，默认连接阿里云 DashScope
- DuckDuckGo 联网搜索工具

## 目录结构

```text
ai-office-agent/
├─ app/
│  ├─ main.py
│  ├─ api/
│  │  ├─ chat.py
│  │  ├─ rag.py
│  │  └─ tools.py
│  ├─ agent/
│  │  ├─ graph.py
│  │  ├─ nodes.py
│  │  └─ state.py
│  ├─ core/
│  │  ├─ config.py
│  │  ├─ exceptions.py
│  │  └─ path.py
│  ├─ llm/
│  │  ├─ factory.py
│  │  ├─ router.py
│  │  └─ providers/
│  │     └─ openai_compatible.py
│  ├─ rag/
│  │  ├─ ingest.py
│  │  ├─ pipeline.py
│  │  └─ retriever.py
│  ├─ services/
│  │  ├─ chat_service.py
│  │  ├─ rag_service.py
│  │  └─ tool_service.py
│  └─ tools/
│     ├─ registry.py
│     └─ builtins/
│        ├─ draft_email.py
│        ├─ retrieve_docs.py
│        └─ web_search.py
├─ tests/
├─ requirements.txt
├─ .env.example
└─ README.md
```

## 安装方法

### 1. 创建虚拟环境

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

如果只是本地启动 API，不一定需要马上配置模型。  
如果要走真实大模型调用，可以先复制配置模板：

```bash
cp .env.example .env
```

然后设置 DashScope API Key：

```powershell
$env:DASHSCOPE_API_KEY="your_api_key"
```

默认模型相关配置：

- Provider: `dashscope`
- 默认模型路由：`qwen-turbo` / `qwen3.5-flash` / `qwen3.5-plus`
- Base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- Embedding Model: `all-MiniLM-L6-v2`

## 启动命令

```bash
uvicorn app.main:app --reload
```

启动后可访问：

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/redoc`

## 接口示例

### 1. 健康检查

```bash
curl http://127.0.0.1:8000/health
```

### 2. 聊天接口

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"please draft an email to my manager about project delay\", \"use_rag\": false}"
```

### 3. 查看工具列表

```bash
curl http://127.0.0.1:8000/tools
```

### 4. 调用工具

```bash
curl -X POST http://127.0.0.1:8000/tools/run \
  -H "Content-Type: application/json" \
  -d "{\"tool_name\": \"draft_email\", \"args\": {\"recipient\": \"team@example.com\", \"subject\": \"Project Update\", \"purpose\": \"Share the latest project status\"}}"
```

### 5. 导入本地文档

目录示例：

```bash
curl -X POST http://127.0.0.1:8000/rag/ingest \
  -H "Content-Type: application/json" \
  -d "{\"directory\": \"./docs\"}"
```

单文件示例：

```bash
curl -X POST http://127.0.0.1:8000/rag/ingest \
  -H "Content-Type: application/json" \
  -d "{\"file_path\": \"./docs/office.md\"}"
```

### 6. 文档检索问答

```bash
curl -X POST http://127.0.0.1:8000/rag/query \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"Project Alpha deadline\", \"top_k\": 3}"
```

## 运行测试

```bash
pytest
```

## 使用说明

- 项目内部默认路径统一按项目根目录解析，不依赖当前启动目录
- `app/core/path.py` 已拆出应用目录、用户数据目录、上传目录、日志目录和向量库目录
- 默认用户数据目录为项目根目录下的 `.app_data`
- 默认上传目录为 `.app_data/uploads`
- 默认日志目录为 `.app_data/logs`
- 默认向量库目录为 `.app_data/.chroma`
- 支持导入本地 `md` 和 `txt` 文件
- 当前 RAG 使用递归切块、Embedding、Chroma 向量检索和基础 rerank
- 如果没有配置 `DASHSCOPE_API_KEY`，聊天链路无法调用真实模型
- 多模型路由会在 `qwen-turbo`、`qwen3.5-flash`、`qwen3.5-plus` 之间按成本、延迟和推理能力进行选择
