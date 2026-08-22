# 智能学业咨询 Agent（基于 RAG）

一个基于 RAG（检索增强生成）的智能学业咨询助手。用通义千问 + ChromaDB 实现：
把 PDF / CSV 知识库向量化，用户提问时先检索相关内容，再让 LLM 结合资料回答，并标注引用来源。

## 功能

1. **RAG 问答**：PDF 按段落切分、CSV 按行加载，存入 ChromaDB，检索 Top-3 相关内容回答。
2. **引用溯源**：回答标注来源（PDF 显示页码 / CSV 显示文件名），前端单独展示来源列表。
3. **意图识别**：LLM 判断问题属于「学业咨询 / 数据分析 / 其他」，据此路由到 RAG 或普通对话。
4. **多轮记忆**：SQLite 保存最近 3 轮对话，追问时自动带上上下文。
5. **Gradio 前端**：输入框 + 对话展示 + 来源展示。
6. **pytest 测试**：健康检查、单轮问答、多轮记忆 3 个用例（测试不联网、不消耗 API）。

## 技术栈

Python · FastAPI · LangChain · ChromaDB · 通义千问 API（DashScope）· SQLite · Gradio · pytest

## 目录结构

```
agent-assistant/
├── app/
│   ├── main.py              # FastAPI 入口，注册路由
│   ├── api/chat.py          # /api/chat 接口 + Pydantic 模型
│   ├── core/config.py       # 阿里云 API Key 配置
│   ├── core/memory.py       # SQLite 多轮对话记忆
│   ├── services/rag_service.py  # RAG 核心：加载→切分→向量化→检索
│   ├── services/intent.py   # 意图识别
│   └── utils/document_loader.py # 多格式加载（PyPDFLoader + CSVLoader）
├── knowledge/               # 知识库（放你的 PDF / CSV）
│   ├── sample.pdf
│   └── sample.csv
├── frontend/app.py          # Gradio 交互界面
├── tests/test_chat.py       # pytest 测试
├── conftest.py              # pytest 路径配置
├── requirements.txt
└── README.md
```

## 环境要求

- Python 3.10 ～ 3.12（chromadb 对 Python 3.13 支持尚不稳定，建议用 3.10/3.11）

## 安装

```bash
# 1. 进入项目目录
cd agent-assistant

# 2. 创建并激活虚拟环境（推荐）
python -m venv .venv
# Windows（Git Bash / CMD / PowerShell）：
source .venv/Scripts/activate      # Git Bash
# 或 .venv\Scripts\activate.bat    # CMD
# 或 .venv\Scripts\Activate.ps1    # PowerShell

# 3. 安装依赖
pip install -r requirements.txt
```

> 首次安装 chromadb 等包可能较慢，耐心等待即可。

## 配置 API Key

`app/core/config.py` 里已内置了默认 Key，**开箱即用**。但建议改用环境变量，避免泄露：

```bash
# Windows Git Bash
export ALIBABA_API_KEY="sk-xxx你的key"

# 或 Windows PowerShell
$env:ALIBABA_API_KEY = "sk-xxx你的key"
```

> ⚠️ 安全提醒：你提供的 Key 已经出现在聊天记录里，请尽快去
> [阿里云百炼控制台](https://bailian.console.aliyun.com/) 轮换（重新生成）一个，并改用环境变量方式注入。

## 准备知识库

- `knowledge/sample.csv` 已提供示例课程数据（中文，可直接用）。
- `knowledge/sample.pdf` 是英文占位文件，**请替换为你自己的中文 PDF**（例如培养方案、选课指南）。
  - 想要中文 PDF 也简单：用 Word/WPS 写好内容后「另存为 PDF」，放到 `knowledge/` 目录即可。
- 任何新增的 `.pdf` / `.csv` 放到 `knowledge/` 后，**删除 `chroma_db/` 目录再重启后端**即可重建索引。

## 启动

先启动后端（FastAPI），再启动前端（Gradio）。**两个终端窗口分别运行**：

```bash
# 终端 1：启动后端
cd agent-assistant
uvicorn app.main:app --reload --port 8000
```

看到 `[App] 正在启动...`、`[RAG] 向量索引构建完成` 即成功。

```bash
# 终端 2：启动前端
cd agent-assistant
python frontend/app.py
```

打开浏览器访问 Gradio 给出的地址（默认 `http://127.0.0.1:7860`），即可提问。

后端接口文档：`http://127.0.0.1:8000/docs`

## 运行测试

```bash
cd agent-assistant
pytest -v
```

应看到 3 个用例全部通过：

```
test_health_check        PASSED
test_single_round_chat   PASSED
test_multi_round_memory  PASSED
```

## 常见问题

- **启动时报 `知识库索引构建失败`**：检查 `knowledge/` 是否有文件、网络是否可用、API Key 是否有效。
- **改了知识库但答案没变**：删除 `chroma_db/` 目录后重启后端，强制重建索引。
- **问答说“知识库尚未就绪”**：说明索引没建成功，看后端日志里的报错原因。
- **数据分析类问题**：当前通过检索 Top-3 条 CSV 行交给 LLM 分析，适合小数据集演示；数据量大时需另行聚合。
