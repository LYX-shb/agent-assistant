# 基于 RAG 的学业咨询助手

一个本地运行的 RAG Demo：从 PDF 页面和 CSV 行加载知识，使用 Chroma 检索相关内容，再调用 DashScope 兼容的通义千问模型生成回答。Gradio 页面会分开展示回答和后端返回的来源信息。

项目不包含用户认证，Gradio 会话 ID 只用于隔离演示期间的对话历史。数据统计、可靠引用校验和生产部署不在当前 Demo 能力范围内。

## 功能

- 加载 PDF 和 CSV：PDF 按页加载后切分为文本块；CSV 每行保持为独立文档。
- RAG 问答：Chroma 向量检索最多返回 5 条文本块，模型按上下文生成回答。
- 来源展示：后端返回来源文件名和 PDF 页码，Gradio 前端展示来源列表。来源列表说明检索依据，不代表程序已验证模型回答中的每个引用编号。
- 问题路由：LLM 将问题分为学业咨询、数据分析或其他；前两类走知识库检索，其他问题走普通模型对话。数据分析路由当前仍基于检索片段，不会对完整 CSV 执行统计聚合。
- 会话记忆：SQLite 按 session_id 保存对话，向模型提供最近 3 轮历史。直接调用 API 的客户端应在同一对话的多次请求中复用响应返回的 session_id。
- 自动化测试：`tests/test_chat.py` 使用替身模型和检索逻辑，不联网、不消耗 API 费用。

## 技术栈

Python、FastAPI、Pydantic、LangChain、Chroma、DashScope 兼容 API、SQLite、Gradio、pytest

## 目录结构

```text
app/
  main.py                     FastAPI 应用与启动索引
  api/chat.py                 聊天接口与请求/响应模型
  core/config.py              配置与路径
  core/errors.py              上游服务错误
  core/memory.py              SQLite 对话历史
  services/intent.py          问题意图识别
  services/rag_service.py     文档切分、向量索引、检索和生成
  utils/document_loader.py    PDF/CSV 加载
frontend/app.py               Gradio 页面
knowledge/                    放置 PDF 和 CSV
tests/test_chat.py            离线 API 用例
requirements.txt
```

## 环境

建议使用 Python 3.10–3.12。首次建立向量索引需要能够访问 DashScope API，并配置有效的 `ALIBABA_API_KEY`。

## 安装

在项目根目录创建虚拟环境并安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果 PowerShell 禁止激活脚本，也可以直接使用虚拟环境中的 Python：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 配置 API Key

项目不会在源码中内置 API Key。请在当前 PowerShell 窗口设置环境变量：

```powershell
$env:ALIBABA_API_KEY = "你的 DashScope API Key"
```

不要将真实 Key 提交到 Git、截图或公开日志。如果 Key 曾经泄露，请在对应服务控制台撤销并重新生成。

## 准备知识库

- `knowledge/sample.csv` 是示例课程表，CSV 每一行会单独建立向量记录。
- `knowledge/sample.pdf` 是示例文件。实际演示前请替换为有权使用的资料，并确认 PDF 可提取文本。
- 增删改知识文件后，需要重建向量索引。关闭后端，删除项目根目录下的 `chroma_db/`，再重新启动后端。
- SQLite 历史保存在 `chat_memory.db`。演示前若需清除对话记录，先关闭后端再删除该文件。

## 启动

先在第一个终端启动 FastAPI：

```powershell
uvicorn app.main:app --reload --port 8000
```

首次启动会尝试创建向量索引，过程需要 DashScope API。索引构建失败时进程仍会启动，但知识库问答会返回 HTTP 503；请检查后端日志、资料目录和 API 配置。`/api/health` 仅表示 API 进程可响应，不代表知识库已经就绪。

再在第二个终端启动 Gradio（API Key 只需配置在运行 FastAPI 的终端）：

```powershell
python frontend/app.py
```

打开 Gradio 输出的本地地址。FastAPI 接口文档位于 `http://127.0.0.1:8000/docs`。

## API 示例

```json
{
  "session_id": "client-generated-session-id",
  "message": "高等数学多少学分？"
}
```

首次请求也可以省略 `session_id`，服务会生成并在响应中返回。客户端若要保持多轮历史，之后的请求应复用该 ID。不同对话必须使用不同 ID。这个字段本身不是登录凭证或身份认证。

模型、向量化服务、知识库索引不可用时，`POST /api/chat` 返回 HTTP 503 和面向用户的错误说明；服务端日志只记录异常类型，不记录提供商的完整异常文本。

## 运行离线测试

```powershell
pytest -v
```

测试会替换 LLM 与向量检索调用，因此不验证真实 API Key、DashScope 额度、真实向量索引或在线模型质量。

## 常见问题

- **知识库问答返回 503**：检查后端启动日志、`knowledge/` 文件、API Key、网络和 DashScope 额度。若更新过文件，删除旧 `chroma_db/` 后重建。
- **模型请求超时或被限流**：页面会显示服务暂不可用；检查网络及供应商控制台状态与用量，再重试。
- **API 直接调用时多轮上下文丢失**：客户端需要在连续请求中复用响应里的 `session_id`。
- **回答显示来源，但引用编号不正确**：目前展示的是检索来源列表，尚未自动核验模型生成的引用编号与内容是否一致。
- **统计类问题结果不可靠**：当前数据分析意图仍将检索片段交给模型；需要准确统计时应使用 SQL 或 Python 对完整数据聚合。
