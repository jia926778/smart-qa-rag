# Smart QA RAG - 智能问答系统

基于 RAG（检索增强生成）技术的智能问答系统，支持多种文档格式上传、知识库管理和自然语言问答。

## 功能特性

- **多格式文档支持**：PDF、TXT、DOCX、Markdown
- **知识库管理**：创建、查看、删除多个独立知识库
- **智能检索**：基于向量相似度的语义检索，可配置阈值
- **中文优化**：针对中文文本的分块策略与 Prompt 设计
- **来源引用**：回答附带参考来源与页码信息
- **对话历史**：支持多轮对话上下文
- **现代前端**：响应式聊天界面，支持 Markdown 渲染
- **Docker 部署**：一键容器化部署

## 技术栈

- **后端**：Python 3.11 + FastAPI
- **AI 框架**：LangChain
- **向量数据库**：ChromaDB
- **LLM**：OpenAI GPT（可配置 API Base 以使用兼容接口）
- **Embedding**：OpenAI / HuggingFace（本地）

## 快速开始

### 1. 克隆项目

```bash
git clone <repo-url>
cd smart-qa-rag
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入 OPENAI_API_KEY 等配置
```

### 3. 本地运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

访问 http://localhost:8000 打开聊天界面。

### 4. Docker 运行

```bash
docker compose up --build -d
```

## API 文档

启动后访问以下地址查看自动生成的 API 文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 主要接口

| 方法   | 路径                                | 说明         |
| ------ | ----------------------------------- | ------------ |
| POST   | `/api/v1/qa/ask`                    | 智能问答     |
| POST   | `/api/v1/documents/upload`          | 上传文档     |
| GET    | `/api/v1/documents/{collection}`    | 列出文档     |
| DELETE | `/api/v1/documents/{col}/{source}`  | 删除文档     |
| POST   | `/api/v1/collections/`              | 创建知识库   |
| GET    | `/api/v1/collections/`              | 列出知识库   |
| DELETE | `/api/v1/collections/{name}`        | 删除知识库   |
| GET    | `/api/v1/collections/{name}/stats`  | 知识库统计   |
| GET    | `/health`                           | 健康检查     |

## 运行测试

```bash
pip install -r requirements.txt
pytest -v
```

## 项目结构

```
smart-qa-rag/
├── app/                  # 后端应用
│   ├── main.py           # FastAPI 入口
│   ├── config.py         # 配置管理
│   ├── dependencies.py   # 依赖注入
│   ├── routers/          # API 路由
│   ├── models/           # 数据模型
│   ├── services/         # 业务逻辑
│   └── utils/            # 工具类
├── static/               # 前端文件
├── tests/                # 测试
├── data/                 # ChromaDB 持久化
└── uploads/              # 临时上传目录
```

## License

MIT
