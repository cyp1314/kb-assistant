# LangChain (Python) 学习路线图：从入门到干活

> 适用对象：有后端开发经验、想用 Python 系统掌握 LangChain 并具备独立搭建 AI 应用能力的学习者。
> 更新时间：2026-09（对应 LangChain 1.x 时代）

---

## 第 0 阶段：先建立正确的心智模型（第 1~2 天）

LangChain 不是一个"大模型"，而是一个**编排框架（Orchestration Framework）**——它负责把"提示词、模型调用、工具、记忆、检索、流程控制"这些零件标准化并粘合成应用。类比后端开发：LangChain 之于 LLM 应用，有点像 Spring 之于 Web 服务，或者更准确地说，像 Airflow/Celery 之于任务编排。

2026 年的 LangChain 生态分成几个层次，先认清名字，学习时不会被绕晕：

| 组件 | 作用 | 后端类比 |
|---|---|---|
| `langchain-core` | 基础抽象：消息、模型接口、工具协议、Runnable 接口 | 标准库 / SPI 接口层 |
| `langchain` | 主包：`create_agent`、提示词模板、常用集成入口 | Spring Boot  starters |
| `langchain-openai` / `langchain-deepseek` / ... | 各家模型的适配器 | JDBC 驱动 |
| **LangGraph** | 有状态工作流/多 Agent 编排引擎（图、状态机、检查点持久化） | 工作流引擎（如 Camunda） |
| **LangSmith** | 调试、追踪（tracing）、评估（evals）、监控 | APM / SkyWalking |
| `deepagents` | 官方高层 Agent 框架（基于 LangGraph） | 脚手架 |

**关键认知（很重要，避免走弯路）：**
1. 网上大量 2024~2025 年初的教程使用 `LLMChain`、`ConversationChain`、`initialize_agent`、`AgentExecutor` 等**已废弃 API**，看到这些直接跳过。1.x 的官方推荐入口是 `create_agent` + LangGraph。
2. 官方建议：**尽量用 `provider-agnostic`（跨模型通用）接口**，不要绑死某家模型的私有特性，这样换模型只改一行。
3. LCEL（`prompt | model | parser` 管道语法）依然是基础构件，但业务重心已经转向 Agent。

---

## 第 1 阶段：核心基础（第 1 周末）

目标：能调用模型、管理多轮对话、拿到结构化输出。这是所有上层应用的地基。

### 学习内容

1. **环境搭建**
   ```bash
   python -m venv .venv && .venv\Scripts\activate   # Windows
   pip install langchain langchain-openai langgraph langchain-community
   pip install python-dotenv
   ```
   - API Key 放 `.env`，用 `python-dotenv` 加载——和数据库密码同样对待。
   - 国内网络环境下可先用 DashScope（通义）、DeepSeek、智谱等国内模型服务商的兼容接口练手；注册各家平台通常会送免费额度。

2. **Models（聊天模型）**
   - `init_chat_model("openai:gpt-4o-mini")` 一行创建模型对象
   - `invoke` / `stream` / `batch` 三种调用方式（对应同步、SSE 流式、批量）
   - 温度、最大 token 等参数的实际含义

3. **Messages（消息体系）**
   - `SystemMessage` / `HumanMessage` / `AIMessage` / `ToolMessage` 四种角色
   - 消息列表就是"对话历史"——理解这一点，多轮对话就没有魔法

4. **Structured Output（结构化输出）**
   - `with_structured_output(PydanticModel)` 让模型返回 JSON 对象
   - 这是后端同学最该练熟的一项：把 LLM 当"返回 DTO 的远程函数"用

5. **Prompts（提示词模板）**
   - `ChatPromptTemplate.from_messages`，动态注入变量
   - 系统提示词的角色（类比：接口契约）

6. **LCEL 管道**
   ```python
   chain = prompt | model | parser
   chain.invoke({"topic": "..."})
   ```
   理解 Runnable 协议：凡是有 `invoke/stream/batch` 的东西都能用 `|` 串起来。

### 练手作业
- 写一个命令行"文档摘要器"：读文件 → 模板拼装 → 流式输出摘要
- 写一个"信息抽取器"：从一段地址文本里抽出姓名/电话/省市区，用 Pydantic 校验

---

## 第 2 阶段：工具与 Agent（第 2 周）★ 分水岭

目标：理解 ReAct 循环，能做出"会自己调用工具回答问题"的 Agent。

### 学习内容

1. **Tools**
   - `@tool` 装饰器把普通 Python 函数变成工具（docstring 就是给模型看的"接口文档"）
   - `StructuredTool` / Pydantic args schema
   - 工具错误处理：返回 `ToolMessage(status="error")` 让模型自我修正

2. **ReAct 原理（必须亲手理解一次）**
   - 循环本质：`模型思考 → 请求调用工具 → 执行工具 → 结果回填 → 再思考…` 直到产出最终回答
   - 用 LangSmith 追踪一条消息，看清每一轮 step——这一步做了，Agent 对你不再有黑盒

3. **create_agent（1.x 的标准 Agent 入口）**
   ```python
   from langchain.agents import create_agent
   agent = create_agent(model="openai:gpt-4o-mini", tools=[...])
   agent.invoke({"messages": [...]})
   ```
   - 系统提示词、工具列表、返回值结构
   - MCP 工具接入：`langchain-mcp-adapters`（把 MCP 服务器工具直接喂给 Agent，是当前主流做法）

4. **Memory / 对话状态**
   - Agent 自带 message history；跨会话持久化靠 LangGraph 的 checkpointer（SQLite/Postgres）
   - 理解"短期记忆（本线程消息）vs 长期记忆（跨线程 store）"

### 练手作业
- "天气+日历助手"：给它查天气、查节假日两个工具，问"我下周末适合去杭州玩吗"
- 给 Agent 加上会话记忆，重启进程后历史不丢（checkpointer 存 SQLite）

---

## 第 3 阶段：RAG 检索增强（第 3 周）★ 就业最常用

目标：能搭"知识库问答"——这是企业里 LangChain 岗位需求量最大的场景。

### 学习内容

1. **加载与切分**：`DocumentLoader`（PDF/Markdown/网页）、`RecursiveCharacterTextSplitter` 及 chunk 大小/重叠的权衡
2. **Embeddings**：文本向量化原理（一句话版：把语义变成几何距离）；选一个 embedding 模型（OpenAI、BGE、DashScope 等）
3. **向量库**：先学内存版 Chroma 或 FAISS；了解 In-Memory Index（教程够用）与 Persistent Index（生产）的区别；进阶看 Milvus / Pgvector
4. **检索模式**
   - 朴素 RAG：`retriever | prompt | model`
   - 索引构建：`load → split → embed → index`
   - 父子文档（Parent-Document）、上下文压缩（Contextual Compression）等提高召回质量的手段
5. **LangGraph 版 Agentic RAG**：让 Agent 自己决定"要不要检索、检索几次、结果够不够"——比固定管道效果好，是当前面试热点
6. **评估意识**：怎么知道 RAG 答得对不对？（引用 LangSmith evals，答案忠实度/检索相关性）

### 练手作业
- 把你手头的一份文档（技术手册/政策文件）做成问答机器人，要求回答必须带来源引用
- 升级版：用 LangGraph 改造成"先判断问题类型，再选择向量检索或查数据库"的混合路由

---

## 第 4 阶段：LangGraph 与工作流编排（第 4~5 周）

目标：从"调 API 的人"升级为"能设计多步骤、可恢复、可人审流程的人"。

### 学习内容

1. **StateGraph 基础**：Node / Edge / State（TypedDict 状态在节点间流转）、条件路由
2. **持久化与中断**：checkpointer（断点续跑）、`interrupt`（Human-in-the-loop，人审批后继续）——企业场景刚需
3. **多 Agent 模式**：Supervisor（主管派发）、Swarm（互相移交）、子图（subgraph）
4. **流式输出**：`stream_mode="values"/"updates"`，把中间过程推给前端
5. **time travel**：回滚到某个历史状态重新执行（调试利器）

### 练手作业
- "内容生产流水线"：选题 Agent → 写作 Agent → 审校 Agent，审校不通过则回环重写
- 给流水线加人工审核节点：审校通过后 interrupt，人点"同意"才发布

---

## 第 5 阶段：工程化与"能干活"（第 6~7 周）

目标：把 demo 变成可交付的服务。这一阶段大量利用你已有的后端经验。

### 学习内容

1. **服务化**
   - LangServe（`add_routes` 一行把 chain/agent 变 REST API）或 FastAPI 自己包装
   - SSE 流式响应接入前端
2. **可观测性**：LangSmith 项目配置、tracing、成本与延迟监控、生产告警
3. **评估与回归**：用 evaluator（LLM-as-judge + 规则断言）建测试集，防止改提示词改出回归
4. **健壮性**：重试与超时（`with_retry`/`with_fallbacks`，模型宕机自动切备用模型）、限流、幂等
5. **安全**：提示词注入的防御思路（工具权限最小化、敏感操作必须人审）、API Key 管理、输出过滤
6. **部署**：容器化；了解 LangGraph Platform / 自托管的取舍

### 结业项目（简历级）
选一个做完整：**企业知识库智能客服**——RAG + 工单查询工具 + 转人工（interrupt）+ LangSmith 评估面板 + FastAPI 服务 + 前端聊天页。做完这一个，入门到干活这条线就闭环了。

---

## 推荐学习资源

**第一优先（官方，英文但有机器翻译）：**
- docs.langchain.com 官方文档，尤其 "Introduction"、"Get started with agents"、RAG 与 LangGraph 教程章节——2026 年官方文档质量已大幅提升，且与 1.x API 同步
- LangChain Academy（academy.langchain.com）：免费视频课，短小，有 LangGraph 专课
- GitHub `langchain-ai/langchain` 与 `langgraph` 仓库的 examples 目录

**中文资料：**
- runoob 的 LangChain 中文教程（适合快速查语法，注意核对是否为 1.x 写法）
- 官方文档的中文镜像/社区翻译，CSDN 与掘金上的 2025 年后新文（警惕 2024 旧 API 文章）

**练习数据源：**
- HuggingFace 上任意公开 PDF 语料、Kaggle 客服对话数据集

## 常见坑速查

| 坑 | 说明 |
|---|---|
| 照抄旧教程 | `LLMChain`/`initialize_agent`/`AgentExecutor` 已废弃，认准 `create_agent` + LangGraph |
| 一上来啃全部源码 | 不需要。把它当框架用，遇到行为异常再用 LangSmith trace 看真相 |
| RAG 效果差就换模型 | 80% 的问题是切分和检索质量，先调 chunk、再考虑模型 |
| Agent 循环烧钱 | 设置 `recursion_limit`、在开发期用便宜模型 + tracing 看每一步 token |
| 忽略流式 | 生产应用里非流式的 Agent 体验极差，第 1 阶段就练 `stream` |

## 时间预算参考

| 投入 | 结果 |
|---|---|
| 每天 1~2h × 7 周 | 完成全部阶段 + 结业项目 |
| 每天 4h+（冲刺） | 3 周到达"能干活" |
| 只求会用 RAG | 第 0/1/3 阶段，约 2 周 |
