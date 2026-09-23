# LangChain 系统学习全记录（2026-09-23 起）

> 学员画像：有后端开发经验 / Python / 目标从入门到能干活。
> 配套文件：`LangChain学习路线图.md`（总路线）、`kb-assistant/`（阶段五结业工程）。
> 本文按授课顺序记录了每个作业的可运行代码与讲解要点，作为个人复习讲义使用。

## 目录

| 课程 | 内容 | 对应阶段 |
|---|---|---|
| 第 1 课 | 命令行文档摘要器（流式输出） | 第 1 阶段 |
| 第 2 课 | 信息抽取器（Pydantic 结构化输出） | 第 1 阶段 |
| 第 3 课 | 天气+日历助手（第一个 Agent） | 第 2 阶段 |
| 第 4 课 | Agent 持久会话（SQLite checkpointer） | 第 2 阶段 |
| 第 5 课 | 文档问答机器人（朴素 RAG + 引用） | 第 3 阶段 |
| 第 6 课 | 混合路由（LangGraph 第一课） | 第 3→4 阶段 |
| 第 7 课 | 人审闸门（interrupt + 持久挂起） | 第 4 阶段 |
| 第 8 课 | 企业智能客服完整工程 | 第 5 阶段 |

贯穿全程的基线认知：API 以 LangChain 1.x 为准（`create_agent` + LangGraph），
`LLMChain` / `initialize_agent` / `AgentExecutor` 等旧教程写法已废弃。

---

# 第 1 课 · 文档摘要器

知识点：模型创建、消息体系、ChatPromptTemplate、流式输出。

```python
"""读文件 → 模板拼装 → 流式输出摘要
依赖: pip install langchain langchain-openai python-dotenv"""
import argparse, sys
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

def read_document(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        sys.exit(f"文件为空: {path}")
    if len(text) > 30_000:          # 简单截断；生产应分段摘要再汇总(map-reduce)
        text = text[:30_000]
    return text

PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage("""你是一名专业的技术文档摘要员。
要求：用{language}输出；先一句话总括，再列3~5条核心要点；
总长度不超过{max_words}字，不编造原文没有的信息"""),
    HumanMessage("请摘要以下文档：\n\n<document>\n{document}\n</document>"),
])

def main():
    parser = argparse.ArgumentParser(description="文档摘要器")
    parser.add_argument("file", type=Path)
    parser.add_argument("--lang", default="中文")
    parser.add_argument("--max", dest="max_words", default=300)
    args = parser.parse_args()
    load_dotenv()

    model = init_chat_model("openai:gpt-4o-mini", temperature=0)  # 换provider只改这行
    messages = PROMPT.invoke({"language": args.lang,
                              "max_words": args.max_words,
                              "document": read_document(args.file)})
    for chunk in model.stream(messages):
        print(chunk.content, end="", flush=True)   # flush 是流式体验的关键
    print()

if __name__ == "__main__":
    main()
```

要点：占位符在 `invoke` 时填充产出标准消息列表；`model.stream()` 返回迭代器，
`end="" + flush=True` 才能实现打字机效果；大文档截断是偷懒做法，正确姿势是 map-reduce 分段摘要。

---

# 第 2 课 · 信息抽取器（Pydantic 校验）

知识点：`with_structured_output` —— 把 LLM 当"返回 DTO 的远程函数"。

```python
"""从地址文本抽取 姓名/电话/省市区
依赖: pip install langchain langchain-openai python-dotenv pydantic"""
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

load_dotenv()

class AddressInfo(BaseModel):
    """从文本中抽取出的收件人地址信息"""
    name: str = Field(description="收件人姓名")
    phone: str = Field(description="11位手机号，若原文没有则为空字符串")
    province: str = Field(description="省级行政区，如 广东省")
    city: str = Field(description="地级市，如 深圳市")
    district: str = Field(description="区县，如 南山区")
    detail: str = Field(default="", description="区以下的详细地址")

model = init_chat_model("openai:gpt-4o-mini", temperature=0)
extractor = model.with_structured_output(AddressInfo)  # schema→JSON Schema；返回前自动校验

SAMPLES = [
    "李伟 13812345678 广东省深圳市南山区科技园南路88号腾讯大厦B座12楼",
    "收件人：王芳，电话：159-8765-4321，寄到浙江杭州西湖区文三路 259 号 501室",
    "江苏省南京市鼓楼区汉口路22号 南京大学 苏文 025-83593333 邮编210093",
    "麻烦发到重庆市渝北区龙溪街道嘉州路31号附1号 陈杰收 18600001111",  # 直辖市陷阱
]

for text in SAMPLES:
    result: AddressInfo = extractor.invoke(f"从以下文本中抽取收件人地址信息，缺失的字段填空字符串：\n{text}")
    print(result.model_dump_json(indent=2))
```

要点：生成类任务调高 temperature，抽取/分类类任务归零；测试语料故意含脏数据
（电话带分隔符、座机、直辖市）——"用脏数据探测 schema 边界"就是 eval 意识的起点；
描述里必须定义缺失策略，否则模型幻觉编造电话。进阶：`extractor.batch(SAMPLES)` 并发抽取。

---

# 第 3 课 · 天气+日历助手（第一个 Agent）

知识点：`@tool`、ReAct 循环、`create_agent`、错误即信息。

```python
"""Agent 自主决定调用工具回答出行问题
依赖: pip install langchain langchain-openai python-dotenv"""
import sys
from datetime import date
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

@tool
def get_weather(city: str, days_ahead: int) -> str:
    """查询某城市未来某天的天气预报。
    city: 城市名，如 "杭州"; days_ahead: 距今几天, 0=今天, 6=下周六"""
    mock_db = {6: "晴转多云, 22~30°C, 降水概率10%", 7: "小雨, 20~26°C, 降水概率70%"}
    return f"{city} {days_ahead}天后: " + mock_db.get(days_ahead, "多云, 21~28°C, 降水概率30%")

@tool
def get_holiday_info(date_str: str) -> str:
    """查询某一天是否是法定节假日。date_str 格式 YYYY-MM-DD"""
    try:
        d = date.fromisoformat(date_str)
    except ValueError:
        return f"日期格式错误: {date_str}，请使用 YYYY-MM-DD 格式"  # 返回错误文字让模型自我修正
    return f"{date_str} 是{'周末' if d.weekday() >= 5 else '工作日'}，非法定节假日"

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather, get_holiday_info],
    system_prompt=f"你是出行规划助手，先用工具查证天气和日期性质再给建议。今天: {date.today().isoformat()}。",
)

question = sys.argv[1] if len(sys.argv) > 1 else "我下周末适合去杭州玩吗？"
result = agent.invoke({"messages": [{"role": "user", "content": question}]},
                      config={"recursion_limit": 15})   # 防死循环烧钱
for msg in result["messages"]:
    msg.pretty_print()
```

ReAct 循环的真实消息轨迹：

```
HumanMessage: 我下周末适合去杭州玩吗？
AIMessage:    content="" + tool_calls:[get_holiday_info(周六), get_holiday_info(周日)]   ← 第1轮确认日期
ToolMessage:  ×2
AIMessage:    tool_calls:[get_weather(杭州,6), get_weather(杭州,7)]                        ← 第2轮查天气
ToolMessage:  ×2
AIMessage:    "周六适合！周日降雨概率70%……"                                                ← 第3轮综合推理
```

要点：`@tool` 把签名翻译成 JSON Schema（与第2课 `with_structured_output` 是 function calling 的正反两面）；
AIMessage 的 content 可以为空（那轮只输出调用意图）；工具报错返回文字而非 raise，模型会自己改参数重试；
观察实验：删掉 docstring 看模型还会不会正确传参——接口文档真的被模型阅读。

---

# 第 4 课 · 持久会话（checkpointer 存 SQLite）

知识点：checkpointer=状态存档器、thread_id=会话线、每轮只传新消息。

```python
"""重启进程历史不丢的天气助手
依赖: pip install langgraph-checkpoint-sqlite"""
import sys
from datetime import date
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()
# ...（两个 @tool 与第3课相同，略）...

saver_cm = SqliteSaver.from_conn_string("chat_history.db")
saver = saver_cm.__enter__()          # 生产代码用 with 块包住 main()

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather, get_holiday_info],
    system_prompt=f"你是出行规划助手。今天: {date.today().isoformat()}。",
    checkpointer=saver,               # ← 加这一个参数就"记得住"
)

thread_id = sys.argv[1] if len(sys.argv) > 1 else "thread-1"
config = {"configurable": {"thread_id": thread_id}}

while True:
    try:
        user_input = input("\n你: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if user_input.lower() in ("exit", "quit") or not user_input:
        break
    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]},  # 只传新消息！
                          config=config)
    print(f"助手: {result['messages'][-1].content}")
```

验证实验：会话A问完退出 → 重进同 thread 问"刚才建议里周几去最好"答对（历史复活）；
开 thread-2 问同题说不知道（线程隔离）；`sqlite3 chat_history.db "SELECT DISTINCT thread_id FROM checkpoints;"`
看真相。要点：记忆的本质是"存档恢复"不是算法；对比无状态 API 手动拼全量 messages——范式差异；
短期记忆（本线程）≠ 长期记忆（跨线程 store）；隐患=历史无限膨胀，解法 trim/summarize。
生产迁移：SqliteSaver → PostgresSaver 只改 import + `saver.setup()`。

---

# 第 5 课 · 朴素 RAG 问答机器人（带来源引用）

知识点：load→split→embed→index、chunk 权衡、编号引用法、LCEL RAG 管道。

```python
"""文档问答机器人
依赖: pip install langchain langchain-openai langchain-community langchain-chroma chromadb"""
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
DB_DIR = "./rag_db"                    # Persistent Index

def build_index(paths):
    docs = [{"page_content": p.read_text(encoding="utf-8", errors="ignore"),
             "metadata": {"source": p.name}} for p in paths]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=80,
        separators=["\n## ", "\n# ", "\n\n", "\n", "。", "，", ""],  # 按结构切，别按字数硬切
    )
    return Chroma.from_documents(splitter.create_documents(docs),
        OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name="docs", persist_directory=DB_DIR)

def format_context(docs):                # 编号是我们拼的，模型只复述——引用可靠的诀窍
    return "\n\n---\n\n".join(f"[{i}] (来源: {d.metadata['source']})\n{d.page_content}"
                              for i, d in enumerate(docs, 1))

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是文档问答助手。只根据参考资料回答，资料没有的明确说不知道。
引用来源用编号如 [1] 或 [1][2]，编号必须真实存在。
参考资料：\n{context}"""),
    ("user", "{question}"),
])

index = build_index([Path(a) for a in sys.argv[1:]] or [Path("demo.md")])
retriever = index.as_retriever(search_kwargs={"k": 4})
rag_chain = (RunnableLambda(lambda q: {"docs": retriever.invoke(q["question"]), **q})
             .assign(context=RunnableLambda(lambda x: format_context(x["docs"])))
             | PROMPT | ChatOpenAI(model="gpt-4o-mini", temperature=0))

q = input("你: ")
hits = retriever.invoke(q)              # RAG调试第一步：先看检索到了什么
print(rag_chain.invoke({"question": q}).content)
```

三个必做实验：①问文档里没有的"餐补"验证拒答（再删掉约束句对比）；②chunk_size=100 答案残缺 /
2000 等于没检索——"效果差先调切分别换模型"的实证；③理解编号映射可回原文做校验。
固定管道 RAG 的缺陷：必检索、只检索一次、闲聊也检索 → 引出 Agentic RAG。

---

# 第 6 课 · 混合路由（LangGraph 第一课）

知识点：State/Node/条件边/compile；分诊台+专科通道架构。

架构：`问题 → [分类] → knowledge→[向量检索]→[生成] / data→[SQL]→[生成] / chat→[直接答]`

```python
"""LangGraph 混合路由问答"""
import sqlite3
from typing import Literal, TypedDict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langgraph.graph import END, StateGraph
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel

load_dotenv()
llm = init_chat_model("openai:gpt-4o-mini", temperature=0)

class State(TypedDict):        # 流水线上的 DTO 约定，节点间唯一的通信契约
    question: str; route: str; evidence: str; evidence_source: str; answer: str

# 数据侧：retriever(上课的rag_db) + sqlite orders 表(建表灌数略)

class Route(BaseModel):
    category: Literal["knowledge", "data", "chat"]

def classify_node(state):      # 节点=纯函数：输入State，返回"要更新的字段"，可单测
    r = llm.with_structured_output(Route).invoke(f"分类。knowledge=文档知识; data=数字统计; chat=闲聊。\n{state['question']}")
    return {"route": r.category}

def vector_search_node(state): ...   # 检索拼编号上下文 → 更新 evidence（同第5课）
def sql_node(state):                 # Text-to-SQL：生成→安全闸门(只放SELECT)→执行
    ...
    if not sql.lower().startswith("select"): return {"evidence": f"SQL被拦截: {sql}"}
    ...

def answer_node(state): ...          # 统一汇总口：证据+问题→生成回答
def chat_direct_node(state): ...     # 闲聊通道：省token

graph = StateGraph(State)
# add_node ×5 ...
graph.set_entry_point("classify")
graph.add_conditional_edges("classify", lambda s: s["route"],      # ← 网关路由
    {"knowledge": "vector_search", "data": "sql_query", "chat": "chat_direct"})
graph.add_edge("vector_search", "answer"); graph.add_edge("sql_query", "answer")
graph.add_edge("answer", END); graph.add_edge("chat_direct", END)
app = graph.compile()               # 声明拓扑≠可执行；compile 后才有执行引擎

final = app.invoke({"question": "..."})
```

五个抽象的后端映射：State=显式参数非ThreadLocal；节点=幂等处理器（加通道不改旧节点，开闭原则）；
条件边=Gateway 按特征转发（状态与路由决策分离，为 interrupt 留余地）；compile=源码要构建；
`create_agent` 内部本来就是一张编译好的"模型⇄工具"标准图。调试：`app.get_graph().draw_mermaid_png()`
可视化；`langgraph dev` + Studio 看每步 State diff。局限（已埋）：单一 route 处理不了一句混合两类问题。

---

# 第 7 课 · 人审闸门（interrupt + 跨进程挂起恢复）

知识点：propose→gate→commit 三段式；双闸门分层；挂起的本质=存档+原地续跑。

```python
"""SQL执行前人工审批，重启进程可续"""
from langgraph.types import Command, interrupt

def sql_generate_node(state):   # ① 只生成，存进State，绝不执行
    return {"sql": ...}

def approval_node(state):       # ② 闸门
    decision = interrupt({"type": "sql_approval", "sql": state["sql"],   # ← 挂起点
                          "question": state["question"]})
    if decision == "approved":
        return {}                       # 放行 → execute
    return {"evidence": f"SQL被人工拒绝: {state['sql']}（原因: {decision}）"}
    # 拒绝不删边不改路由，只写evidence，answer统一出口——图越扁平越好维护

def sql_execute_node(state):    # ③ 审批通过才会走到
    ...

# 安全路由（规则闸门，先于人审）：非SELECT / 含 delete|drop|update|insert|attach 不进执行通道
graph.add_conditional_edges("sql_generate", safety_router, {"approve": "approve", "answer": "answer"})

app = graph.compile(checkpointer=saver)   # ★ 没有 checkpointer 就没有"恢复"

result = app.invoke({"question": q}, config)
while "__interrupt__" in result:                    # 图可能停在挂起态
    req = result["__interrupt__"][0].value
    decision = input(f"待执行SQL: {req['sql']}\n批准? (yes/拒绝原因): ")
    result = app.invoke(Command(resume="approved" if decision == "yes" else decision), config)
```

灵魂实验：触发审批时 Ctrl+C 杀进程 → 重启 → `app.get_state(config)` 看到
`state.next=('approve',)`，SQL 还躺在存档里 → `invoke(Command(resume="approved"), config)` 复活继续。
第 4 课存的是聊天历史，这里存的是**执行到一半的程序状态**——这才是 LangGraph 存在的理由。
三个设计决策：生成与执行物理分离（SQL 经 State 中转才有挂起点和审查对象）；
规则闸门(毫秒级拦明显危险) + 人闸门(语义级风险) 分层；任何"先提议后执行"的敏感操作都套 propose→gate→commit。
伏笔：真实审批人在钉钉/飞书点按钮 → 挂起与恢复分属两进程、由 HTTP 触发——即第 8 课。

---

# 第 8 课 · 阶段五结业工程「企业知识库智能客服」

完整工程位于 `kb-assistant/`，README 含架构图、运行三步、6 步体验脚本、5 个迭代练习。

```
kb-assistant/
├── app/config.py      # 路径/模型配置唯一定义处
├── app/rag.py         # 索引构建：python -m app.rag
├── app/ticket_db.py   # 工单库：参数化访问，模型拿不到裸SQL
├── app/graph.py       # create_agent + 4工具 + escalate(interrupt) + SQLite checkpointer
├── app/main.py        # FastAPI: /api/chat(SSE) /api/resume /api/pending /api/reindex
├── app/evaluate.py    # 回归评估集：7条断言，非零退出码可接CI
├── static/index.html  # 聊天前端：流式渲染 + 人审卡片 + 刷新状态恢复
└── data/docs/         # 员工手册.md + 产品FAQ.md
```

相对第 7 课的工程升级点：
1. interrupt 的审批人解耦：挂起进 SQLite，SSE 以 `waiting=true` 结束，人工答复走独立
   `POST /api/resume` 注入 `Command(resume=...)`——审批人与用户完全分离；
2. `/api/pending` 查询口：刷新页面能恢复"停在人审"的卡片（分布式状态机恢复必须提供查询）；
3. 安全策略优先级：能枚举参数就不生成 SQL（query_ticket 只收工单号，语句写死参数化）；
4. 幻觉防线交给评估用例看守：问文档没有的"餐补"，编造即红灯；越权请求（写竞品差评）必拒；
5. 事件流协议：stream_mode=["messages","updates"] 混合消费，翻译为 token/tool/interrupt/done。

运行：`pip install -r requirements.txt` → 配 `.env` → `python -m app.rag` →
`python -m app.evaluate` → `python -m app.main` → http://127.0.0.1:8000

迭代练习（对应第 5 阶段主题）：with_fallbacks/with_retry 健壮性、LangSmith 可观测、
每 thread token 限额成本护栏、评估迁 pytest 接 CI、Docker 化部署。

---

# 全课程概念主线（一张表复习）

| 概念 | 首次出现 | 一句话本质 |
|---|---|---|
| 消息列表 = 对话 | 第1课 | messages 数组就是全部状态 |
| function calling 正反两面 | 第2/3课 | with_structured_output 收结构 / @tool 供结构 |
| ReAct 循环 | 第3课 | 模型→tool_calls→回填→再想，直到答案 |
| checkpointer | 第4课 | 记忆与挂起都是"存档恢复"，thread_id=session id |
| 历史膨胀 | 第4课作业 | 50轮后撞窗口，解法 trim/summarize |
| 编号引用 | 第5课 | 引用格式拼进上下文，模型只复述，可回验 |
| State/节点/条件边 | 第6课 | 纯函数+DTO+网关路由；compile 才成机器 |
| interrupt | 第7课 | 存档暂停，Command(resume) 原地续跑，可跨进程 |
| propose→gate→commit | 第7课 | 一切敏感操作三段式 |
| SSE 事件协议 | 第8课 | 图的挂起态是一种业务状态，要能查询和恢复 |
| 评估即闸门 | 第8课 | 非零退出码进 CI，改 prompt 不再裸奔 |

未完事项：Agentic RAG（分类器换成自主 Agent / 混合路由的两类问题并行版）、
多 Agent Supervisor 模式（路线图第 4 阶段作业 1"内容生产流水线"）、
五个迭代练习任选其一。
