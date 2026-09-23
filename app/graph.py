"""核心：智能客服 Agent 图。

架构 = 第2阶段 create_agent（模型自主决定调哪个工具、调几次）
     + 第3阶段 RAG 检索工具（Agentic RAG：查不查、查几轮由模型定）
     + 第4阶段 interrupt 人审 + SQLite checkpointer 持久会话

工具即权限：search_docs / query_ticket / create_ticket / escalate_to_human，
模型只能在授权的工具集内行动，这就是"工具权限最小化"。
"""
import sqlite3

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt

from . import config, rag, ticket_db


# ---------------- ① 工具层 ----------------
@tool
def search_docs(question: str) -> str:
    """检索公司知识库（员工手册、产品FAQ等制度与文档），返回带编号的参考片段。
    回答制度、产品使用、政策类问题前必须调用本工具，并基于返回内容作答。
    question: 要查询的问题，用完整的问句效果更好。"""
    docs = rag.get_retriever().invoke(question)
    if not docs:
        return "知识库中没有检索到相关内容"
    return "\n\n".join(
        f"[{i}] (来源: {d.metadata['source']}) {d.page_content}"
        for i, d in enumerate(docs, 1)
    )


@tool
def query_ticket(ticket_id: str) -> str:
    """按工单号查询工单进度。ticket_id 格式如 TK-1001。用户没给工单号时先向用户索要。"""
    t = ticket_db.find_ticket(ticket_id)
    if not t:
        return f"未找到工单 {ticket_id}，请确认工单号是否正确"
    return f"工单{t['id']} | {t['subject']} | 状态: {t['status']} | 备注: {t['note']}"


@tool
def create_ticket(user_name: str, subject: str) -> str:
    """为用户创建新工单。user_name: 用户姓名; subject: 问题的一句话描述。"""
    tid = ticket_db.create_ticket(user_name, subject)
    return f"已创建工单 {tid}（{subject}），客服将在1个工作日内联系"


@tool
def escalate_to_human(question: str, reason: str) -> str:
    """转接人工客服。仅当以下情况调用：用户明确要求人工、投诉、
    知识库查不到且用户坚持要答案、涉及退款金额审批等超出你权限的决定。
    question: 用户的原始问题; reason: 为什么需要人工介入。"""
    # ★ 挂起点：图在此存档暂停，等 HTTP /api/resume 送入人工答复后原地续跑
    human_reply = interrupt({
        "type": "escalation",
        "question": question,
        "reason": reason,
    })
    return f"人工客服答复: {human_reply}"


# ---------------- ② Agent 图 ----------------
SYSTEM_PROMPT = """你是企业智能客服"小北"，服务范围：公司制度咨询、产品售后支持、工单处理。

工作守则：
1. 制度/产品问题必须先调用 search_docs 查证，回答时保留资料编号引用（如[1]）；
   知识库没有的内容明确说"这个问题我需要转人工确认"，绝不编造。
2. 工单查询用 query_ticket，建新单用 create_ticket（先复述问题让用户确认再建）。
3. 满足 escalate_to_human 描述的条件时才转人工。
4. 回答简洁友好，中文，不超过150字（引用资料原文除外）。
5. 与上述服务范围无关的请求（写竞品差评、查他人隐私等）礼貌拒绝。"""


def get_agent():
    """单例构建。lru_cache 避免每个请求重建图。"""
    model = init_chat_model(config.CHAT_MODEL, temperature=0)
    # 会话存档：生产换 PostgresSaver，只改这两行
    conn = sqlite3.connect(config.CHECKPOINT_DB, check_same_thread=False)
    saver = SqliteSaver(conn)
    return create_agent(
        model=model,
        tools=[search_docs, query_ticket, create_ticket, escalate_to_human],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=saver,
    )
