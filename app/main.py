"""HTTP 服务层：FastAPI + SSE 流式输出。

接口：
  GET  /                 聊天页面（static/index.html）
  POST /api/chat         {thread_id, message}      -> SSE 流
  POST /api/resume       {thread_id, reply}        -> 人工答复注入挂起的 interrupt，继续 SSE 流
  GET  /api/pending      ?thread_id=xxx            -> 该会话是否停在人审等待（页面刷新恢复用）
  POST /api/reindex      重建向量索引
"""
import json
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, graph as agent_mod, rag
from langgraph.types import Command

app = FastAPI(title="KB Assistant", version="1.0")

STATIC_DIR = Path(config.ROOT) / "static"


@lru_cache
def agent():
    return agent_mod.get_agent()


def cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 15}


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def stream_run(input_payload, thread_id: str):
    """把 LangGraph 的混合事件流翻译成前端友好的 SSE 协议：
       token / tool / interrupt / done
    interrupt 发生时流直接结束（done 事件带 waiting=true），
    人工通过 /api/resume 送回 Command(resume=...) 再继续。"""
    waiting = False
    for mode, event in agent().stream(
        input_payload, config=cfg(thread_id),
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            chunk, meta = event
            content = getattr(chunk, "content", "")
            node = meta.get("langgraph_node", "")
            if not content:
                continue
            if node == "tools":
                # 工具返回的检索结果，折叠成一条 tool 事件
                yield sse({"type": "tool", "text": content[:200]})
            else:
                # model 节点的增量 token
                yield sse({"type": "token", "text": content})
        elif mode == "updates" and isinstance(event, dict):
            for node, update in event.items():
                if "__interrupt__" in update or (
                    node == "__interrupt__" and isinstance(update, (list, tuple))
                ):
                    interrupts = update if node == "__interrupt__" else update["__interrupt__"]
                    payload = getattr(interrupts[0], "value", interrupts[0])
                    waiting = True
                    yield sse({"type": "interrupt", "payload": payload})
    yield sse({"type": "done", "waiting": waiting})


class ChatIn(BaseModel):
    thread_id: str
    message: str


class ResumeIn(BaseModel):
    thread_id: str
    reply: str


@app.post("/api/chat")
def chat(body: ChatIn):
    if not body.message.strip():
        raise HTTPException(400, "empty message")
    payload = {"messages": [{"role": "user", "content": body.message}]}
    return StreamingResponse(stream_run(payload, body.thread_id),
                             media_type="text/event-stream")


@app.post("/api/resume")
def resume(body: ResumeIn):
    # 从存档恢复：输入不再是新消息，而是人类决策
    payload = Command(resume=body.reply)
    return StreamingResponse(stream_run(payload, body.thread_id),
                             media_type="text/event-stream")


@app.get("/api/pending")
def pending(thread_id: str):
    state = agent().get_state(cfg(thread_id))
    if state.next:  # 图停在非 END 的节点 => 正在等人
        for task in state.tasks:
            intr = getattr(task, "interrupts", None)
            if intr:
                return {"waiting": True, "payload": intr[0].value}
    return {"waiting": False}


@app.post("/api/reindex")
def reindex():
    n = rag.build_index()
    return {"chunks": n}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
