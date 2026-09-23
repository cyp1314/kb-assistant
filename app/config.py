"""集中配置：所有路径/模型名只在这里定义一次，其他模块从这里读。"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
DB_DIR = DATA_DIR / "rag_db"                 # Chroma 持久目录
CHECKPOINT_DB = DATA_DIR / "checkpoints.db"  # LangGraph 会话存档
TICKET_DB = DATA_DIR / "tickets.db"          # 业务工单库

load_dotenv(ROOT / ".env")

CHAT_MODEL = os.getenv("CHAT_MODEL", "openai:gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
