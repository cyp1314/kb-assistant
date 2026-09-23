"""模拟业务工单库（sqlite）。真实项目里这就是你们公司的工单系统 API/DB。

约定：工具层只通过这里的函数访问，Agent 永远拿不到裸 SQL 执行权——
      查询是写死的参数化语句，模型只能提供"参数"（工单号），不能提供"语句"。
      这与第4阶段 Text-to-SQL 的"生成+闸门"是不同的安全策略：能枚举参数就不生成SQL。
"""
import sqlite3

from . import config

_SEED = [
    # (id, user, subject, status, note)
    ("TK-1001", "张三", "X200路由器红灯常亮无法上网", "处理中", "已寄换机，物流单号SF1234"),
    ("TK-1002", "李四", "年假3天未休申请结转", "已完成", "HR已确认结转至2027-03-31"),
    ("TK-1003", "王五", "差旅报销被驳回：超30天", "已驳回", "2026-06-01消费 2026-08-05提交"),
]


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(config.TICKET_DB)


def ensure_db() -> None:
    config.TICKET_DB.parent.mkdir(parents=True, exist_ok=True)
    con = _conn()
    con.execute("""
        CREATE TABLE IF NOT EXISTS tickets(
            id TEXT PRIMARY KEY, user TEXT, subject TEXT, status TEXT, note TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )""")
    if con.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 0:
        con.executemany("INSERT INTO tickets VALUES (?,?,?,?,?,datetime('now'))", _SEED)
    con.commit()
    con.close()


def find_ticket(ticket_id: str) -> dict | None:
    ensure_db()
    con = _conn()
    row = con.execute(
        "SELECT id, user, subject, status, note FROM tickets WHERE id = ?",
        (ticket_id.strip().upper(),),
    ).fetchone()
    con.close()
    if not row:
        return None
    return dict(zip(["id", "user", "subject", "status", "note"], row))


def create_ticket(user: str, subject: str) -> str:
    ensure_db()
    con = _conn()
    seq = con.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] + 1
    tid = f"TK-{1000 + seq}"
    con.execute("INSERT INTO tickets(id, user, subject, status, note) VALUES (?,?,?,?,?)",
                (tid, user, subject, "待受理", "由智能客服创建"))
    con.commit()
    con.close()
    return tid
