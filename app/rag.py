"""RAG 索引模块：建库 / 增量增删 / 取检索器。

命令行全量重建：  python -m app.rag
命令行增量添加：  python -m app.rag.add data/docs/新文档.md
"""
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL)


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n## ", "\n# ", "\n\n", "\n", "。", "，", ""],
    )


def _chroma() -> Chroma:
    """打开已存在的 Chroma 客户端（只读+追加）。"""
    return Chroma(
        collection_name="docs",
        persist_directory=str(config.DB_DIR),
        embedding_function=_embeddings(),
    )


def build_index() -> int:
    """全量重建：清空旧 collection → 加载所有 .md → 切分 → 写入。返回 chunk 数。"""
    texts, metadatas = [], []
    for p in sorted(Path(config.DOCS_DIR).glob("*.md")):
        texts.append(p.read_text(encoding="utf-8", errors="ignore"))
        metadatas.append({"source": p.name})
    if not texts:
        raise SystemExit(f"{config.DOCS_DIR} 下没有 .md 文档")

    chunks = _splitter().create_documents(texts, metadatas=metadatas)

    # 先删后建，保证幂等
    existing = Path(config.DB_DIR)
    if existing.exists():
        import shutil
        shutil.rmtree(existing)

    Chroma.from_documents(
        documents=chunks,
        embedding=_embeddings(),
        collection_name="docs",
        persist_directory=str(config.DB_DIR),
    )
    print(f"索引完成：{len(texts)} 个文档 -> {len(chunks)} 个 chunk -> {config.DB_DIR}")
    return len(chunks)


def add_document(path: str) -> int:
    """增量添加单个 .md 文件到已有索引。返回新增 chunk 数。
    若该文件已存在于索引中，先删除旧 chunks 再写入，避免重复。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在：{p}")
    if not p.suffix.lower() == ".md":
        raise ValueError("目前只支持 .md 文件")

    if not Path(config.DB_DIR).exists():
        raise SystemExit("向量库不存在，先执行:  python -m app.rag")

    text = p.read_text(encoding="utf-8", errors="ignore")
    chunks = _splitter().create_documents([text], metadatas=[{"source": p.name}])

    client = _chroma()
    # 先删旧（如果同 source 已存在）
    client.delete(where={"source": p.name})
    # 再追加
    client.add_documents(chunks)

    print(f"已添加 {p.name}：{len(chunks)} 个 chunk")
    return len(chunks)


def delete_document(source: str) -> int:
    """从索引中删除指定 source 的所有 chunks。返回删除数。"""
    if not Path(config.DB_DIR).exists():
        raise SystemExit("向量库不存在")

    client = _chroma()
    existing = client.get(where={"source": source})
    count = len(existing.get("ids", []))
    if count == 0:
        print(f"索引中没有 source={source} 的文档")
        return 0

    client.delete(where={"source": source})
    print(f"已删除 {source}：{count} 个 chunk")
    return count


def list_documents() -> list[dict]:
    """列出索引中所有文档（source → chunk 数）。"""
    if not Path(config.DB_DIR).exists():
        return []
    client = _chroma()
    result = client.get(include=["metadatas"])
    from collections import Counter
    c = Counter(m["source"] for m in result.get("metadatas", []) if m)
    return [{"source": s, "chunks": n} for s, n in sorted(c.items())]


def get_retriever(k: int = 4):
    """打开已建好的 Persistent Index。"""
    if not Path(config.DB_DIR).exists():
        raise SystemExit("向量库不存在，先执行:  python -m app.rag")
    return _chroma().as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        add_document(sys.argv[1])
    else:
        build_index()
