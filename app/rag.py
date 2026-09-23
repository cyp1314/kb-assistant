"""RAG 索引模块：build_index() 建库 / get_retriever() 取检索器。

命令行重建索引：  python -m app.rag
"""
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL)


def build_index() -> int:
    """load -> split -> embed -> index。返回 chunk 数。"""
    texts, metadatas = [], []
    for p in sorted(Path(config.DOCS_DIR).glob("*.md")):
        texts.append(p.read_text(encoding="utf-8", errors="ignore"))
        metadatas.append({"source": p.name})
    if not texts:
        raise SystemExit(f"{config.DOCS_DIR} 下没有 .md 文档")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n## ", "\n# ", "\n\n", "\n", "。", "，", ""],
    )
    chunks = splitter.create_documents(texts, metadatas=metadatas)
    Chroma.from_documents(
        documents=chunks,
        embedding=_embeddings(),
        collection_name="docs",
        persist_directory=str(config.DB_DIR),
    )
    print(f"索引完成：{len(texts)} 个文档 -> {len(chunks)} 个 chunk -> {config.DB_DIR}")
    return len(chunks)


def get_retriever(k: int = 4):
    """打开已建好的 Persistent Index。建库前先跑 python -m app.rag"""
    if not Path(config.DB_DIR).exists():
        raise SystemExit("向量库不存在，先执行:  python -m app.rag")
    return Chroma(
        collection_name="docs",
        persist_directory=str(config.DB_DIR),
        embedding_function=_embeddings(),
    ).as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    build_index()
