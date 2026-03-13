import shutil
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.core.path import get_vector_db_dir, resolve_project_path, to_relative_display_path
from app.rag import retriever as rag_retriever


SUPPORTED_RAG_SUFFIXES = {".md", ".txt", ".pdf"}


def load_documents(data_path: str) -> list[Document]:
    path = resolve_project_path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    files: list[Path]
    if path.is_dir():
        files = [
            file_path
            for file_path in path.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_RAG_SUFFIXES
        ]
        source_root = path
    else:
        if path.suffix.lower() not in SUPPORTED_RAG_SUFFIXES:
            raise ValueError("Only .md, .txt, and .pdf files are supported")
        files = [path]
        source_root = path.parent

    documents: list[Document] = []
    for file_path in files:
        source = to_relative_display_path(file_path, base_path=source_root)
        documents.extend(_build_documents_for_file(file_path, source))

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.RAG_CHUNK_SIZE,
        chunk_overlap=settings.RAG_CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "")
        page = chunk.metadata.get("page")
        if page is None:
            chunk.metadata["chunk_id"] = f"{source}::chunk-{index}"
        else:
            chunk.metadata["chunk_id"] = f"{source}::page-{page}::chunk-{index}"

    return chunks


def build_vector_index(chunks: list[Document]) -> dict:
    settings = get_settings()
    persist_directory = _get_store_path()
    if persist_directory.exists():
        shutil.rmtree(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    Chroma.from_documents(
        documents=chunks,
        embedding=rag_retriever.get_embedding_model(),
        collection_name=settings.CHROMA_COLLECTION_NAME,
        persist_directory=str(persist_directory),
    )

    return {
        "store_path": persist_directory,
        "chunk_count": len(chunks),
    }


def ingest_documents(data_path: str) -> dict:
    resolved_data_path = resolve_project_path(data_path)
    documents = load_documents(data_path)
    chunks = split_documents(documents)
    index_info = build_vector_index(chunks)
    return {
        "data_path": to_relative_display_path(resolved_data_path),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "store_path": to_relative_display_path(index_info["store_path"]),
    }


def _build_documents_for_file(file_path: Path, source: str) -> list[Document]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf_documents(file_path, source)

    content = file_path.read_text(encoding="utf-8-sig").strip()
    if not content:
        return []

    return [
        Document(
            page_content=content,
            metadata={
                "source": source,
                "page": 1,
                "file_type": suffix.lstrip("."),
            },
        )
    ]


def _read_pdf_documents(file_path: Path, source: str) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(file_path))
    documents: list[Document] = []
    for index, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if not page_text:
            continue
        documents.append(
            Document(
                page_content=page_text,
                metadata={
                    "source": source,
                    "page": index,
                    "file_type": "pdf",
                },
            )
        )
    return documents


def _get_store_path() -> Path:
    settings = get_settings()
    return get_vector_db_dir(settings.VECTOR_DB_PATH)
