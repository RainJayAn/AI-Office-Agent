from pathlib import Path

from app.core.exceptions import RAGPipelineError
from app.core.config import get_settings
from app.core.path import get_upload_dir, to_relative_display_path
from app.rag.ingest import ingest_documents
from app.rag.pipeline import run_rag


class RAGService:
    def query(self, query: str, top_k: int = 3) -> dict:
        settings = get_settings()
        effective_top_k = top_k or settings.RAG_DEFAULT_TOP_K
        try:
            return run_rag(query=query, top_k=effective_top_k)
        except Exception as exc:
            raise RAGPipelineError(
                "Failed to run RAG query",
                status_code=500,
                details={"query": query, "top_k": effective_top_k},
            ) from exc

    def ingest(self, data_path: str) -> dict:
        try:
            return ingest_documents(data_path=data_path)
        except FileNotFoundError as exc:
            raise RAGPipelineError(
                str(exc),
                status_code=404,
                details={"data_path": data_path},
            ) from exc
        except ValueError as exc:
            raise RAGPipelineError(
                str(exc),
                status_code=400,
                details={"data_path": data_path},
            ) from exc
        except Exception as exc:
            raise RAGPipelineError(
                "Failed to ingest documents",
                status_code=500,
                details={"data_path": data_path},
            ) from exc

    def save_uploaded_file(self, filename: str, content: bytes) -> dict:
        suffix = Path(filename).suffix.lower()
        if suffix not in {".md", ".txt", ".pdf"}:
            raise RAGPipelineError(
                "Only .md, .txt, and .pdf files are supported",
                status_code=400,
                details={"filename": filename},
            )

        upload_dir = get_upload_dir()
        safe_name = Path(filename).name
        target_path = upload_dir / safe_name
        target_path.write_bytes(content)

        try:
            ingest_result = self.ingest(data_path=str(target_path))
        except Exception:
            if target_path.exists():
                target_path.unlink()
            raise

        return {
            "filename": safe_name,
            "saved_path": to_relative_display_path(target_path),
            "ingest": ingest_result,
        }
