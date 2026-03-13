from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from app.services.rag_service import RAGService


router = APIRouter(prefix="/rag", tags=["rag"])
rag_service = RAGService()


class RAGQueryRequest(BaseModel):
    query: str
    top_k: int = 3


class RAGQueryResponse(BaseModel):
    answer: str
    docs: list[dict]
    citations: list[dict]


class IngestRequest(BaseModel):
    file_path: str | None = None
    directory: str | None = None


@router.post("/query", response_model=RAGQueryResponse, summary="Query RAG")
async def query_rag(request: RAGQueryRequest) -> RAGQueryResponse:
    result = rag_service.query(query=request.query, top_k=request.top_k)
    return RAGQueryResponse(**result)


@router.post("/ingest", summary="Ingest Documents")
async def ingest_docs(request: IngestRequest) -> dict:
    data_path = request.file_path or request.directory
    if not data_path:
        raise HTTPException(status_code=400, detail="file_path or directory is required")

    return rag_service.ingest(data_path=data_path)


@router.post("/upload", summary="Upload and Ingest Document")
async def upload_document(
    request: Request,
    filename: str = Query(..., description="Original filename, e.g. report.pdf"),
) -> dict:
    content = await request.body()
    if not content:
        raise HTTPException(status_code=400, detail="uploaded file is empty")

    return rag_service.save_uploaded_file(
        filename=filename,
        content=content,
    )
