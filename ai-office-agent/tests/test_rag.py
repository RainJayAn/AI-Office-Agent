from langchain_core.documents import Document


def test_rag_ingest_and_query(client, sample_docs_dir):
    ingest_response = client.post(
        "/rag/ingest",
        json={"directory": str(sample_docs_dir)},
    )
    assert ingest_response.status_code == 200
    ingest_data = ingest_response.json()
    assert ingest_data["document_count"] == 2
    assert ingest_data["chunk_count"] >= 2

    query_response = client.post(
        "/rag/query",
        json={
            "query": "Project Alpha deadline",
            "top_k": 2,
        },
    )
    assert query_response.status_code == 200
    query_data = query_response.json()
    assert "Project Alpha deadline" in query_data["answer"]
    assert query_data["docs"]
    assert query_data["citations"]
    assert "vector_score" in query_data["docs"][0]
    assert "rerank_score" in query_data["docs"][0]


def test_rag_can_ingest_pdf_file(client, monkeypatch, sample_docs_dir):
    pdf_path = sample_docs_dir / "report.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake pdf")
    monkeypatch.setattr(
        "app.rag.ingest._read_pdf_documents",
        lambda *_: [
            Document(
                page_content="OpenAI funding report on February 27, 2026.",
                metadata={"source": "report.pdf", "page": 2, "file_type": "pdf"},
            )
        ],
    )

    ingest_response = client.post(
        "/rag/ingest",
        json={"file_path": str(pdf_path)},
    )

    assert ingest_response.status_code == 200
    ingest_data = ingest_response.json()
    assert ingest_data["document_count"] == 1

    query_response = client.post(
        "/rag/query",
        json={
            "query": "February 27 2026 funding report",
            "top_k": 1,
        },
    )
    assert query_response.status_code == 200
    query_data = query_response.json()
    assert query_data["docs"]
    assert query_data["docs"][0]["source"].endswith("report.pdf")
    assert query_data["docs"][0]["page"] == 2


def test_rag_upload_endpoint_can_save_and_ingest_pdf(client, monkeypatch):
    monkeypatch.setattr(
        "app.rag.ingest._read_pdf_documents",
        lambda *_: [
            Document(
                page_content="Uploaded PDF content about Project Alpha.",
                metadata={"source": "project.pdf", "page": 1, "file_type": "pdf"},
            )
        ],
    )

    response = client.post(
        "/rag/upload?filename=project.pdf",
        content=b"%PDF-1.4 upload",
        headers={"Content-Type": "application/pdf"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "project.pdf"
    assert data["saved_path"].endswith("project.pdf")
    assert data["ingest"]["document_count"] == 1


def test_chat_can_use_retrieve_docs_tool(client, sample_docs_dir):
    client.post("/rag/ingest", json={"directory": str(sample_docs_dir)})

    response = client.post(
        "/chat",
        json={
            "message": "please search the document for Project Alpha deadline",
            "use_rag": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["tool_traces"]
    assert data["tool_traces"][0]["tool_name"] == "retrieve_docs"
    assert "model tool answer:" in data["answer"]
    assert data["citations"]
