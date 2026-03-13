def test_list_tools(client):
    response = client.get("/tools")

    assert response.status_code == 200
    data = response.json()
    tool_names = {tool["name"] for tool in data["tools"]}
    assert "draft_email" in tool_names
    assert "retrieve_docs" in tool_names
    assert "web_search" in tool_names


def test_run_draft_email_tool(client):
    response = client.post(
        "/tools/run",
        json={
            "tool_name": "draft_email",
            "args": {
                "recipient": "team@example.com",
                "subject": "Project Update",
                "purpose": "Share the latest project status",
                "key_points": ["timeline is stable", "next review is Friday"],
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["tool_name"] == "draft_email"
    assert "Project Update" in data["output"]
    assert "team@example.com" in data["output"]


def test_run_missing_tool_returns_404(client):
    response = client.post(
        "/tools/run",
        json={
            "tool_name": "missing_tool",
            "args": {},
        },
    )

    assert response.status_code == 404
    data = response.json()
    assert data["error"]["code"] == "tool_execution_error"
    assert data["error"]["details"]["tool_name"] == "missing_tool"


def test_run_web_search_tool(client):
    response = client.post(
        "/tools/run",
        json={
            "tool_name": "web_search",
            "args": {
                "query": "latest qwen release",
                "max_results": 2,
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["tool_name"] == "web_search"
    assert "Search result for latest qwen release" in data["output"]
    assert "https://example.com/search-result" in data["output"]
