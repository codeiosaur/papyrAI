import pdfwiki.ai_client as ai_client


def test_query_ollama_uses_task_specific_model(monkeypatch):
    monkeypatch.setattr(ai_client, "PROVIDER", "ollama")
    monkeypatch.setattr(
        ai_client,
        "OLLAMA_TASK_MODELS",
        {
            "cheap": "llama3.1:8b",
            "extract": "mistral:latest",
            "write": "qwen3:32b",
        },
    )

    captured = {}

    def fake_query_ollama(prompt, system, max_tokens, model):
        captured["model"] = model
        return "ok"

    monkeypatch.setattr(ai_client, "_query_ollama", fake_query_ollama)

    result = ai_client.query("hello", task="extract")

    assert result == "ok"
    assert captured["model"] == "mistral:latest"


def test_query_ollama_falls_back_to_single_model(monkeypatch):
    monkeypatch.setattr(ai_client, "PROVIDER", "ollama")
    monkeypatch.setattr(ai_client, "OLLAMA_MODEL", "llama3.1:8b")
    monkeypatch.setattr(ai_client, "OLLAMA_TASK_MODELS", {"cheap": "", "extract": "", "write": ""})

    captured = {}

    def fake_query_ollama(prompt, system, max_tokens, model):
        captured["model"] = model
        return "ok"

    monkeypatch.setattr(ai_client, "_query_ollama", fake_query_ollama)

    result = ai_client.query("hello", task="write")

    assert result == "ok"
    assert captured["model"] == "llama3.1:8b"


def test_query_rejects_invalid_task_for_ollama_too(monkeypatch):
    monkeypatch.setattr(ai_client, "PROVIDER", "ollama")

    try:
        ai_client.query("hello", task="invalid")
    except ValueError as exc:
        assert "Invalid task" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid task")
