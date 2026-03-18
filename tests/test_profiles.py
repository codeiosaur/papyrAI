import pdfwiki.main as main


def test_resolve_run_profile_uses_alias_balanced_to_hybrid(monkeypatch):
    monkeypatch.delenv("PDF_TO_NOTES_PROFILE", raising=False)
    name, settings = main._resolve_run_profile("balanced")
    assert name == "hybrid"
    assert settings["write_max_tokens"] == main.RUN_PROFILE_SETTINGS["hybrid"]["write_max_tokens"]


def test_resolve_run_profile_invalid_falls_back_to_hybrid(monkeypatch):
    monkeypatch.delenv("PDF_TO_NOTES_PROFILE", raising=False)
    name, _ = main._resolve_run_profile("ultra-fast")
    assert name == "hybrid"


def test_resolve_max_workers_uses_profile_default_when_no_arg_or_env(monkeypatch):
    monkeypatch.delenv("PDF_TO_NOTES_MAX_WORKERS", raising=False)
    assert main._resolve_max_workers(None, default_workers=2) == 2
