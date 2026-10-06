import json

from radar import brief


def test_generate_parses_and_cleans_llm_output(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(brief, "document_text", lambda url: "x" * 1000)
    reply = {"summary": "Guarding at the Tshwane office for 60 months.",
             "scope": "Two guards per shift", "requirements": ["PSIRA registration", ""],
             "cidb_grading": "null", "watch_out": None, "preference": "80/20"}
    monkeypatch.setattr(brief, "_call_gemini",
                        lambda prompt, key, model: "```json\n" + json.dumps(reply) + "\n```")
    tender = {"id": 1, "description": "Security", "documents": json.dumps(
        [{"name": "doc.pdf", "url": "https://example.org/doc.pdf"}])}
    out = brief.generate(tender)
    assert out["scope"] == ["Two guards per shift"]
    assert out["requirements"] == ["PSIRA registration"]
    assert out["cidb_grading"] is None and out["watch_out"] == []
    assert out["source"] == "doc.pdf" and out["model"].startswith("gemini/")
