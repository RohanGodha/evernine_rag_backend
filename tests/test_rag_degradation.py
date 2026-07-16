"""Tests that RAG degrades gracefully when embeddings are unavailable.

We don't require torch/pgvector to be installed for the suite to pass: the
guards must make these paths safe.
"""
from app import embeddings, rag


def test_embeddings_encode_returns_none_when_unavailable(monkeypatch):
    monkeypatch.setattr(embeddings, "_load_model", lambda: None)
    assert embeddings.encode("some text") is None
    assert embeddings.is_available() is False


def test_compute_embedding_none_without_pgvector(monkeypatch):
    monkeypatch.setattr(rag, "PGVECTOR_AVAILABLE", False)
    vec = rag.compute_embedding("name", "desc", "Meta", "conversion")
    assert vec is None


def test_build_embedding_text_composition():
    text = embeddings.build_embedding_text(
        "Diwali Blowout", "festive push", "Meta", "conversion"
    )
    assert "Diwali Blowout" in text
    assert "channel=Meta" in text
    assert "objective=conversion" in text


def test_deterministic_observations_from_aggregate():
    agg = {
        "campaign_count": 2,
        "total_spend": 100.0,
        "total_revenue": 300.0,
        "portfolio_roas": 3.0,
        "by_channel": {
            "Meta": {"spend": 50.0, "revenue": 50.0, "count": 1, "score_sum": 40, "roas": 1.0, "avg_health": 40.0},
            "Email": {"spend": 50.0, "revenue": 250.0, "count": 1, "score_sum": 80, "roas": 5.0, "avg_health": 80.0},
        },
    }
    obs = rag._deterministic_observations(agg)
    assert 1 <= len(obs) <= 3
    joined = " ".join(obs)
    assert "Meta" in joined  # worst ROAS surfaced
