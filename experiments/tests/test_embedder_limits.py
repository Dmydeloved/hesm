from types import SimpleNamespace

import openai

from core.embedder import (
    BailianEmbedder,
    embedding_token_count,
    split_embedding_text,
)


def test_short_embedding_content_is_unchanged():
    text = "A short English and 中文 embedding input."
    chunks, token_count = split_embedding_text(text)
    assert chunks == [text]
    assert token_count == embedding_token_count(text)
    assert token_count <= 8192


def test_long_embedding_content_is_losslessly_split_below_limit():
    text = ("Long memory content 中文内容。" * 5000) + "END"
    chunks, token_count = split_embedding_text(text)
    assert token_count > 8192
    assert len(chunks) > 1
    assert "".join(chunks) == text
    assert all(embedding_token_count(chunk) <= 7800 for chunk in chunks)


def test_embedder_never_sends_more_than_configured_chunk_tokens(monkeypatch):
    calls = []

    class FakeEmbeddings:
        def create(self, *, model, input):
            calls.append((model, input))
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[1.0, 2.0, 3.0])]
            )

    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **kwargs: SimpleNamespace(embeddings=FakeEmbeddings()),
    )
    embedder = BailianEmbedder(
        api_key="test",
        model="text-embedding-v4",
        base_url="https://example.invalid/v1",
        max_input_tokens=8192,
        chunk_tokens=7800,
    )
    vector = embedder.embed("memory content " * 10000)

    assert len(calls) > 1
    assert all(model == "text-embedding-v4" for model, _ in calls)
    assert all(embedding_token_count(chunk) <= 7800 for _, chunk in calls)
    assert len(vector) == 3
