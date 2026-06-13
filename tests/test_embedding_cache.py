import numpy as np

from ragbench.models.embeddings import (
    EMBEDDING_CACHE,
    CachedEmbeddingModel,
    HashingEmbeddingModel,
)


class CountingModel(HashingEmbeddingModel):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.texts_embedded = 0

    def embed_texts(self, texts):
        self.calls += 1
        self.texts_embedded += len(texts)
        return super().embed_texts(texts)


def setup_function(_):
    EMBEDDING_CACHE.clear()
    EMBEDDING_CACHE.enabled = True


def test_second_system_reuses_corpus_embeddings():
    texts = ["alpha beta gamma", "delta epsilon"]
    first = CountingModel()
    second = CountingModel()
    result_a = CachedEmbeddingModel(first).embed_texts(texts)
    result_b = CachedEmbeddingModel(second).embed_texts(texts)
    assert first.texts_embedded == 2
    assert second.texts_embedded == 0
    assert np.allclose(result_a.vectors, result_b.vectors)
    assert EMBEDDING_CACHE.stats()["hits"] == 2


def test_cache_hits_still_charge_tokens_to_each_system():
    texts = ["alpha beta gamma"]
    CachedEmbeddingModel(CountingModel()).embed_texts(texts)
    result = CachedEmbeddingModel(CountingModel()).embed_texts(texts)
    assert result.input_tokens > 0


def test_duplicate_texts_within_batch_embedded_once():
    model = CountingModel()
    result = CachedEmbeddingModel(model).embed_texts(["same text", "same text", "other"])
    assert model.texts_embedded == 2
    assert result.vectors.shape[0] == 3
    assert np.allclose(result.vectors[0], result.vectors[1])


def test_embed_query_bypasses_cache():
    model = CountingModel()
    cached = CachedEmbeddingModel(model)
    cached.embed_query("what is alpha?")
    cached.embed_query("what is alpha?")
    assert model.texts_embedded == 2
    assert EMBEDDING_CACHE.stats()["hits"] == 0


def test_cache_can_be_disabled():
    EMBEDDING_CACHE.enabled = False
    model = CountingModel()
    cached = CachedEmbeddingModel(model)
    cached.embed_texts(["alpha"])
    cached.embed_texts(["alpha"])
    assert model.texts_embedded == 2
