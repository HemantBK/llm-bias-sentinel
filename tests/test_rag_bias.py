"""Tests for RAG bias testing module (no Ollama/ChromaDB needed)."""


class TestRAGBiasDocuments:
    """Test the built-in test document sets."""

    def test_biased_documents_exist(self):
        from src.rag_bias.rag_bias_tester import BIASED_DOCUMENTS

        assert "race_negative" in BIASED_DOCUMENTS
        assert "gender_stereotyped" in BIASED_DOCUMENTS
        assert "neutral" in BIASED_DOCUMENTS

    def test_documents_have_required_fields(self):
        from src.rag_bias.rag_bias_tester import BIASED_DOCUMENTS

        for category, docs in BIASED_DOCUMENTS.items():
            assert len(docs) >= 2, f"{category} needs at least 2 docs"
            for doc in docs:
                assert "text" in doc
                assert "topic" in doc
                assert "bias_level" in doc
                assert len(doc["text"]) > 20

    def test_neutral_docs_are_neutral(self):
        from src.rag_bias.rag_bias_tester import BIASED_DOCUMENTS

        for doc in BIASED_DOCUMENTS["neutral"]:
            assert doc["bias_level"] == "none"

    def test_biased_docs_are_marked(self):
        from src.rag_bias.rag_bias_tester import BIASED_DOCUMENTS

        for doc in BIASED_DOCUMENTS["race_negative"]:
            assert doc["bias_level"] in ("medium", "high")


class TestRAGTestConfig:
    """Test RAG test configuration."""

    def test_default_config(self):
        from src.rag_bias.rag_bias_tester import RAGTestConfig

        cfg = RAGTestConfig()
        assert cfg.embedding_model == "all-MiniLM-L6-v2"
        assert cfg.top_k == 3
        assert cfg.collection_name == "bias_test_collection"

    def test_custom_config(self):
        from src.rag_bias.rag_bias_tester import RAGTestConfig

        cfg = RAGTestConfig(top_k=5, embedding_model="custom-model")
        assert cfg.top_k == 5
        assert cfg.embedding_model == "custom-model"

    def test_rag_availability_flag(self):
        from src.rag_bias.rag_bias_tester import RAG_AVAILABLE

        assert isinstance(RAG_AVAILABLE, bool)
