import pdfwiki.retriever as retriever


def test_keywords_from_concept_extracts_words_abbreviations_and_phrase():
    keywords = retriever._keywords_from_concept("One-Time Pad (OTP)")

    assert "otp" in keywords
    assert "pad" in keywords
    assert "one-time pad" in keywords


def test_retrieve_chunks_returns_top_scored_chunks_with_separator():
    chunks = [
        "Symmetric encryption uses shared keys.",
        "RSA is a public key cryptosystem based on factoring.",
        "Diffie-Hellman supports key exchange.",
    ]

    selected = retriever.retrieve_chunks(
        chunks,
        concept="RSA",
        related_concepts=["Public Key Cryptography"],
        top_k=2,
        max_chars=2000,
    )

    assert "RSA is a public key cryptosystem" in selected
    assert "---" in selected


def test_retrieve_chunks_falls_back_when_no_scores():
    chunks = [
        "Topic A has no overlap.",
        "Topic B has no overlap.",
    ]

    selected = retriever.retrieve_chunks(chunks, concept="Quantum", top_k=2)

    assert selected == chunks[0]


def test_score_chunk_prefers_primary_concept_over_related_context():
    chunk = "RSA provides public key encryption and supports secure key exchange."

    primary = retriever.score_chunk(
        chunk,
        concept="RSA",
        related_concepts=["Diffie-Hellman Key Exchange"],
    )
    related_only = retriever.score_chunk(
        chunk,
        concept="Hash Function",
        related_concepts=["RSA"],
    )

    assert primary > related_only


def test_deduplicate_chunks_removes_exact_and_near_duplicates():
    chunks = [
        "RSA is a public key cryptosystem.",
        "RSA is a public key cryptosystem.",
        "RSA is a public key cryptosystem and enables secure communication.",
        "Diffie-Hellman is for key exchange.",
    ]

    deduped = retriever.deduplicate_chunks(chunks)

    assert "RSA is a public key cryptosystem." in deduped
    assert "Diffie-Hellman is for key exchange." in deduped
    assert len(deduped) < len(chunks)


def test_bm25_scores_prefers_more_query_term_overlap():
    chunks = [
        "RSA RSA factoring modulus.",
        "Symmetric encryption with block ciphers.",
    ]

    scores = retriever.bm25_scores(chunks, ["RSA factoring"])

    assert len(scores) == 2
    assert scores[0] > scores[1]


def test_retrieve_chunks_uses_bm25_for_specific_term_matching():
    chunks = [
        "General cryptography overview with no named scheme.",
        "The Kasiski test helps break Vigenere ciphers.",
        "Public key systems include RSA and Diffie-Hellman.",
    ]

    selected = retriever.retrieve_chunks(
        chunks,
        concept="Kasiski Test",
        related_concepts=["Vigenere Cipher"],
        top_k=1,
        max_chars=2000,
    )

    assert "Kasiski test" in selected
