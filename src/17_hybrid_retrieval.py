import json
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("HYBRID RETRIEVAL")
print("=" * 70)

with open("../data/article_chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

print("\nChunks loaded:", len(chunks))


# ============================================================
# 2. PREPARE TEXT
# ============================================================

texts = []

for chunk in chunks:

    text = (
        chunk.get("title", "")
        + " "
        + chunk.get("text", "")
    )

    texts.append(text)


# ============================================================
# 3. TF-IDF
# ============================================================

print("\nBuilding TF-IDF model...")

tfidf_vectorizer = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = tfidf_vectorizer.fit_transform(texts)


# ============================================================
# 4. LOAD EMBEDDINGS
# ============================================================

print("Loading embeddings...")

embedding_matrix = np.load(
    "../data/article_embeddings.npz"
)["embeddings"]

print("Embedding matrix:", embedding_matrix.shape)


# ============================================================
# 5. LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Model loaded.")


# ============================================================
# 6. HYBRID SEARCH
# ============================================================

def hybrid_search(
    query,
    embedding_weight=0.7,
    top_k=3
):

    tfidf_weight = 1 - embedding_weight

    # -----------------------------
    # TF-IDF score
    # -----------------------------

    query_tfidf = tfidf_vectorizer.transform(
        [query]
    )

    tfidf_scores = cosine_similarity(
        query_tfidf,
        tfidf_matrix
    )[0]


    # -----------------------------
    # Embedding score
    # -----------------------------

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    embedding_scores = cosine_similarity(
        query_embedding,
        embedding_matrix
    )[0]


    # -----------------------------
    # Normalize scores
    # -----------------------------

    tfidf_min = tfidf_scores.min()
    tfidf_max = tfidf_scores.max()

    embedding_min = embedding_scores.min()
    embedding_max = embedding_scores.max()


    tfidf_normalized = (
        (tfidf_scores - tfidf_min)
        / (tfidf_max - tfidf_min + 1e-10)
    )


    embedding_normalized = (
        (embedding_scores - embedding_min)
        / (embedding_max - embedding_min + 1e-10)
    )


    # -----------------------------
    # Hybrid score
    # -----------------------------

    hybrid_scores = (
        tfidf_weight * tfidf_normalized
        +
        embedding_weight * embedding_normalized
    )


    # -----------------------------
    # Top results
    # -----------------------------

    top_indices = np.argsort(
        hybrid_scores
    )[::-1][:top_k]


    results = []

    for index in top_indices:

        results.append({
            "article_id": chunks[index]["article_id"],
            "title": chunks[index]["title"],
            "tfidf_score": float(
                tfidf_scores[index]
            ),
            "embedding_score": float(
                embedding_scores[index]
            ),
            "hybrid_score": float(
                hybrid_scores[index]
            )
        })

    return results


# ============================================================
# 7. TEST QUESTION
# ============================================================

question = (
    "customer says claim was rejected "
    "even though they had coverage"
)


print("\n")
print("=" * 70)
print("TEST QUESTION")
print("=" * 70)

print(question)


# ============================================================
# 8. TEST DIFFERENT WEIGHTS
# ============================================================

weights = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0
]


print("\n")
print("=" * 70)
print("HYBRID WEIGHT EXPERIMENT")
print("=" * 70)

print(
    "\nEmbedding Weight | TF-IDF Weight | Top Result"
)

print("-" * 70)


for embedding_weight in weights:

    results = hybrid_search(
        question,
        embedding_weight=embedding_weight,
        top_k=1
    )

    top_result = results[0]

    tfidf_weight = 1 - embedding_weight

    print(
        f"{embedding_weight:15.1f} | "
        f"{tfidf_weight:13.1f} | "
        f"{top_result['article_id']}"
    )


# ============================================================
# 9. SHOW BEST BALANCED RESULT
# ============================================================

print("\n")
print("=" * 70)
print("EXAMPLE HYBRID RESULT")
print("=" * 70)

results = hybrid_search(
    question,
    embedding_weight=0.7,
    top_k=3
)


for rank, result in enumerate(
    results,
    start=1
):

    print(
        f"\n{rank}. "
        f"{result['article_id']} | "
        f"{result['title']}"
    )

    print(
        f"   TF-IDF score: "
        f"{result['tfidf_score']:.4f}"
    )

    print(
        f"   Embedding score: "
        f"{result['embedding_score']:.4f}"
    )

    print(
        f"   Hybrid score: "
        f"{result['hybrid_score']:.4f}"
    )


print("\n")
print("=" * 70)
print("HYBRID RETRIEVAL COMPLETE")
print("=" * 70)