import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD CHUNKS
# ============================================================

print("=" * 70)
print("SEMANTIC SEARCH")
print("=" * 70)

with open("../data/article_chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

print("\nChunks loaded:", len(chunks))


# ============================================================
# 2. LOAD SAVED EMBEDDINGS
# ============================================================

embeddings = np.load("../data/article_embeddings.npz")["embeddings"]

print("Embeddings loaded:", len(embeddings))
print("Embedding shape:", embeddings.shape)


# ============================================================
# 3. LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully.")


# ============================================================
# 4. SEMANTIC SEARCH FUNCTION
# ============================================================

def semantic_search(query, top_k=3):

    # Convert user's question into an embedding
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    # Compare query vector with all article vectors
    scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    # Get highest scoring articles
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({
            "article_id": chunks[index]["article_id"],
            "title": chunks[index]["title"],
            "page": chunks[index]["page"],
            "score": float(scores[index]),
            "text": chunks[index]["text"]
        })

    return results


# ============================================================
# 5. TEST QUESTIONS
# ============================================================

questions = [

    "customer says claim was rejected even though they had coverage",

    "member wants to sign up for a new insurance plan",

    "customer is angry and wants someone senior to handle the issue",

    "customer says the amount on their bill is incorrect"
]


# ============================================================
# 6. RUN SEARCH
# ============================================================

for question in questions:

    print("\n" + "=" * 70)
    print("QUESTION:")
    print(question)

    results = semantic_search(question, top_k=3)

    print("\nTOP RESULTS:")
    print("-" * 70)

    for rank, result in enumerate(results, start=1):

        print(
            f"{rank}. "
            f"{result['article_id']} | "
            f"{result['title']} | "
            f"Score: {result['score']:.4f}"
        )

        print(
            f"   Page: {result['page']}"
        )


# ============================================================
# 7. FINISHED
# ============================================================

print("\n" + "=" * 70)
print("SEMANTIC SEARCH COMPLETE")
print("=" * 70)