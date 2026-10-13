import json
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD CHUNKS
# ============================================================

print("=" * 70)
print("HYBRID RETRIEVAL EVALUATION")
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

vectorizer = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = vectorizer.fit_transform(texts)


# ============================================================
# 4. LOAD EMBEDDINGS
# ============================================================

embedding_matrix = np.load(
    "../data/article_embeddings.npz"
)["embeddings"]

print(
    "Embedding matrix:",
    embedding_matrix.shape
)


# ============================================================
# 5. LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Model loaded.")


# ============================================================
# 6. TEST QUESTIONS
# ============================================================

test_questions = [

    {
        "question": "customer says claim was rejected even though they had coverage",
        "expected": "CLM-001"
    },

    {
        "question": "customer wants to know the current status of their claim",
        "expected": "CLM-002"
    },

    {
        "question": "the same claim appears to have been submitted twice",
        "expected": "CLM-004"
    },

    {
        "question": "customer says the amount on their bill is incorrect",
        "expected": "BIL-004"
    },

    {
        "question": "member wants to sign up for a new plan",
        "expected": "ENR-001"
    },

    {
        "question": "member wants to change their current insurance plan",
        "expected": "ENR-005"
    },

    {
        "question": "customer is angry and wants someone senior to handle the issue",
        "expected": "COM-002"
    },

    {
        "question": "customer has an urgent issue that needs escalation",
        "expected": "ESC-004"
    },

    {
        "question": "member needs to verify their identity before receiving support",
        "expected": "MEM-001"
    },

    {
        "question": "member needs a replacement insurance card",
        "expected": "MEM-004"
    },

    {
        "question": "quality review finds that an agent missed a required verification step",
        "expected": "QA-001"
    },

    {
        "question": "agent needs to check whether a knowledge article is still current",
        "expected": "QA-004"
    }
]


# ============================================================
# 7. HYBRID SEARCH FUNCTION
# ============================================================

def hybrid_search(
    query,
    embedding_weight,
    top_k=3
):

    tfidf_weight = 1 - embedding_weight


    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    query_tfidf = vectorizer.transform(
        [query]
    )

    tfidf_scores = cosine_similarity(
        query_tfidf,
        tfidf_matrix
    )[0]


    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    embedding_scores = cosine_similarity(
        query_embedding,
        embedding_matrix
    )[0]


    # --------------------------------------------------------
    # NORMALIZE SCORES
    # --------------------------------------------------------

    tfidf_normalized = (
        (tfidf_scores - tfidf_scores.min())
        /
        (
            tfidf_scores.max()
            - tfidf_scores.min()
            + 1e-10
        )
    )


    embedding_normalized = (
        (embedding_scores - embedding_scores.min())
        /
        (
            embedding_scores.max()
            - embedding_scores.min()
            + 1e-10
        )
    )


    # --------------------------------------------------------
    # COMBINE SCORES
    # --------------------------------------------------------

    hybrid_scores = (
        tfidf_weight * tfidf_normalized
        +
        embedding_weight * embedding_normalized
    )


    # --------------------------------------------------------
    # TOP RESULTS
    # --------------------------------------------------------

    top_indices = np.argsort(
        hybrid_scores
    )[::-1][:top_k]


    return [
        chunks[index]["article_id"]
        for index in top_indices
    ]


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


# ============================================================
# 9. EVALUATE EACH WEIGHT
# ============================================================

evaluation_results = []


print("\n")
print("=" * 70)
print("WEIGHT EVALUATION")
print("=" * 70)

print(
    "\nEmbedding | TF-IDF | Top-1 | Recall@3"
)

print("-" * 70)


for embedding_weight in weights:

    tfidf_weight = 1 - embedding_weight

    top1_correct = 0
    recall3_correct = 0


    for item in test_questions:

        results = hybrid_search(
            item["question"],
            embedding_weight,
            top_k=3
        )

        expected = item["expected"]


        # Top-1
        if results[0] == expected:
            top1_correct += 1


        # Recall@3
        if expected in results:
            recall3_correct += 1


    total = len(test_questions)


    top1_accuracy = (
        top1_correct / total
    ) * 100


    recall_at_3 = (
        recall3_correct / total
    ) * 100


    evaluation_results.append({

        "embedding_weight":
            embedding_weight,

        "tfidf_weight":
            tfidf_weight,

        "top1":
            top1_accuracy,

        "recall3":
            recall_at_3
    })


    print(
        f"{embedding_weight:9.1f} | "
        f"{tfidf_weight:7.1f} | "
        f"{top1_accuracy:6.2f}% | "
        f"{recall_at_3:8.2f}%"
    )


# ============================================================
# 10. FIND BEST WEIGHT
# ============================================================

best_result = max(
    evaluation_results,
    key=lambda x: (
        x["top1"],
        x["recall3"]
    )
)


# ============================================================
# 11. FINAL RESULT
# ============================================================

print("\n")
print("=" * 70)
print("BEST HYBRID CONFIGURATION")
print("=" * 70)

print(
    "\nBest embedding weight:",
    best_result["embedding_weight"]
)

print(
    "Best TF-IDF weight:",
    best_result["tfidf_weight"]
)

print(
    "Top-1 Accuracy:",
    f"{best_result['top1']:.2f}%"
)

print(
    "Recall@3:",
    f"{best_result['recall3']:.2f}%"
)


print("\n")
print("=" * 70)
print("HYBRID EVALUATION COMPLETE")
print("=" * 70)