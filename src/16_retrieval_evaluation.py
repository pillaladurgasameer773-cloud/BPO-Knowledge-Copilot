import json
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD KNOWLEDGE CHUNKS
# ============================================================

print("=" * 70)
print("RETRIEVAL EVALUATION")
print("=" * 70)

with open("../data/article_chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

print("\nChunks loaded:", len(chunks))


# ============================================================
# 2. PREPARE ARTICLE TEXT
# ============================================================

texts = []

for chunk in chunks:

    text = (
        chunk.get("title", "")
        + " "
        + chunk.get("text", "")
    )

    texts.append(text)

print("Texts prepared:", len(texts))


# ============================================================
# 3. TF-IDF MODEL
# ============================================================

print("\nBuilding TF-IDF model...")

tfidf_vectorizer = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

print("TF-IDF matrix shape:", tfidf_matrix.shape)


# ============================================================
# 4. EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embedding_matrix = np.load(
    "../data/article_embeddings.npz"
)["embeddings"]

print("Embedding matrix shape:", embedding_matrix.shape)


# ============================================================
# 5. TEST QUESTIONS
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
# 6. SEARCH FUNCTIONS
# ============================================================

def tfidf_search(query, top_k=3):

    query_vector = tfidf_vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    )[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        chunks[index]["article_id"]
        for index in top_indices
    ]


def embedding_search(query, top_k=3):

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        query_embedding,
        embedding_matrix
    )[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        chunks[index]["article_id"]
        for index in top_indices
    ]


# ============================================================
# 7. EVALUATION
# ============================================================

tfidf_top1_correct = 0
tfidf_recall3_correct = 0

embedding_top1_correct = 0
embedding_recall3_correct = 0


print("\n")
print("=" * 70)
print("INDIVIDUAL RESULTS")
print("=" * 70)


for item in test_questions:

    question = item["question"]
    expected = item["expected"]

    tfidf_results = tfidf_search(
        question,
        top_k=3
    )

    embedding_results = embedding_search(
        question,
        top_k=3
    )

    tfidf_top1 = tfidf_results[0]
    embedding_top1 = embedding_results[0]

    tfidf_correct = tfidf_top1 == expected
    embedding_correct = embedding_top1 == expected

    tfidf_recall = expected in tfidf_results
    embedding_recall = expected in embedding_results

    if tfidf_correct:
        tfidf_top1_correct += 1

    if tfidf_recall:
        tfidf_recall3_correct += 1

    if embedding_correct:
        embedding_top1_correct += 1

    if embedding_recall:
        embedding_recall3_correct += 1

    print("\nQuestion:")
    print(question)

    print("Expected:", expected)

    print(
        "TF-IDF Top-3:",
        tfidf_results
    )

    print(
        "Embedding Top-3:",
        embedding_results
    )

    print(
        "TF-IDF Top-1:",
        "PASS" if tfidf_correct else "FAIL"
    )

    print(
        "Embedding Top-1:",
        "PASS" if embedding_correct else "FAIL"
    )


# ============================================================
# 8. CALCULATE METRICS
# ============================================================

total_questions = len(test_questions)


tfidf_top1_accuracy = (
    tfidf_top1_correct / total_questions
) * 100


tfidf_recall_at_3 = (
    tfidf_recall3_correct / total_questions
) * 100


embedding_top1_accuracy = (
    embedding_top1_correct / total_questions
) * 100


embedding_recall_at_3 = (
    embedding_recall3_correct / total_questions
) * 100


# ============================================================
# 9. FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("FINAL RETRIEVAL COMPARISON")
print("=" * 70)

print("\nTotal test questions:", total_questions)

print("\nTF-IDF")
print("-" * 40)

print(
    f"Top-1 Accuracy: {tfidf_top1_accuracy:.2f}%"
)

print(
    f"Recall@3:       {tfidf_recall_at_3:.2f}%"
)


print("\nEMBEDDINGS")
print("-" * 40)

print(
    f"Top-1 Accuracy: {embedding_top1_accuracy:.2f}%"
)

print(
    f"Recall@3:       {embedding_recall_at_3:.2f}%"
)


# ============================================================
# 10. WINNER
# ============================================================

print("\n")
print("=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)

if embedding_top1_accuracy > tfidf_top1_accuracy:

    print(
        "Embedding retrieval has better Top-1 accuracy."
    )

elif embedding_top1_accuracy < tfidf_top1_accuracy:

    print(
        "TF-IDF has better Top-1 accuracy."
    )

else:

    print(
        "TF-IDF and Embeddings have the same Top-1 accuracy."
    )


if embedding_recall_at_3 > tfidf_recall_at_3:

    print(
        "Embedding retrieval has better Recall@3."
    )

elif embedding_recall_at_3 < tfidf_recall_at_3:

    print(
        "TF-IDF has better Recall@3."
    )

else:

    print(
        "TF-IDF and Embeddings have the same Recall@3."
    )


print("\n")
print("=" * 70)
print("RETRIEVAL EVALUATION COMPLETE")
print("=" * 70)