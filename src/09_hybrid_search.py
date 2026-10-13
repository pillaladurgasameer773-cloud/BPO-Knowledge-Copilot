import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load knowledge base
df = pd.read_csv("../data/BPO_Knowledge_Copilot_Gold.csv")


# Combine article fields
df["search_text"] = (
    df["title"].fillna("")
    + " "
    + df["scenario"].fillna("")
    + " "
    + df["procedure"].fillna("")
    + " "
    + df["aliases"].fillna("")
)


# --------------------------------------------------
# 1. TF-IDF
# --------------------------------------------------

tfidf = TfidfVectorizer(stop_words="english")

tfidf_vectors = tfidf.fit_transform(
    df["search_text"]
)


# --------------------------------------------------
# 2. EMBEDDINGS
# --------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

embedding_vectors = model.encode(
    df["search_text"].tolist(),
    normalize_embeddings=True
)


# --------------------------------------------------
# 3. HYBRID SEARCH
# --------------------------------------------------

def hybrid_search(query, top_k=3):

    # TF-IDF similarity
    query_tfidf = tfidf.transform([query])

    tfidf_scores = cosine_similarity(
        query_tfidf,
        tfidf_vectors
    )[0]


    # Embedding similarity
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    embedding_scores = cosine_similarity(
        query_embedding,
        embedding_vectors
    )[0]


    # Combine both scores
    hybrid_scores = (
        0.5 * tfidf_scores
        + 0.5 * embedding_scores
    )


    # Get top results
    top_indices = hybrid_scores.argsort()[::-1][:top_k]

    results = df.iloc[top_indices].copy()

    results["tfidf_score"] = tfidf_scores[top_indices]
    results["embedding_score"] = embedding_scores[top_indices]
    results["hybrid_score"] = hybrid_scores[top_indices]

    return results


# --------------------------------------------------
# 4. HARD TEST SET
# --------------------------------------------------

test_cases = [
    {
        "question": "Why was my medical request refused even though I was insured?",
        "expected": "CLM-001"
    },
    {
        "question": "The member wants to change their contact details.",
        "expected": "MEM-003"
    },
    {
        "question": "Why did I get billed twice for the same thing?",
        "expected": "BIL-001"
    },
    {
        "question": "The customer disagrees with the decision and wants the case reviewed again.",
        "expected": "CLM-003"
    },
    {
        "question": "The customer wants to know whether their family member can receive coverage.",
        "expected": "ELG-006"
    },
    {
        "question": "The member is trying to sign up for a new plan.",
        "expected": "ENR-001"
    },
    {
        "question": "The customer is angry and wants someone senior to handle the issue.",
        "expected": "COM-001"
    },
    {
        "question": "The customer says the amount on their bill doesn't look right.",
        "expected": "BIL-004"
    }
]


# --------------------------------------------------
# 5. EVALUATE HYBRID SEARCH
# --------------------------------------------------

top1_correct = 0
top3_correct = 0


for case in test_cases:

    question = case["question"]
    expected = case["expected"]

    results = hybrid_search(question)

    retrieved_ids = results["article_id"].tolist()

    if retrieved_ids[0] == expected:
        top1_correct += 1

    if expected in retrieved_ids:
        top3_correct += 1

    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("EXPECTED:", expected)

    print(
        results[
            [
                "article_id",
                "title",
                "tfidf_score",
                "embedding_score",
                "hybrid_score"
            ]
        ].to_string(index=False)
    )


# --------------------------------------------------
# 6. FINAL RESULTS
# --------------------------------------------------

total = len(test_cases)

print("\n" + "=" * 70)
print("HYBRID RETRIEVAL RESULTS")
print("=" * 70)

print(f"Top-1 Accuracy: {top1_correct / total:.2%}")
print(f"Recall@3:       {top3_correct / total:.2%}")