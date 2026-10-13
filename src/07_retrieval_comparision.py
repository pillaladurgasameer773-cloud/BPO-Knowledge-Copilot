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
# 1. TF-IDF SETUP
# --------------------------------------------------

tfidf_vectorizer = TfidfVectorizer(stop_words="english")

tfidf_vectors = tfidf_vectorizer.fit_transform(
    df["search_text"]
)


def tfidf_search(query, top_k=3):

    query_vector = tfidf_vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        tfidf_vectors
    )[0]

    top_indices = scores.argsort()[::-1][:top_k]

    return df.iloc[top_indices]["article_id"].tolist()


# --------------------------------------------------
# 2. EMBEDDING SETUP
# --------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

embedding_vectors = model.encode(
    df["search_text"].tolist(),
    normalize_embeddings=True
)


def embedding_search(query, top_k=3):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        query_embedding,
        embedding_vectors
    )[0]

    top_indices = scores.argsort()[::-1][:top_k]

    return df.iloc[top_indices]["article_id"].tolist()


# --------------------------------------------------
# 3. TEST DATA
# --------------------------------------------------

test_cases = [
    {
        "question": "customer says claim was rejected but they had coverage",
        "expected": "CLM-001"
    },
    {
        "question": "customer wants to know the status of their claim",
        "expected": "CLM-002"
    },
    {
        "question": "customer wants to request a reconsideration of a claim",
        "expected": "CLM-003"
    },
    {
        "question": "customer is asking about dependent eligibility",
        "expected": "ELG-006"
    },
    {
        "question": "customer was charged an unexpected billing amount",
        "expected": "BIL-004"
    },
    {
        "question": "customer wants to start a new enrollment",
        "expected": "ENR-001"
    },
    {
        "question": "customer needs to speak to a supervisor",
        "expected": "COM-001"
    },
    {
        "question": "customer wants to update their phone number",
        "expected": "MEM-003"
    }
]


# --------------------------------------------------
# 4. EVALUATION
# --------------------------------------------------

tfidf_top1 = 0
tfidf_top3 = 0

embedding_top1 = 0
embedding_top3 = 0


for case in test_cases:

    question = case["question"]
    expected = case["expected"]

    tfidf_results = tfidf_search(question, top_k=3)
    embedding_results = embedding_search(question, top_k=3)

    if tfidf_results[0] == expected:
        tfidf_top1 += 1

    if expected in tfidf_results:
        tfidf_top3 += 1

    if embedding_results[0] == expected:
        embedding_top1 += 1

    if expected in embedding_results:
        embedding_top3 += 1

    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("EXPECTED:", expected)

    print("\nTF-IDF:")
    print(tfidf_results)

    print("\nEMBEDDINGS:")
    print(embedding_results)


# --------------------------------------------------
# 5. FINAL COMPARISON
# --------------------------------------------------

total = len(test_cases)

print("\n" + "=" * 70)
print("RETRIEVAL METHOD COMPARISON")
print("=" * 70)

print(
    f"TF-IDF Top-1 Accuracy: "
    f"{tfidf_top1 / total:.2%}"
)

print(
    f"TF-IDF Recall@3:       "
    f"{tfidf_top3 / total:.2%}"
)

print(
    f"Embedding Top-1 Accuracy: "
    f"{embedding_top1 / total:.2%}"
)

print(
    f"Embedding Recall@3:       "
    f"{embedding_top3 / total:.2%}"
)