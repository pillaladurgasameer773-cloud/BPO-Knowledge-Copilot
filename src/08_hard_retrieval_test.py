import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load knowledge base
df = pd.read_csv("../data/BPO_Knowledge_Copilot_Gold.csv")

df["search_text"] = (
    df["title"].fillna("")
    + " "
    + df["scenario"].fillna("")
    + " "
    + df["procedure"].fillna("")
    + " "
    + df["aliases"].fillna("")
)


# -----------------------------
# TF-IDF
# -----------------------------

tfidf = TfidfVectorizer(stop_words="english")

tfidf_vectors = tfidf.fit_transform(df["search_text"])


def tfidf_search(query, top_k=3):

    query_vector = tfidf.transform([query])

    scores = cosine_similarity(
        query_vector,
        tfidf_vectors
    )[0]

    top_indices = scores.argsort()[::-1][:top_k]

    return df.iloc[top_indices]["article_id"].tolist()


# -----------------------------
# EMBEDDINGS
# -----------------------------

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


# -----------------------------
# HARD TEST QUESTIONS
# -----------------------------

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


# -----------------------------
# EVALUATION
# -----------------------------

tfidf_top1 = 0
tfidf_top3 = 0

embedding_top1 = 0
embedding_top3 = 0


for case in test_cases:

    question = case["question"]
    expected = case["expected"]

    tfidf_results = tfidf_search(question)
    embedding_results = embedding_search(question)

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


# -----------------------------
# FINAL RESULTS
# -----------------------------

total = len(test_cases)

print("\n" + "=" * 70)
print("HARD RETRIEVAL TEST")
print("=" * 70)

print(f"TF-IDF Top-1 Accuracy:      {tfidf_top1 / total:.2%}")
print(f"TF-IDF Recall@3:            {tfidf_top3 / total:.2%}")

print(f"Embedding Top-1 Accuracy:   {embedding_top1 / total:.2%}")
print(f"Embedding Recall@3:         {embedding_top3 / total:.2%}")