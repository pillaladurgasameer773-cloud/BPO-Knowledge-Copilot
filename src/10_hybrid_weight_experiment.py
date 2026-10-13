import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. LOAD KNOWLEDGE BASE
# --------------------------------------------------

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


# --------------------------------------------------
# 2. TF-IDF
# --------------------------------------------------

tfidf = TfidfVectorizer(stop_words="english")

tfidf_vectors = tfidf.fit_transform(
    df["search_text"]
)


# --------------------------------------------------
# 3. EMBEDDINGS
# --------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

embedding_vectors = model.encode(
    df["search_text"].tolist(),
    normalize_embeddings=True
)


# --------------------------------------------------
# 4. HARD TEST QUESTIONS
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
# 5. PRE-CALCULATE SCORES
# --------------------------------------------------

all_tfidf_scores = []
all_embedding_scores = []

for case in test_cases:

    query = case["question"]

    # TF-IDF score
    query_tfidf = tfidf.transform([query])

    tfidf_scores = cosine_similarity(
        query_tfidf,
        tfidf_vectors
    )[0]

    # Embedding score
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    embedding_scores = cosine_similarity(
        query_embedding,
        embedding_vectors
    )[0]

    all_tfidf_scores.append(tfidf_scores)
    all_embedding_scores.append(embedding_scores)


# --------------------------------------------------
# 6. TEST DIFFERENT WEIGHTS
# --------------------------------------------------

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


print("\n" + "=" * 70)
print("HYBRID WEIGHT EXPERIMENT")
print("=" * 70)

print("\nEmbedding weight | TF-IDF weight | Top-1 | Recall@3")
print("-" * 60)


for embedding_weight in weights:

    tfidf_weight = 1 - embedding_weight

    top1_correct = 0
    top3_correct = 0

    for i, case in enumerate(test_cases):

        hybrid_scores = (
            tfidf_weight * all_tfidf_scores[i]
            + embedding_weight * all_embedding_scores[i]
        )

        top_indices = hybrid_scores.argsort()[::-1][:3]

        retrieved_ids = df.iloc[
            top_indices
        ]["article_id"].tolist()

        expected = case["expected"]

        if retrieved_ids[0] == expected:
            top1_correct += 1

        if expected in retrieved_ids:
            top3_correct += 1

    total = len(test_cases)

    top1_accuracy = top1_correct / total
    recall_at_3 = top3_correct / total

    print(
        f"{embedding_weight:16.1f} | "
        f"{tfidf_weight:13.1f} | "
        f"{top1_accuracy:6.2%} | "
        f"{recall_at_3:9.2%}"
    )