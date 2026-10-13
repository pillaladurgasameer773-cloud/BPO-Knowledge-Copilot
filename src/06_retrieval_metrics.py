import pandas as pd
from sentence_transformers import SentenceTransformer
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


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Create article embeddings
article_embeddings = model.encode(
    df["search_text"].tolist(),
    normalize_embeddings=True
)


def semantic_search(query, top_k=3):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        query_embedding,
        article_embeddings
    )[0]

    top_indices = scores.argsort()[::-1][:top_k]

    results = df.iloc[top_indices].copy()
    results["score"] = scores[top_indices]

    return results


# Questions with known correct articles
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


top1_correct = 0
top3_correct = 0


for case in test_cases:

    results = semantic_search(case["question"], top_k=3)

    retrieved_ids = results["article_id"].tolist()

    top1 = retrieved_ids[0] == case["expected"]
    top3 = case["expected"] in retrieved_ids

    if top1:
        top1_correct += 1

    if top3:
        top3_correct += 1

    print("\nQuestion:", case["question"])
    print("Expected:", case["expected"])
    print("Retrieved:", retrieved_ids)
    print("Top-1:", "Correct" if top1 else "Wrong")
    print("Top-3:", "Found" if top3 else "Not Found")


# Calculate metrics
total = len(test_cases)

top1_accuracy = top1_correct / total
recall_at_3 = top3_correct / total


print("\n" + "=" * 60)
print("RETRIEVAL EVALUATION")
print("=" * 60)

print(f"Top-1 Accuracy: {top1_accuracy:.2%}")
print(f"Recall@3:       {recall_at_3:.2%}")