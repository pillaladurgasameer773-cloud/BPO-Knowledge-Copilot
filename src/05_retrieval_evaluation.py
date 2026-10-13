import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load knowledge base
df = pd.read_csv("../data/BPO_Knowledge_Copilot_Gold.csv")


# Combine article text
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


# Create embeddings for all articles
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


# Test questions
test_questions = [
    "customer says claim was rejected but they had coverage",
    "customer wants to know the status of their claim",
    "customer wants to request a reconsideration of a claim",
    "customer is asking about eligibility",
    "customer has a billing problem",
    "customer wants help with enrollment",
    "customer needs to speak to a supervisor",
    "customer wants to update their information"
]


# Run evaluation
for question in test_questions:

    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("=" * 70)

    results = semantic_search(question)

    print(
        results[
            ["article_id", "title", "category", "score"]
        ].to_string(index=False)
    )