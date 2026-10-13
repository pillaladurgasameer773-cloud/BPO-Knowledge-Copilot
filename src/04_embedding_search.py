import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load our knowledge base
df = pd.read_csv("../data/BPO_Knowledge_Copilot_Gold.csv")


# Combine important article fields into one searchable text
df["search_text"] = (
    df["title"].fillna("")
    + " "
    + df["scenario"].fillna("")
    + " "
    + df["procedure"].fillna("")
    + " "
    + df["aliases"].fillna("")
)


# Load the pretrained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Convert every article into an embedding
article_embeddings = model.encode(
    df["search_text"].tolist(),
    normalize_embeddings=True
)


# Semantic search function
def semantic_search(query, top_k=3):

    # Convert the user's question into an embedding
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    # Compare question with every article
    scores = cosine_similarity(
        query_embedding,
        article_embeddings
    )[0]

    # Get the highest-scoring articles
    top_indices = scores.argsort()[::-1][:top_k]

    results = df.iloc[top_indices].copy()

    # Store similarity score
    results["score"] = scores[top_indices]

    return results


# Test question
question = "customer says claim was rejected but they had coverage"

results = semantic_search(question)


print("Search question:")
print(question)

print("\nTop semantic search results:")
print(
    results[
        ["article_id", "title", "category", "score"]
    ].to_string(index=False)
)