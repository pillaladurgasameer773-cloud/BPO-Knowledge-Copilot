import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load knowledge base
df = pd.read_csv("../data/BPO_Knowledge_Copilot_Gold.csv")


# Combine important fields into one searchable text
df["search_text"] = (
    df["title"].fillna("")
    + " "
    + df["scenario"].fillna("")
    + " "
    + df["procedure"].fillna("")
    + " "
    + df["aliases"].fillna("")
)


# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words="english")

article_vectors = vectorizer.fit_transform(df["search_text"])


def tfidf_search(query, top_k=3):

    # Convert the user question into a TF-IDF vector
    query_vector = vectorizer.transform([query])

    # Calculate similarity between question and articles
    similarity_scores = cosine_similarity(
        query_vector,
        article_vectors
    )[0]

    # Get the indexes of the highest scores
    top_indices = similarity_scores.argsort()[::-1][:top_k]

    results = df.iloc[top_indices].copy()

    # Add similarity score
    results["score"] = similarity_scores[top_indices]

    return results


# Test question
question = "customer says claim was rejected but they had coverage"

results = tfidf_search(question)

print("Search question:")
print(question)

print("\nTop results:")

print(
    results[
        ["article_id", "title", "category", "score"]
    ].to_string(index=False)
)