import os
import json
import numpy as np
import pandas as pd

from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. PROJECT PATHS
# ============================================================

SRC_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    SRC_DIR
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data"
)


# ============================================================
# 2. OPENROUTER / LLM CONNECTION
# ============================================================

api_key = os.environ.get(
    "OPENROUTER_API_KEY"
)

if not api_key:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set."
    )

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)


# ============================================================
# 3. LOAD KNOWLEDGE CHUNKS
# ============================================================

chunks_file = os.path.join(
    DATA_DIR,
    "article_chunks.json"
)

with open(
    chunks_file,
    "r",
    encoding="utf-8"
) as file:

    chunks = json.load(file)


chunks_df = pd.DataFrame(chunks)


# ============================================================
# 4. LOAD EMBEDDING MODEL
# ============================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# 5. LOAD SAVED EMBEDDINGS
# ============================================================

embeddings_file = os.path.join(
    DATA_DIR,
    "article_embeddings.npz"
)

embedding_data = np.load(
    embeddings_file
)

article_embeddings = embedding_data[
    "embeddings"
]


# ============================================================
# 6. PREPARE TF-IDF
# ============================================================

chunks_df["search_text"] = (
    chunks_df["title"].fillna("")
    + " "
    + chunks_df["text"].fillna("")
)

tfidf_vectorizer = TfidfVectorizer(
    stop_words="english"
)

tfidf_vectors = tfidf_vectorizer.fit_transform(
    chunks_df["search_text"]
)


# ============================================================
# 7. HYBRID RETRIEVAL
# ============================================================

def hybrid_retrieve(
    query,
    top_k=3
):

    # TF-IDF
    query_tfidf = (
        tfidf_vectorizer.transform(
            [query]
        )
    )

    tfidf_scores = cosine_similarity(
        query_tfidf,
        tfidf_vectors
    )[0]


    # Embeddings
    query_embedding = (
        embedding_model.encode(
            [query],
            normalize_embeddings=True
        )
    )

    embedding_scores = cosine_similarity(
        query_embedding,
        article_embeddings
    )[0]


    # Normalize TF-IDF
    tfidf_min = tfidf_scores.min()
    tfidf_max = tfidf_scores.max()

    if tfidf_max > tfidf_min:

        tfidf_normalized = (
            (tfidf_scores - tfidf_min)
            / (tfidf_max - tfidf_min)
        )

    else:

        tfidf_normalized = np.zeros_like(
            tfidf_scores
        )


    # Normalize embeddings
    embedding_min = embedding_scores.min()
    embedding_max = embedding_scores.max()

    if embedding_max > embedding_min:

        embedding_normalized = (
            (embedding_scores - embedding_min)
            / (embedding_max - embedding_min)
        )

    else:

        embedding_normalized = np.zeros_like(
            embedding_scores
        )


    # Hybrid score
    hybrid_scores = (
        0.6 * embedding_normalized
        + 0.4 * tfidf_normalized
    )


    # Top results
    top_indices = np.argsort(
        hybrid_scores
    )[::-1][:top_k]


    results = chunks_df.iloc[
        top_indices
    ].copy()


    results["tfidf_score"] = (
        tfidf_scores[top_indices]
    )

    results["embedding_score"] = (
        embedding_scores[top_indices]
    )

    results["hybrid_score"] = (
        hybrid_scores[top_indices]
    )


    return results


# ============================================================
# 8. BUILD CONTEXT
# ============================================================

def build_context(results):

    context_parts = []


    for _, row in results.iterrows():

        article_id = row.get(
            "article_id",
            "Unknown"
        )

        title = row.get(
            "title",
            "Unknown"
        )

        page = row.get(
            "page",
            row.get(
                "page_number",
                "Unknown"
            )
        )

        text = row.get(
            "text",
            ""
        )


        source = (
            f"{article_id} - {title}"
        )


        context = (
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Knowledge:\n{text}"
        )


        context_parts.append(
            context
        )


    return "\n\n---\n\n".join(
        context_parts
    )


# ============================================================
# 9. GENERATE GROUNDED ANSWER
# ============================================================

def generate_answer(question, context):
    system_prompt = """
You are a BPO Knowledge Copilot.

Your job is to help BPO agents answer customer issues using ONLY the knowledge provided in the context.

Rules:
1. Use only the provided knowledge.
2. Do not invent procedures, policies, or facts.
3. If the knowledge does not contain enough information, clearly say so.
4. Give a concise and practical answer for a BPO agent.
5. Mention important escalation requirements when present.
6. At the end, provide the source article ID and title.
7. Do not claim that you performed an action.
"""

    user_prompt = f"""
KNOWLEDGE BASE CONTEXT:
{context}

CUSTOMER / AGENT QUESTION:
{question}

Provide the recommended action based only on the knowledge above.
"""

    try:
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.2
        )

        if response is None:
            return "The LLM returned no response. Please try again."

        if not getattr(response, "choices", None):
            return "The LLM returned no usable answer. The model may be temporarily unavailable. Please try again."

        message = response.choices[0].message

        if message is None:
            return "The LLM returned an empty message. Please try again."

        content = getattr(message, "content", None)

        if not content:
            return "The LLM returned an empty answer. Please try again."

        return content

    except Exception as e:
        return f"LLM request failed: {str(e)}"