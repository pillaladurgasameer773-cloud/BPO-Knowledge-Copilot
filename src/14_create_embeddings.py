import json
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. CONFIGURATION
# ============================================================

input_path = "../data/article_chunks.json"
output_path = "../data/article_embeddings.npz"

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# 2. LOAD CHUNKS
# ============================================================

print("\n" + "=" * 70)
print("EMBEDDING GENERATION")
print("=" * 70)

print("\nLoading article chunks...")

with open(
    input_path,
    "r",
    encoding="utf-8"
) as file:

    chunks = json.load(file)

print(
    "Chunks loaded:",
    len(chunks)
)


# ============================================================
# 3. LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print(
    "Model loaded:",
    MODEL_NAME
)


# ============================================================
# 4. PREPARE TEXT
# ============================================================

texts = []

for chunk in chunks:

    # Combine title and article content.
    #
    # Including the title helps the model understand
    # what the chunk is about.

    text = (
        chunk["title"]
        + "\n"
        + chunk["text"]
    )

    texts.append(text)


print(
    "\nTexts prepared:",
    len(texts)
)


# ============================================================
# 5. GENERATE EMBEDDINGS
# ============================================================

print("\nGenerating embeddings...")

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)


# ============================================================
# 6. CONVERT TO NUMPY ARRAY
# ============================================================

embeddings = np.asarray(
    embeddings,
    dtype=np.float32
)


print(
    "\nEmbedding matrix shape:",
    embeddings.shape
)


# ============================================================
# 7. BASIC VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("EMBEDDING VALIDATION")
print("-" * 70)

print(
    "Number of chunks:",
    len(chunks)
)

print(
    "Number of embeddings:",
    len(embeddings)
)

print(
    "Embedding dimensions:",
    embeddings.shape[1]
)

print(
    "Data type:",
    embeddings.dtype
)


# ============================================================
# 8. CHECK NORMALIZATION
# ============================================================

norms = np.linalg.norm(
    embeddings,
    axis=1
)

print(
    "Average vector norm:",
    round(float(np.mean(norms)), 4)
)


# ============================================================
# 9. SAVE EMBEDDINGS
# ============================================================

np.savez_compressed(
    output_path,
    embeddings=embeddings
)


print("\n" + "=" * 70)

print(
    "Embeddings saved to:",
    output_path
)

print("=" * 70)


# ============================================================
# 10. FINAL VALIDATION
# ============================================================

if len(chunks) == len(embeddings):

    print(
        "\nSUCCESS: Every chunk has an embedding."
    )

else:

    print(
        "\nWARNING: Chunk and embedding counts do not match."
    )