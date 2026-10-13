import json
import re
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

input_path = Path(__file__).resolve().parent.parent / "data" / "pdf_articles.json"
output_path = Path(__file__).resolve().parent.parent / "data" / "article_chunks.json"

# Approximate maximum characters per chunk.
# This is intentionally a starting point, not a permanent rule.
MAX_CHARS = 1200

# Small overlap helps preserve context between chunks.
OVERLAP_CHARS = 150


# ============================================================
# 2. LOAD EXTRACTED ARTICLES
# ============================================================

print("\n" + "=" * 70)
print("STRUCTURE-AWARE DOCUMENT CHUNKING")
print("=" * 70)

print("\nLoading extracted articles...")

with open(
    input_path,
    "r",
    encoding="utf-8"
) as file:

    articles = json.load(file)


print(
    "Articles loaded:",
    len(articles)
)


# ============================================================
# 3. SECTION DETECTION
# ============================================================

# Our PDF articles contain sections such as:
#
# Category:
# Scenario:
# Procedure:
# Escalation:
# Search aliases:
#
# We keep these logical sections together instead of
# randomly cutting text.


section_pattern = re.compile(
    r"(Category:|Scenario:|Procedure:|Escalation:|Search aliases:)",
    re.IGNORECASE
)


# ============================================================
# 4. SPLIT ARTICLE INTO LOGICAL SECTIONS
# ============================================================

def split_into_sections(text):

    matches = list(
        section_pattern.finditer(text)
    )

    if not matches:
        return [text.strip()]

    sections = []

    for i, match in enumerate(matches):

        start = match.start()

        if i + 1 < len(matches):

            end = matches[i + 1].start()

        else:

            end = len(text)

        section = text[start:end].strip()

        if section:
            sections.append(section)

    return sections


# ============================================================
# 5. CREATE CHUNKS
# ============================================================

def create_chunks(article):

    article_id = article["article_id"]
    title = article["title"]
    page = article["page"]
    text = article["text"]

    sections = split_into_sections(text)

    chunks = []

    current_sections = []
    current_length = 0

    for section in sections:

        section_length = len(section)

        # ----------------------------------------------------
        # If adding this section stays within the limit
        # ----------------------------------------------------

        if (
            current_sections
            and current_length + section_length
            > MAX_CHARS
        ):

            chunk_text = "\n".join(
                current_sections
            ).strip()

            chunks.append({
                "article_id": article_id,
                "title": title,
                "page": page,
                "chunk_id": f"{article_id}-CHUNK-{len(chunks) + 1:03d}",
                "text": chunk_text
            })

            # ------------------------------------------------
            # Keep a small overlap from previous chunk
            # ------------------------------------------------

            overlap_text = chunk_text[-OVERLAP_CHARS:]

            current_sections = [
                overlap_text
            ]

            current_length = len(
                overlap_text
            )

        current_sections.append(section)

        current_length += section_length


    # ========================================================
    # SAVE FINAL CHUNK
    # ========================================================

    if current_sections:

        chunk_text = "\n".join(
            current_sections
        ).strip()

        chunks.append({
            "article_id": article_id,
            "title": title,
            "page": page,
            "chunk_id": f"{article_id}-CHUNK-{len(chunks) + 1:03d}",
            "text": chunk_text
        })

    return chunks


# ============================================================
# 6. PROCESS ALL ARTICLES
# ============================================================

all_chunks = []

print("\nCreating chunks...")

for article in articles:

    article_chunks = create_chunks(article)

    all_chunks.extend(
        article_chunks
    )


# ============================================================
# 7. DISPLAY SUMMARY
# ============================================================

print("\n" + "-" * 70)
print("CHUNKING SUMMARY")
print("-" * 70)

print(
    "Original articles:",
    len(articles)
)

print(
    "Total chunks:",
    len(all_chunks)
)


# ============================================================
# 8. CHUNK SIZE STATISTICS
# ============================================================

chunk_lengths = [
    len(chunk["text"])
    for chunk in all_chunks
]

if chunk_lengths:

    print(
        "Smallest chunk:",
        min(chunk_lengths),
        "characters"
    )

    print(
        "Largest chunk:",
        max(chunk_lengths),
        "characters"
    )

    print(
        "Average chunk:",
        round(
            sum(chunk_lengths)
            / len(chunk_lengths),
            2
        ),
        "characters"
    )


# ============================================================
# 9. DISPLAY FIRST 5 CHUNKS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 5 CHUNKS")
print("=" * 70)

for chunk in all_chunks[:5]:

    print("\nChunk ID:")
    print(chunk["chunk_id"])

    print(
        "Article ID:",
        chunk["article_id"]
    )

    print(
        "Title:",
        chunk["title"]
    )

    print(
        "Page:",
        chunk["page"]
    )

    print(
        "Text:",
        chunk["text"][:500]
    )

    print("-" * 70)


# ============================================================
# 10. CHECK ARTICLE COVERAGE
# ============================================================

article_ids = {
    article["article_id"]
    for article in articles
}

chunk_article_ids = {
    chunk["article_id"]
    for chunk in all_chunks
}

missing_articles = (
    article_ids - chunk_article_ids
)

print("\n" + "=" * 70)
print("ARTICLE COVERAGE CHECK")
print("=" * 70)

if not missing_articles:

    print(
        "SUCCESS: Every article has at least one chunk."
    )

else:

    print(
        "WARNING: Missing articles:",
        sorted(missing_articles)
    )


# ============================================================
# 11. SAVE CHUNKS
# ============================================================

with open(
    output_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        all_chunks,
        file,
        indent=2,
        ensure_ascii=False
    )


print("\n" + "=" * 70)

print(
    "Chunks saved to:",
    output_path
)

print("=" * 70)


# ============================================================
# 12. FINAL STATUS
# ============================================================

if (
    len(articles) > 0
    and len(all_chunks) > 0
    and not missing_articles
):

    print(
        "\nCHUNKING COMPLETE - ALL ARTICLES COVERED"
    )

else:

    print(
        "\nCHUNKING COMPLETED WITH WARNINGS"
    )