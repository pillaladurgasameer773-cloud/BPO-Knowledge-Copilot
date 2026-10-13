from pypdf import PdfReader
import re
import json
from collections import Counter
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================
pdf_path = Path(__file__).resolve().parent.parent / "data" / "BPO_Knowledge_Articles_Gold-1.pdf"
output_path = Path(__file__).resolve().parent.parent / "data" / "pdf_articles.json"

EXPECTED_ARTICLES = 44


# ============================================================
# 2. LOAD PDF
# ============================================================

print("\n" + "=" * 70)
print("PDF KNOWLEDGE ARTICLE EXTRACTION")
print("=" * 70)

print("\nLoading PDF...")

reader = PdfReader(pdf_path)

print("PDF loaded successfully.")
print("Total pages:", len(reader.pages))


# ============================================================
# 3. EXTRACT TEXT FROM EVERY PAGE
# ============================================================

pages = []

print("\nExtracting page text...")

for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text() or ""

    pages.append({
        "page": page_number,
        "text": text
    })

    print(
        f"Page {page_number}: "
        f"{len(text)} characters"
    )


# ============================================================
# 4. ARTICLE ID PATTERN
# ============================================================

# IMPORTANT:
#
# Our article IDs have different prefixes:
#
# CLM-001
# ELG-001
# BIL-001
# ENR-001
# MEM-001
# ESC-001
# COM-001
# QA-001
#
# Therefore we use:
#
# [A-Z]+
#
# instead of:
#
# [A-Z]{3}
#
# This allows both:
# QA-001  -> 2 letters
# CLM-001 -> 3 letters
#
# ============================================================

article_pattern = re.compile(
    r"(?m)^([A-Z]+-\d{3})\s*[-–—]\s*(.+)$"
)


# ============================================================
# 5. FIND ARTICLES
# ============================================================

articles = []

print("\nFinding articles...")

for page_data in pages:

    page_number = page_data["page"]
    text = page_data["text"]

    matches = list(
        article_pattern.finditer(text)
    )

    if not matches:
        continue

    for i, match in enumerate(matches):

        # --------------------------------------------
        # Article ID
        # --------------------------------------------

        article_id = match.group(1).strip()

        # --------------------------------------------
        # Article title
        # --------------------------------------------

        title = match.group(2).strip()

        # --------------------------------------------
        # Start of article body
        # --------------------------------------------

        start = match.end()

        # --------------------------------------------
        # End of article body
        # --------------------------------------------

        if i + 1 < len(matches):

            end = matches[i + 1].start()

        else:

            end = len(text)

        # --------------------------------------------
        # Extract article body
        # --------------------------------------------

        article_text = text[start:end].strip()

        # --------------------------------------------
        # Clean excessive whitespace
        # --------------------------------------------

        article_text = re.sub(
            r"\s+",
            " ",
            article_text
        ).strip()

        # --------------------------------------------
        # Save article
        # --------------------------------------------

        articles.append({
            "article_id": article_id,
            "title": title,
            "page": page_number,
            "text": article_text
        })


# ============================================================
# 6. SORT ARTICLES
# ============================================================

articles.sort(
    key=lambda x: x["article_id"]
)


# ============================================================
# 7. DISPLAY ARTICLE COUNT
# ============================================================

print("\n" + "-" * 70)

print(
    "Total articles found:",
    len(articles)
)

print("-" * 70)


# ============================================================
# 8. DISPLAY FIRST 5 ARTICLES
# ============================================================

print("\nFIRST 5 ARTICLES")
print("-" * 70)

for article in articles[:5]:

    print(
        "\nArticle ID:",
        article["article_id"]
    )

    print(
        "Title:",
        article["title"]
    )

    print(
        "Page:",
        article["page"]
    )

    print(
        "Text preview:",
        article["text"][:250]
    )

    print("-" * 70)


# ============================================================
# 9. DISPLAY ALL ARTICLE IDs
# ============================================================

article_ids = [
    article["article_id"]
    for article in articles
]

print("\nARTICLE IDs FOUND")
print("-" * 70)

print(article_ids)


# ============================================================
# 10. CHECK FOR DUPLICATE ARTICLE IDs
# ============================================================

id_counts = Counter(article_ids)

duplicate_ids = [
    article_id
    for article_id, count in id_counts.items()
    if count > 1
]

print("\nDUPLICATE ARTICLE IDs")
print("-" * 70)

if duplicate_ids:

    print(duplicate_ids)

else:

    print("None")


# ============================================================
# 11. CHECK QA ARTICLES
# ============================================================

qa_articles = [
    article_id
    for article_id in article_ids
    if article_id.startswith("QA-")
]

print("\nQA ARTICLES CHECK")
print("-" * 70)

print(
    "QA articles found:",
    qa_articles
)


# ============================================================
# 12. CHECK EXPECTED ARTICLE COUNT
# ============================================================

print("\n" + "=" * 70)
print("EXTRACTION VALIDATION")
print("=" * 70)

if len(articles) == EXPECTED_ARTICLES:

    print(
        f"SUCCESS: Found all {EXPECTED_ARTICLES} articles."
    )

else:

    print(
        f"WARNING: Expected {EXPECTED_ARTICLES} articles."
    )

    print(
        f"Found: {len(articles)} articles."
    )


# ============================================================
# 13. CHECK REQUIRED ARTICLE PREFIXES
# ============================================================

expected_prefixes = [
    "CLM-",
    "ELG-",
    "BIL-",
    "ENR-",
    "MEM-",
    "ESC-",
    "COM-",
    "QA-"
]

print("\nARTICLE CATEGORY CHECK")
print("-" * 70)

for prefix in expected_prefixes:

    count = sum(
        1
        for article_id in article_ids
        if article_id.startswith(prefix)
    )

    print(
        f"{prefix:<5} -> {count} articles"
    )


# ============================================================
# 14. SAVE STRUCTURED JSON
# ============================================================

with open(
    output_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        articles,
        file,
        indent=2,
        ensure_ascii=False
    )


print("\n" + "=" * 70)

print(
    "Saved to:",
    output_path
)

print("=" * 70)


# ============================================================
# 15. FINAL STATUS
# ============================================================

if (
    len(articles) == EXPECTED_ARTICLES
    and not duplicate_ids
    and len(qa_articles) == 4
):

    print("\nEXTRACTION COMPLETE - ALL CHECKS PASSED")

else:

    print("\nEXTRACTION COMPLETE - CHECK WARNINGS ABOVE")