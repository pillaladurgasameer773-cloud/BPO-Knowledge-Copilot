from pypdf import PdfReader


pdf_path = "../data/BPO_Knowledge_Articles_Gold-1.pdf"

reader = PdfReader(pdf_path)

print("Number of pages:", len(reader.pages))

print("\nPDF PAGE SUMMARY")
print("=" * 60)

for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text()

    if text:
        cleaned_text = " ".join(text.split())

        print(
            f"Page {page_number}: "
            f"{len(text)} characters"
        )

        print(
            "Preview:",
            cleaned_text[:250]
        )

    else:
        print(
            f"Page {page_number}: "
            "No text extracted"
        )

    print("-" * 60)