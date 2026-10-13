BPO Knowledge Copilot 🤖

An AI-powered knowledge assistant designed to help Business Process Outsourcing (BPO) agents find relevant knowledge articles and generate practical, source-grounded recommendations for customer queries.

🎯 Project Objective

BPO agents often spend time searching knowledge articles to resolve unfamiliar customer issues. This project combines hybrid information retrieval and a Large Language Model (LLM) to help agents find relevant guidance more efficiently.

✨ Features

- PDF ingestion: Extract structured knowledge articles from PDF documents.
- Structure-aware chunking: Preserve useful article sections such as scenarios, procedures, and escalation instructions.
- Semantic search: Use sentence embeddings to find meaningfully related knowledge.
- TF-IDF search: Match important terms from the user's question.
- Hybrid retrieval: Combine semantic and lexical search scores.
- RAG pipeline: Pass retrieved knowledge to an LLM to generate grounded recommendations.
- Source transparency: Display retrieved article IDs and titles.
- Interactive interface: Ask questions through a Streamlit web application.
- Retrieval evaluation: Compare TF-IDF, embedding-based, and hybrid retrieval approaches.

🏗️ Architecture

Knowledge PDF
     ↓
PDF Text Extraction
     ↓
Article Identification
     ↓
Structure-Aware Chunking
     ↓
Sentence Embeddings
     ↓
Hybrid Retrieval
(TF-IDF + Semantic Search)
     ↓
Relevant Knowledge Context
     ↓
Large Language Model
     ↓
Grounded Recommendation + Sources
     ↓
Streamlit Interface

🛠️ Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Sentence Transformers
- PyTorch
- OpenAI-compatible SDK
- OpenRouter API
- Streamlit
- PyPDF

📊 Retrieval Evaluation

The retrieval pipeline was evaluated using a small dataset of 12 representative BPO questions with expected article IDs.

Retrieval approach| Top-1 accuracy| Recall@3
TF-IDF| 91.67%| 91.67%
Embedding-based retrieval| 100%| 100%
Hybrid retrieval (0.6 embedding / 0.4 TF-IDF)| 100%| 100%

Important: These results apply only to the current 12-query evaluation set. They do not establish performance on every possible customer question or on unseen production data.

📁 Project Structure

BPO-Knowledge-Copilot/
├── app/
│   └── app.py
├── data/
│   ├── BPO_Knowledge_Articles_Gold-1.pdf
│   ├── BPO_Knowledge_Copilot_Gold.csv
│   ├── pdf_articles.json
│   ├── article_chunks.json
│   └── article_embeddings.npz
├── src/
│   ├── 11_pdf_inspection.py
│   ├── 12_pdf_to_articles.py
│   ├── 13_structure_aware_chunking.py
│   ├── 14_create_embeddings.py
│   ├── 15_semantic_search.py
│   ├── 16_retrieval_evaluation.py
│   ├── 17_hybrid_retrieval.py
│   ├── 18_hybrid_evaluation.py
│   └── rag.py
├── tests/
├── README.md
└── requirements.txt

🚀 Run Locally

1. Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL
cd BPO-Knowledge-Copilot

2. Create and activate a virtual environment

Windows PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1

3. Install dependencies

python -m pip install -r requirements.txt

4. Configure the API key

Set your OpenRouter API key in the environment. In Windows PowerShell:

$env:OPENROUTER_API_KEY="YOUR_API_KEY"

Replace the placeholder with your own key. Never commit API keys to GitHub.

5. Start the application

python -m streamlit run app/app.py

Open the local URL shown in the terminal, usually "http://localhost:8501".

🔍 Example Use Case

Question: A customer says their insurance claim was rejected even though they had active coverage. What should the agent do?

Expected workflow:

1. Retrieve relevant knowledge articles.
2. Select the most relevant guidance.
3. Provide the retrieved context to the LLM.
4. Generate practical recommendations based on the knowledge.
5. Display the source article so the agent can verify the guidance.

🔐 Responsible Use

- Recommendations depend on the accuracy and completeness of the knowledge base.
- Agents should verify the source article and follow their organization's approved procedures.
- The system should not invent policy details or claim that it completed actions it has not performed.
- API credentials must be kept private.

📌 Limitations and Future Improvements

- Expand the evaluation set with more diverse questions and independently verified expected answers.
- Test retrieval on unseen documents and queries.
- Add automated tests for retrieval and answer grounding.
- Improve error handling and API availability management.
- Explore deployment and performance monitoring.

👨‍💻 Project Summary

This project demonstrates document processing, text embeddings, information retrieval, hybrid search, retrieval-augmented generation, evaluation, and interactive application development in a practical customer-support use case.