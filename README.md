BPO Knowledge Copilot 🤖

An AI-powered knowledge assistant that helps Business Process Outsourcing (BPO) agents find relevant knowledge articles and generate practical, source-grounded recommendations for customer queries.

🎯 Project Objective

BPO agents can spend valuable time searching knowledge articles when handling unfamiliar customer issues. BPO Knowledge Copilot combines hybrid information retrieval with Retrieval-Augmented Generation (RAG) to retrieve relevant guidance and provide concise recommendations supported by knowledge sources.

✨ Features

- PDF ingestion: Extract structured knowledge articles from PDF documents.
- Structure-aware chunking: Preserve useful sections such as scenarios, procedures, and escalation instructions.
- Semantic search: Use sentence embeddings to retrieve conceptually related knowledge.
- TF-IDF search: Match important terms in user questions.
- Hybrid retrieval: Combine semantic and lexical relevance scores.
- RAG pipeline: Supply retrieved knowledge to an LLM to generate grounded recommendations.
- Source transparency: Display retrieved article IDs and titles.
- Interactive application: Ask questions through a Streamlit interface.
- Retrieval evaluation: Compare TF-IDF, embedding-based, and hybrid retrieval.

🏗️ Architecture

Knowledge PDF
     ↓
PDF Text Extraction
     ↓
Article Identification
     ↓
Structure-Aware Chunking
     ↓
Sentence Embeddings + TF-IDF
     ↓
Hybrid Retrieval
     ↓
Relevant Knowledge Context
     ↓
LLM Answer Generation
     ↓
Recommendation + Source Articles
     ↓
Streamlit Interface

🛠️ Tech Stack

- Language: Python
- Data processing: Pandas, NumPy
- Information retrieval: Scikit-learn, TF-IDF, cosine similarity
- Embeddings: Sentence Transformers ("all-MiniLM-L6-v2")
- Deep learning framework: PyTorch
- PDF processing: pypdf
- LLM integration: OpenAI-compatible Python SDK with OpenRouter
- User interface: Streamlit

📊 Retrieval Evaluation

The retrieval pipeline was evaluated on 12 representative BPO questions with expected article IDs.

Retrieval approach| Top-1 accuracy| Recall@3
TF-IDF| 91.67%| 91.67%
Embedding-based retrieval| 100%| 100%
Hybrid retrieval (0.6 embedding / 0.4 TF-IDF)| 100%| 100%

Interpretation: On this evaluation set, embedding-based and hybrid retrieval both achieved 100% Top-1 accuracy and Recall@3.

These results apply only to the current 12-query dataset. They do not establish performance on unseen questions or production data. The selected hybrid weights are the current configuration, not proof of a universally optimal weighting.

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
├── .gitignore
├── README.md
└── requirements.txt

🚀 Run Locally

1. Clone the repository

git clone https://github.com/pillaladurgasameer773-cloud/BPO-Knowledge-Copilot.git
cd BPO-Knowledge-Copilot

2. Create and activate a virtual environment

On Windows PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1

3. Install dependencies

python -m pip install -r requirements.txt

4. Configure the OpenRouter API key

In Windows PowerShell, set your key for the current terminal session:

$env:OPENROUTER_API_KEY="YOUR_API_KEY"

Replace "YOUR_API_KEY" with your own key. Never commit API credentials to GitHub.

5. Prepare the generated knowledge files

The application requires the generated files "data/article_chunks.json" and "data/article_embeddings.npz". If they are not included in the repository, generate them using the project's ingestion, chunking, and embedding scripts in the appropriate order:

1. Inspect and extract the PDF articles.
2. Create structure-aware chunks.
3. Generate sentence embeddings.

The corresponding scripts are located in "src/". Run them from the project root using commands such as:

python src/12_pdf_to_articles.py
python src/13_structure_aware_chunking.py
python src/14_create_embeddings.py

Check that each script's output paths match the expected files under "data/" before starting the app.

6. Start the application

python -m streamlit run app/app.py

Open the local URL displayed in the terminal, usually "http://localhost:8501".

🔍 Example Use Case

Question: A customer says their insurance claim was rejected even though they had active coverage. What should the agent do?

The application:

1. Retrieves relevant knowledge articles.
2. Ranks articles using hybrid retrieval.
3. Passes the retrieved context to the LLM.
4. Generates a practical recommendation grounded in that context.
5. Displays source articles so the agent can verify the guidance.

🔐 Responsible Use

- Recommendations depend on the accuracy and completeness of the knowledge base.
- Agents should verify source articles and follow approved organizational procedures.
- The system should not invent policy details or claim to have performed actions it has not performed.
- API credentials must remain private.
- The application is a decision-support prototype, not a replacement for approved business procedures.

🚧 Limitations and Future Improvements

- Expand the evaluation dataset with more diverse questions and independently verified expected answers.
- Test retrieval against unseen documents and queries.
- Add automated tests for retrieval quality and answer grounding.
- Improve handling of API errors, rate limits, and model availability.
- Deploy the application and monitor its performance.

👨‍💻 Project Summary

BPO Knowledge Copilot demonstrates document ingestion, structure-aware chunking, sentence embeddings, information retrieval, hybrid search, RAG, retrieval evaluation, and interactive application development for a practical customer-support use case