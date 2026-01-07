# local-rag-document-assistant
Companies rarely upload sensitive financial reports, legal contracts, or HR data to public AI APIs (like ChatGPT) due to data privacy concerns.

The Solution: I built a Local Retrieval-Augmented Generation (RAG) application. It runs entirely on the user's laptop using **Ollama (Llama 3) for inference and HuggingFace for embeddings. No data ever leaves the local machine.

Key Features:
Privacy First:Runs offline using local LLMs.
RAG Architecture:Retrieves exact context from uploaded PDFs to prevent "hallucinations."
Vector Search:Uses ChromaDB for semantic search and efficient retrieval.
Zero Cost: Replaces expensive OpenAI API calls with open-source models.

---

Architecture
Data Flow:`PDF Input` → `Chunking` → `Vector Embeddings` → `ChromaDB` → `Llama 3 Inference`

1.  Ingestion: The user uploads a PDF via Streamlit.
2.  Processing: The text is split into chunks of 1000 characters using `RecursiveCharacterTextSplitter`.
3.  Embedding: Text chunks are converted into vector representations using a local **HuggingFace model** (`all-MiniLM-L6-v2`).
4.  Storage: Vectors are stored in **ChromaDB** (in-memory).
5.  Retrieval & Generation: When a user asks a question, the system finds the most relevant chunks and feeds them to **Llama 3**, which generates a precise answer based *only* on the document.

---

Technical Stack
LLM Engine: Ollama(https://ollama.com/) (Running Meta Llama 3)
Orchestration: LangChain
Vector Database: ChromaDB
Embeddings: HuggingFace (`sentence-transformers`)
Frontend: Streamlit

---
How to Run This Project
Prerequisites
You need Ollama installed to run the AI model locally.
1.  Download and install Ollama(https://ollama.com/).
2.  Open your terminal and download the model:
    ollama pull llama3

1. Clone the Repository
git clone https://github.com/yourusername/local-rag-document-assistant.git
cd local-rag-document-assistant
2. Install Dependencies
pip install -r requirements.txt
3. Run the App
python -m streamlit run app.py
The application will open automatically in your browser at http://localhost:8501
