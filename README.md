# ⚖️ Faithfulness-Aware Legal RAG

> **Automatic Detection and Mitigation of Contextual Hallucinations in the Legal Domain**
> M.Tech Thesis Project — CSE (AI Specialization)

---

## 📌 Overview

A Graph-RAG system for the Indian legal domain that goes beyond standard answer generation by detecting and mitigating hallucinations at the **span level**. The system retrieves structured legal context, generates answers, identifies unsupported claims, and self-corrects unfaithful generations.

---

## 🧠 Core Contributions

- **GraphRAG over Legal Corpus** — Hierarchical chunking + NetworkX knowledge graph for structured Indian legal documents
- **Span-Level Hallucination Detection** — Three-state tagging: 🟢 Supported / 🟡 Uncertain / 🔴 Hallucinated
- **Faithfulness Scoring** — Faithfulness Score, Unsupported Claim Score, Evidence Coverage Score
- **Self-Correction Feedback Loop** — Targeted re-retrieval and regeneration of hallucinated spans

---

## 🏗️ System Architecture

```
PDF Documents → [PyMuPDF Parser] → [Hierarchical Chunker]
                                        ↓               ↓
                              [InLegalBERT           [Entity Extractor]
                               Embeddings]                  ↓
                                    ↓               [NetworkX Graph]
                              [ChromaDB]
                                    ↓
                         User Query → [Dual Retriever]
                                           ↓
                                   [Groq Llama 3]
                                           ↓
                            [Span Hallucination Detector]
                                           ↓
                               [Faithfulness Scorer]
                                           ↓
                               [Self-Correction Module]
                                           ↓
                               [Streamlit Frontend]
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| LLM | Groq API (Llama 3 — free tier) |
| Embeddings | InLegalBERT (IIT Kharagpur) |
| Vector DB | ChromaDB (local) |
| Knowledge Graph | NetworkX |
| RAG Framework | LangChain |
| Backend | FastAPI |
| Frontend | Streamlit |
| PDF Parsing | PyMuPDF |

---

## 📁 Project Structure

```
faithfulness-aware-legal-rag/
├── data/
│   ├── raw/                  # Raw PDF files (not tracked by git)
│   └── processed/            # Chunked & processed docs (not tracked by git)
├── src/
│   ├── ingestion/            # PDF parsing & hierarchical chunking
│   ├── graph/                # Entity extraction & graph construction
│   ├── retrieval/            # Embeddings, ChromaDB, graph retriever
│   ├── generation/           # LLM answer generation
│   ├── hallucination/        # Span detector & faithfulness scorer
│   ├── correction/           # Self-correction module
│   └── utils/                # Shared utilities
├── api/                      # FastAPI backend
├── ui/                       # Streamlit frontend
├── tests/                    # Unit tests
├── config/
│   └── config.yaml           # All configurations
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/faithfulness-aware-legal-rag.git
cd faithfulness-aware-legal-rag
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY and HUGGINGFACE_TOKEN
```

### 5. Add Legal Documents
Place your PDF files in `data/raw/`:
- `data/raw/constitution_of_india.pdf`
- `data/raw/indian_penal_code.pdf`

### 6. Run the Application
```bash
# Start the FastAPI backend
uvicorn api.main:app --reload

# In a new terminal, start the Streamlit UI
streamlit run ui/app.py
```

---

## 📊 Evaluation Datasets

- **Knowledge Base:** Indian Constitution + IPC (Official PDFs — Ministry of Law)
- **Evaluation:** OpenNyAI InJudgements Dataset (Indian Supreme Court judgments 1950–2017)

---

## 📄 Base Paper

> S.M. Wahidur et al., *"Legal Query RAG (LQ-RAG)"*, IEEE Access, 2025

This project extends LQ-RAG with span-level hallucination detection, faithfulness scoring, and a self-correction feedback loop.

---

## 📜 License

MIT License — For academic and research use.
