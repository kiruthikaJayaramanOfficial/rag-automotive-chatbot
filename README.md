# 🔍 LangRAG Docs — Agentic RAG for Any PDF Document

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://langrag-docs-nhg457n8xhzm4tonlvay4b.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.9-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_RAG-purple)
![Groq](https://img.shields.io/badge/LLM-Groq_Free_Tier-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> Upload **any PDF**, ask questions in plain English, and photograph images to get instant cited answers — powered by a **self-correcting LangGraph agentic pipeline**.

No fixed documents. No vendor lock-in. Bring your own PDF — vehicle manuals, research papers, legal documents, product guides — and get accurate, cited answers instantly.

---

## 🔗 Live Demo

**[langrag-docs.streamlit.app](https://langrag-docs-gjycjc2trydjemb4vtabqh.streamlit.app/)**

---

## 📸 Screenshots

### Home — Clean upload interface
![Main Screen](screenshots/main.png)

### Auto Summary + 4 Suggested Questions on Upload
![Summary and Questions](screenshots/01_summary_questions.png)

### Multimodal Vision — Image Analyzed + Manual Cross-Referenced
![Image Analysis](screenshots/picture_analysis.png)

### LangGraph Agent Trace + Source Citations
![Agent Trace and Sources](screenshots/langraph_souce_expansion.png)

### Conversation Memory — Follow-up Questions
![Memory](screenshots/memory.png)

### Analytics Dashboard
![Analytics](screenshots/analytics_dashboard.png)

---

## 🧠 LangGraph Agentic Pipeline

```
User Input (text or image)
         │
    ┌────▼──────┐
    │  ROUTER   │── text ───────────────────────────┐
    └────┬──────┘                                    │
         │ image                                     │
    ┌────▼──────────┐                   ┌────────────▼────────────┐
    │ VISION AGENT  │                   │       RETRIEVAL         │
    │ Llama-4 Scout │── description ───►│  Hybrid BM25 + ChromaDB │
    │ 17B (Groq)    │                   └────────────┬────────────┘
    └───────────────┘                                │
                                        ┌────────────▼────────────┐
                                        │        GRADER           │
                                        │  LLM checks relevance   │
                                        └────────────┬────────────┘
                                                     │
                                        ┌────────────▼────────────┐
                                        │       GENERATOR         │
                                        │  Context + vision +     │
                                        │  memory → cited answer  │
                                        └─────────────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **LangGraph Agentic RAG** | 5-node pipeline — Router → Vision → Retrieval → Grader → Generator |
| 👁 **Multimodal Vision** | Upload any image → Llama-4 Scout 17B describes it → RAG finds relevant manual section |
| 🔍 **Hybrid Search** | BM25 keyword + ChromaDB vector — catches exact terms AND semantic meaning |
| 🧠 **Conversation Memory** | Last 3 turns in every prompt — natural follow-up questions work |
| 📋 **Auto Document Summary** | 2-3 sentence overview generated immediately on PDF upload |
| 💡 **Suggested Questions** | 4 LLM-generated clickable questions from your document content |
| 📊 **Confidence Scoring** | 🟢 ≥70% · 🟡 ≥40% · 🔴 <40% badge on every answer |
| ⚠️ **Graceful Not-Found** | Orange warning box instead of hallucination when answer is absent |
| 📁 **Multi-PDF + Filter** | Upload multiple PDFs, restrict answers to specific documents |
| ⬇️ **Export Conversation** | Download full chat with timestamps, confidence, citations as `.txt` |
| 📈 **Analytics Dashboard** | Query history, confidence stats, document index info |

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Agent Orchestration** | LangGraph (StateGraph) |
| **LLM** | LLaMA 3.1 8B via Groq API (free) |
| **Vision Model** | Llama-4 Scout 17B via Groq (free) |
| **Fallback Vision** | Google Gemini 1.5 Flash |
| **Embeddings** | all-MiniLM-L6-v2 (HuggingFace, local) |
| **Vector DB** | ChromaDB (in-memory) |
| **Keyword Search** | BM25Okapi (rank-bm25) |
| **PDF Parsing** | LangChain PyPDFLoader |
| **Frontend** | Streamlit |
| **Deployment** | Streamlit Cloud |

---

## 🧪 Evaluation (Toyota Fortuner 2025 Manual — Demo Dataset)

| Metric | Value |
|---|---|
| Test questions | 20 |
| Questions answered | 20 / 20 (100%) |
| ROUGE-L Score | 0.066 |
| Avg Session Confidence | ~80% |

> ROUGE-L is low because RAG returns detailed answers while ground truth is short. Qualitative accuracy is high — all 20 answers correctly cited the right pages.

---

## 🔬 Demo Scenarios (using Toyota Fortuner 2025 Manual)

**Scenario 1 — Image Analysis**
Upload manual + photo of dashboard showing engine warning light → *"What does this warning light mean?"*
→ Llama-4 Scout identifies **Malfunction Indicator Lamp** → retrieves page 388 → explains causes with action steps

**Scenario 2 — Specific Query with Citations**
*"What is the oil change interval?"*
→ Retrieves pages 308, 310, 429 → answers with 6 driving-condition variants

**Scenario 3 — Conversation Memory**
Ask *"What is the oil change interval?"* → follow up *"Is it the same for all variants?"*
→ Second answer references previous exchange, answers variant-specifically

**Scenario 4 — Not-Found Response**
Ask about information absent from the document
→ Orange box: *"Not found in document — try rephrasing or upload a more detailed manual"*

---

## 🚀 Run Locally

```bash
# Clone
git clone https://github.com/kiruthikaJayaramanOfficial/langrag-docs.git
cd langrag-docs

# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# API keys
echo 'GROQ_API_KEY=your_groq_key' > .env
echo 'GOOGLE_API_KEY=your_gemini_key' >> .env

# Run
streamlit run app/streamlit_app.py
```

Free API keys: [console.groq.com](https://console.groq.com) · [aistudio.google.com](https://aistudio.google.com)

---

## 📁 Project Structure

```
langrag-docs/
├── app/
│   └── streamlit_app.py      # LangGraph pipeline + Streamlit UI
├── src/
│   ├── ingest.py             # PDF ingestion pipeline
│   └── rag_chain.py          # Base RAG chain
├── data/
│   └── README.md             # Dataset documentation
├── eval/
│   ├── evaluate.py           # ROUGE-L evaluation script
│   ├── test_qa.json          # 20 ground-truth Q&A pairs
│   └── results.json          # Evaluation results
├── screenshots/              # App screenshots
├── requirements.txt
└── README.md
```

---

## 🎯 LangRAG Docs vs Standard RAG

| Standard RAG | LangRAG Docs |
|---|---|
| Fixed documents only | Upload **any** PDF dynamically |
| Text queries only | **Image + text** multimodal input |
| Single retrieval attempt | **Self-correcting** grader node |
| Vector search only | **Hybrid BM25 + vector** |
| No memory | **3-turn conversation** memory |
| Hallucination on missing info | **Graceful NOT_IN_DOCUMENT** |
| No transparency | **Full LangGraph trace** per answer |

---

## 👩‍💻 Author

**Kiruthika Jayaraman** · [@kiruthikaJayaramanOfficial](https://github.com/kiruthikaJayaramanOfficial)

---

## 📄 License

MIT License
