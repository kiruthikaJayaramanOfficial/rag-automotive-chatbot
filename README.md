# 🚗 Automotive & Laptop Manual Assistant

A RAG (Retrieval-Augmented Generation) chatbot that answers questions from vehicle and laptop manuals with citations.

## 🔗 Live Demo
[Click here to try the app](https://rag-automotive-chatbot-nhg457n8xhzm4tonlvay4b.streamlit.app)

## 🛠 Tech Stack
- **LLM:** LLaMA 3.1 8B via Groq API (free)
- **Embeddings:** all-MiniLM-L6-v2 (HuggingFace)
- **Vector DB:** FAISS
- **Framework:** LangChain + Streamlit
- **Deployment:** Streamlit Cloud

## 📚 Dataset
- Toyota Fortuner 2025 Manual (460 pages)
- Toyota Innova Crysta 2024 Manual (560 pages)
- Dell Inspiron 15 3000 Manual (23 pages)
- HP Laptop Manual (70 pages)
- Lenovo ThinkPad X250 Manual (177 pages)
- **Total: 1,290 pages | 3,977 chunks**

## 📊 Evaluation
- Questions tested: 20
- Questions answered: 20/20
- ROUGE-L Score: 0.066

## 🚀 Run Locally
```bash
git clone https://github.com/kiruthikaJayaramanOfficial/rag-automotive-chatbot
cd rag-automotive-chatbot
python3 -m venv rag_env
source rag_env/bin/activate
pip install -r requirements.txt
python3 src/ingest.py
streamlit run app/streamlit_app.py
```

## 📁 Project Structure
```
rag-automotive/
├── data/raw/         # PDF manuals
├── data/faiss_index/ # Vector index
├── src/ingest.py     # PDF processing
├── src/rag_chain.py  # RAG + LLM logic
├── app/streamlit_app.py  # UI
└── eval/             # Evaluation scripts
```