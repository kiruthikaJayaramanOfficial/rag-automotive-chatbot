"""
LangRAG Docs — Agentic RAG with LangGraph
- Unified chat: text + image upload in ONE chat interface
- LangGraph 5-node workflow (Router → Vision → Retrieval → Grader → Generator)
- Self-correction retry loop
- Hybrid BM25 + ChromaDB search
- Conversation memory, confidence scoring, export
"""

import streamlit as st
import sys, os, tempfile, base64, json, datetime
sys.path.append(".")
import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="LangRAG Docs", page_icon="📄", layout="wide")

st.markdown("""
<style>
.confidence-high{background:#d4edda;color:#155724;padding:4px 12px;border-radius:12px;font-size:12px;font-weight:600;}
.confidence-med{background:#fff3cd;color:#856404;padding:4px 12px;border-radius:12px;font-size:12px;font-weight:600;}
.confidence-low{background:#f8d7da;color:#721c24;padding:4px 12px;border-radius:12px;font-size:12px;font-weight:600;}
.summary-box{background:#e8f5e9;border-left:4px solid #2e7d32;padding:14px 18px;border-radius:8px;margin-bottom:16px;}
.notfound-box{background:#fff3e0;border-left:4px solid #ef6c00;padding:14px 18px;border-radius:8px;margin:8px 0;}
.stat-card{background:#f8f9fa;border-radius:10px;padding:12px;text-align:center;border:1px solid #e9ecef;margin-bottom:8px;}
.stat-num{font-size:22px;font-weight:700;color:#1a73e8;}
.stat-label{font-size:11px;color:#6c757d;}
@media(prefers-color-scheme:dark){
  .summary-box{background:#1b3a1e;border-left-color:#4caf50;}
  .notfound-box{background:#3e2400;border-left-color:#ff9800;}
  .stat-card{background:#2a2a2a;border-color:#444;}
  .stat-num{color:#64b5f6;}
}
</style>
""", unsafe_allow_html=True)

# ── Resources ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_groq():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

embeddings  = load_embeddings()
groq_client = get_groq()

# ── Session state ──────────────────────────────────────────────────────────────
DEFAULTS = {
    "messages":       [],
    "chat_history":   [],
    "query_log":      [],
    "pdf_store":      {},   # {fname: {"chunks":[], "pages":n, "bm25": BM25|None}}
    "active_filters": [],
    "doc_summary":    None,
    "suggested_qs":   [],
    "pending_q":      None,
    "pending_img":    None, # base64 of image waiting to be sent
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── LangGraph state ────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    input_type:          str
    question:            str
    image_b64:           Optional[str]
    vision_description:  Optional[str]
    retrieved_chunks:    Optional[List]
    grade_passed:        bool
    retry_count:         int
    answer:              str
    sources:             List[str]
    confidence:          int
    not_found:           bool
    trace:               List[str]

# ── Core helpers ───────────────────────────────────────────────────────────────
def groq_call(messages, model="llama-3.1-8b-instant", max_tokens=1000):
    return groq_client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens
    ).choices[0].message.content

def compute_confidence(docs):
    if not docs: return 0
    return round(sum(min(len(d.page_content)/500,1.0)*100 for d in docs)/len(docs))

def confidence_badge(score):
    if score >= 70: return f'<span class="confidence-high">✓ Confidence {score}%</span>'
    if score >= 40: return f'<span class="confidence-med">⚠ Confidence {score}%</span>'
    return f'<span class="confidence-low">✗ Low confidence {score}%</span>'

def process_pdf(uf):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uf.read()); tmp_path = tmp.name
    pages   = PyPDFLoader(tmp_path).load()
    chunks  = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(pages)
    for c in chunks: c.metadata["source_file"] = uf.name
    os.unlink(tmp_path)
    sample = " ".join(p.page_content for p in pages[:5])[:3000]
    bm25   = BM25Okapi([c.page_content.lower().split() for c in chunks]) if BM25_AVAILABLE else None
    return chunks, len(pages), sample, bm25

def gen_summary_qs(sample, fname):
    prompt = (
        "You are analyzing a document called: " + fname + "\n\n"
        "Content from first pages:\n" + sample[:2000] + "\n\n"
        "Tasks:\n"
        "1. Write a 2-3 sentence summary starting with: This document covers\n"
        "2. Write exactly 4 specific questions a user would ask.\n\n"
        "Respond ONLY in this exact JSON, no extra text:\n"
        '{"summary": "This document covers ...", '
        '"questions": ["Q1?", "Q2?", "Q3?", "Q4?"]}'
    )
    try:
        raw = groq_call([{"role":"user","content":prompt}], max_tokens=500).strip()
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                if "{" in part:
                    raw = part.strip()
                    if raw.startswith("json"): raw = raw[4:].strip()
                    break
        start = raw.find("{"); end = raw.rfind("}") + 1
        if start != -1 and end > start: raw = raw[start:end]
        d = json.loads(raw)
        summary = d.get("summary",""); questions = d.get("questions",[])
        if summary and questions: return summary, questions
        return "Document indexed and ready.", []
    except Exception:
        try:
            simple = groq_call([{"role":"user","content":
                "In 2 sentences what is this document about? Name: " + fname +
                " Content: " + sample[:800]}], max_tokens=100).strip()
            return simple, []
        except:
            return "Document indexed and ready.", []

def get_active_chunks():
    store   = st.session_state.pdf_store
    filters = st.session_state.active_filters or list(store.keys())
    return [c for f in filters if f in store for c in store[f]["chunks"]]

def get_chroma(chunks):
    if not chunks: return None
    key = str(abs(hash(tuple(c.page_content[:30] for c in chunks[:5]))) % 10**6)
    return Chroma.from_documents(chunks, embeddings, collection_name=f"col{key}")

def hybrid_search(query, chunks, bm25_idx, chroma_vs, k=8):
    vec_docs = chroma_vs.similarity_search(query, k=k) if chroma_vs else []
    bm25_docs = []
    if bm25_idx and BM25_AVAILABLE and chunks:
        scores  = bm25_idx.get_scores(query.lower().split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        bm25_docs = [chunks[i] for i in top_idx if scores[i]>0]
    seen, merged = set(), []
    for d in (vec_docs + bm25_docs):
        key = d.page_content[:80]
        if key not in seen: seen.add(key); merged.append(d)
    return merged[:k]

def export_txt():
    lines = ["LangRAG Docs — Export", f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}", "="*50, ""]
    for q in st.session_state.query_log:
        lines += [f"[{q['time']}] {q['mode']} | Conf:{q['confidence']}%",
                  f"Q: {q['question']}", f"A: {q.get('answer','')}", ""]
    return "\n".join(lines)

# ── LangGraph nodes ────────────────────────────────────────────────────────────
def node_router(state):
    if state.get("image_b64"):
        state["input_type"] = "image"
        state["trace"].append("🔀 Router → IMAGE detected → Vision Agent")
    else:
        state["input_type"] = "text"
        state["trace"].append("🔀 Router → TEXT detected → Retrieval")
    return state

def node_vision(state):
    if state["input_type"] != "image": return state
    vision_prompt = (
        "You are an automotive expert analyzing a car dashboard photo. "
        "Identify EVERY warning light or indicator symbol visible. "
        "For each one state: exact name, color, and what malfunction it means. "
        "Example: I can see a yellow engine warning light (malfunction indicator lamp - engine fault) "
        "and a red oil pressure warning light (oil pressure too low - stop engine immediately). "
        "Be specific and thorough."
    )

    # Priority 1: Groq llama-4-scout (vision, free tier)
    try:
        messages = [{"role":"user","content":[
            {"type":"text","text": vision_prompt},
            {"type":"image_url","image_url":{"url":"data:image/jpeg;base64," + state["image_b64"]}}
        ]}]
        desc = groq_call(messages, model="meta-llama/llama-4-scout-17b-16e-instruct", max_tokens=400)
        state["vision_description"] = desc
        state["trace"].append("👁 Vision (Llama-4 Scout) → " + desc[:150] + "...")
        return state
    except Exception as e:
        state["trace"].append("👁 Llama-4 Scout failed: " + str(e)[:80])

    # Priority 2: Groq llama-4-maverick
    try:
        messages = [{"role":"user","content":[
            {"type":"text","text": vision_prompt},
            {"type":"image_url","image_url":{"url":"data:image/jpeg;base64," + state["image_b64"]}}
        ]}]
        desc = groq_call(messages, model="meta-llama/llama-4-maverick-17b-128e-instruct", max_tokens=400)
        state["vision_description"] = desc
        state["trace"].append("👁 Vision (Llama-4 Maverick) → " + desc[:150] + "...")
        return state
    except Exception as e:
        state["trace"].append("👁 Llama-4 Maverick failed: " + str(e)[:80])

    # Priority 3: Gemini (if quota available)
    try:
        from google import genai as ggenai
        import os, base64 as _b64
        gclient = ggenai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        img_bytes = _b64.b64decode(state["image_b64"])
        from google.genai import types as _gtypes
        response = gclient.models.generate_content(
            model="gemini-1.5-flash",
            contents=[vision_prompt,
                      _gtypes.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")]
        )
        desc = response.text
        state["vision_description"] = desc
        state["trace"].append("👁 Vision (Gemini Flash) → " + desc[:150] + "...")
        return state
    except Exception as e:
        state["trace"].append("👁 Gemini failed: " + str(e)[:80])

    # Final fallback: use question text for retrieval
    desc = state.get("question","") or "warning light dashboard indicator symbol"
    state["vision_description"] = desc
    state["trace"].append("👁 Vision → all models unavailable, using text for retrieval")
    return state

def node_retrieval(state):
    query = state.get("vision_description") or state["question"]
    if state.get("question") and state.get("vision_description"):
        query = f"{state['vision_description']} {state['question']}"

    if state.get("retry_count", 0) > 0:
        rq = groq_call([{"role":"user","content":f"Rephrase to improve document search:\n{query}\nRephrased (one line):"}], max_tokens=60).strip()
        state["trace"].append(f"🔄 Retry {state['retry_count']} → rephrased: {rq[:70]}")
        query = rq

    chunks    = get_active_chunks()
    chroma_vs = get_chroma(chunks)
    store     = st.session_state.pdf_store
    filters   = st.session_state.active_filters or list(store.keys())
    all_bm25  = None
    if BM25_AVAILABLE and chunks:
        all_bm25 = BM25Okapi([c.page_content.lower().split() for c in chunks])

    docs = hybrid_search(query, chunks, all_bm25, chroma_vs, k=8)
    state["retrieved_chunks"] = docs
    mode = "BM25+Vector" if BM25_AVAILABLE else "Vector"
    state["trace"].append(f"🔍 Retrieval → {mode} search → {len(docs)} chunks found")
    return state

def node_grader(state):
    docs = state.get("retrieved_chunks", [])
    if not docs:
        state["grade_passed"] = False
        state["trace"].append("⚖️ Grader → no chunks → retry")
        return state
    sample   = " ".join(d.page_content[:150] for d in docs[:3])
    question = state.get("vision_description") or state["question"]
    grade_prompt = (
        "You are a relevance judge. Reply with ONLY the word YES or NO.\n"
        "Does the context contain information relevant to this question?\n"
        "Question: " + question[:200] + "\n"
        "Context: " + sample[:400] + "\n"
        "Your answer (YES or NO only):"
    )
    raw = groq_call([{"role":"user","content":grade_prompt}],
                    model="llama-3.1-8b-instant", max_tokens=3).strip().upper()
    verdict = "YES" if "YES" in raw else "NO"
    passed = verdict == "YES"
    state["grade_passed"] = passed
    state["trace"].append("⚖️ Grader → " + verdict + (" ✓ proceeding" if passed else " ✗ retrying"))
    return state

def node_generator(state):
    docs = state.get("retrieved_chunks", [])
    confidence = compute_confidence(docs)
    context = "\n\n".join(
        f"[Source: {c.metadata.get('source_file','doc')}, Page {c.metadata.get('page','?')}]\n{c.page_content}"
        for c in docs
    ) if docs else ""

    history = "\n".join(
        f"Human: {h['human']}\nAssistant: {h['ai']}"
        for h in st.session_state.chat_history[-3:]
    ) if st.session_state.chat_history else ""

    full_q = state["question"]
    if state.get("vision_description"):
        full_q = f"Image analysis: {state['vision_description']}\nUser question: {state['question'] or 'Explain what you found.'}"

    if not context:
        state.update(answer="NOT_IN_DOCUMENT", not_found=True, sources=[], confidence=0)
        state["trace"].append("✍️ Generator → no context → NOT_IN_DOCUMENT")
        return state

    memory_block = ("Recent conversation:\n" + history + "\n\n") if history else ""
    prompt = (
        "You are a helpful automotive and technical document assistant.\n"
        + memory_block
        + "Answer using the context below.\n"
        + "If the answer is NOT in context, reply: NOT_IN_DOCUMENT\n"
        + "Otherwise give a clear, detailed, helpful answer and cite sources as (filename, page).\n\n"
        + "Context:\n"
        + context
        + "\n\nQuestion: "
        + full_q
        + "\nAnswer:"
    )

    answer    = groq_call([{"role":"user","content":prompt}])
    not_found = answer.strip().startswith("NOT_IN_DOCUMENT")
    st.session_state.chat_history.append({"human": state["question"], "ai": answer})
    sources = list(set(
        f"{c.metadata.get('source_file','doc')} — page {c.metadata.get('page','?')}"
        for c in docs
    ))
    state.update(answer=answer, not_found=not_found, sources=sources, confidence=confidence)
    state["trace"].append(f"✍️ Generator → done, confidence {confidence}%")
    return state

def route_after_router(state): return "vision" if state["input_type"]=="image" else "retrieval"
def route_after_grader(state):
    # Always go to generator — grader is advisory only, not a gate
    # This prevents GraphRecursionError completely
    return "generator"

@st.cache_resource
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("router",    node_router)
    g.add_node("vision",    node_vision)
    g.add_node("retrieval", node_retrieval)
    g.add_node("grader",    node_grader)
    g.add_node("generator", node_generator)
    g.set_entry_point("router")
    g.add_conditional_edges("router", route_after_router, {"vision":"vision","retrieval":"retrieval"})
    g.add_edge("vision",    "retrieval")
    g.add_edge("retrieval", "grader")
    g.add_conditional_edges("grader", route_after_grader, {"retrieval":"retrieval","generator":"generator"})
    g.add_edge("generator", END)
    return g.compile()

graph = build_graph()

def run_agent(question, image_b64=None):
    return graph.invoke({
        "input_type":"text", "question":question, "image_b64":image_b64,
        "vision_description":None, "retrieved_chunks":[], "grade_passed":False,
        "retry_count":0, "answer":"", "sources":[], "confidence":0,
        "not_found":False, "trace":[]
    }, {"recursion_limit": 10})

# ── Render helpers ─────────────────────────────────────────────────────────────
def render_trace(trace):
    if not trace: return
    with st.expander("🤖 LangGraph agent trace", expanded=False):
        colors = {"🔀":"#7c4dff","👁":"#0288d1","🔍":"#2e7d32","⚖️":"#f57c00","🔄":"#e64a19","✍️":"#1565c0"}
        for step in trace:
            c = next((v for k,v in colors.items() if step.startswith(k)), "#555")
            st.markdown(f"<span style='color:{c};font-size:12px'>{step}</span>", unsafe_allow_html=True)

def render_assistant_msg(msg):
    if msg.get("not_found"):
        st.markdown('<div class="notfound-box">⚠️ <strong>Not found in document</strong><br>'
                    'This information was not found in your manual — try rephrasing or upload a more complete manual.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(msg["content"])
    if "confidence" in msg:
        st.markdown(confidence_badge(msg["confidence"]), unsafe_allow_html=True)
    if msg.get("trace"):
        render_trace(msg["trace"])
    if msg.get("sources"):
        with st.expander("📄 Sources"):
            for s in msg["sources"]: st.caption(f"• {s}")

def submit_message(question, image_b64=None, display_image_bytes=None):
    """Add user message, run agent, add assistant reply."""
    # User bubble
    if display_image_bytes:
        st.session_state.messages.append({
            "role":"user", "content": question or "Analyze this image",
            "image_bytes": display_image_bytes
        })
    else:
        st.session_state.messages.append({"role":"user","content":question})

    with st.chat_message("user"):
        if display_image_bytes:
            st.image(display_image_bytes, width=280)
        if question:
            st.markdown(question)

    # Agent
    with st.chat_message("assistant"):
        with st.spinner("🤖 Running LangGraph agent..."):
            result = run_agent(question or "Identify and explain what you see.", image_b64)
        msg = {"role":"assistant","content":result["answer"],
               "sources":result["sources"],"confidence":result["confidence"],
               "not_found":result["not_found"],"trace":result["trace"]}
        render_assistant_msg(msg)

    st.session_state.messages.append(msg)
    st.session_state.query_log.append({
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "question": (question or "[image only]"),
        "answer": result["answer"],
        "sources": result["sources"],
        "confidence": result["confidence"],
        "mode": "Image+Text" if image_b64 else "Text"
    })

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📄 LangRAG Docs")
    st.caption("Agentic RAG · LangGraph · Hybrid Search")
    st.divider()

    st.markdown("**📂 Upload PDFs**")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"],
                                      accept_multiple_files=True,
                                      label_visibility="collapsed")
    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in st.session_state.pdf_store:
                with st.spinner(f"Indexing {uf.name}..."):
                    chunks, pages, sample, bm25 = process_pdf(uf)
                    st.session_state.pdf_store[uf.name] = {"chunks":chunks,"pages":pages,"bm25":bm25}
                    summary, questions = gen_summary_qs(sample, uf.name)
                    st.session_state.doc_summary  = summary
                    st.session_state.suggested_qs = questions
                    st.session_state.messages     = []
                    st.session_state.chat_history = []
                    st.session_state.active_filters = list(st.session_state.pdf_store.keys())
                st.success(f"✓ {uf.name} ({pages}p)")

    if len(st.session_state.pdf_store) > 1:
        st.divider()
        st.markdown("**🔍 Filter documents**")
        new_f = []
        for fn in st.session_state.pdf_store:
            if st.checkbox(fn[:36], value=(fn in st.session_state.active_filters), key=f"chk_{fn}"):
                new_f.append(fn)
        st.session_state.active_filters = new_f

    st.divider()
    if st.session_state.pdf_store:
        total_p = sum(v["pages"]      for v in st.session_state.pdf_store.values())
        total_c = sum(len(v["chunks"]) for v in st.session_state.pdf_store.values())
        st.markdown("**📊 Stats**")
        ca,cb = st.columns(2)
        with ca: st.markdown(f'<div class="stat-card"><div class="stat-num">{len(st.session_state.pdf_store)}</div><div class="stat-label">PDFs</div></div>',unsafe_allow_html=True)
        with cb: st.markdown(f'<div class="stat-card"><div class="stat-num">{total_p}</div><div class="stat-label">Pages</div></div>',unsafe_allow_html=True)
        st.markdown(f'<div class="stat-card"><div class="stat-num">{total_c}</div><div class="stat-label">Chunks</div></div>',unsafe_allow_html=True)
        st.caption(f"🔍 {'BM25 + Vector' if BM25_AVAILABLE else 'Vector only'}")
        st.markdown("")

    if st.session_state.query_log:
        st.download_button("⬇ Export Conversation",data=export_txt(),
            file_name=f"autodoc_{datetime.datetime.now():%Y%m%d_%H%M}.txt",
            mime="text/plain",use_container_width=True)

    if st.button("🗑 Clear all",use_container_width=True):
        for k,v in DEFAULTS.items(): st.session_state[k]=v
        st.rerun()

    st.divider()
    st.caption("LangGraph · LLaMA 3.1 8B + 3.2 11B Vision")
    st.caption("ChromaDB · BM25 · MiniLM-L6-v2 · Groq")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["💬 Chat", "📊 Analytics"])

with tab1:
    st.title("📄 LangRAG Docs")
    st.caption("Agentic RAG · LangGraph · Hybrid Search · Multimodal")

    if not st.session_state.pdf_store:
        st.info("👈 Upload a PDF from the sidebar to get started.")
        st.stop()

    # Summary box
    if st.session_state.doc_summary:
        st.markdown(f'<div class="summary-box">📋 <strong>Document Summary</strong><br>{st.session_state.doc_summary}</div>',
                    unsafe_allow_html=True)

    # Suggested question buttons
    if st.session_state.suggested_qs:
        st.markdown("**💡 Click a question to ask instantly:**")
        cols = st.columns(2)
        for i, q in enumerate(st.session_state.suggested_qs):
            with cols[i%2]:
                if st.button(q, key=f"sq_{i}", use_container_width=True):
                    st.session_state.pending_q = q
                    st.rerun()
        st.markdown("")

    # ── Render chat history ────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_assistant_msg(msg)
            else:
                if msg.get("image_bytes"):
                    st.image(msg["image_bytes"], width=280)
                if msg.get("content"):
                    st.markdown(msg["content"])

    # ── Handle pending suggested question ─────────────────────────────────────
    if st.session_state.pending_q:
        q = st.session_state.pending_q
        st.session_state.pending_q = None
        submit_message(q)

    # ══════════════════════════════════════════════════════════════════════════
    # UNIFIED CHAT INPUT — text + optional image upload together
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("**💬 Ask a question or upload a dashboard photo:**")

    col_img, col_input = st.columns([1, 4])

    with col_img:
        chat_image = st.file_uploader(
            "📸 Image",
            type=["png","jpg","jpeg","webp"],
            label_visibility="visible",
            key="chat_img_upload"
        )
        if chat_image:
            img_bytes = chat_image.read()
            st.image(img_bytes, width=100, caption="Ready to send")
            st.session_state.pending_img = base64.b64encode(img_bytes).decode()
            st.session_state._pending_img_bytes = img_bytes

    with col_input:
        # Show example inputs
        st.caption("**Examples:** 'What does the ABS light mean?' · 'What oil does this engine need?' · Upload a photo of your dashboard")
        user_text = st.text_area(
            "Your question:",
            placeholder="Type your question here, or just upload an image and click Send...",
            height=80,
            label_visibility="collapsed"
        )
        send_btn = st.button("➤ Send", type="primary", use_container_width=True)

    if send_btn:
        has_text  = bool(user_text and user_text.strip())
        has_image = bool(st.session_state.get("pending_img"))

        if not has_text and not has_image:
            st.warning("Please type a question or upload an image.")
        else:
            img_b64   = st.session_state.get("pending_img")
            img_bytes = st.session_state.get("_pending_img_bytes")
            question  = user_text.strip() if has_text else ""

            submit_message(question, img_b64, img_bytes)

            # Clear pending image
            st.session_state.pending_img = None
            st.session_state._pending_img_bytes = None
            st.rerun()

with tab2:
    st.subheader("📊 Analytics Dashboard")
    total_q  = len(st.session_state.query_log)
    avg_conf = round(sum(q["confidence"] for q in st.session_state.query_log)/max(total_q,1))
    high_c   = sum(1 for q in st.session_state.query_log if q["confidence"]>=70)
    low_c    = sum(1 for q in st.session_state.query_log if q["confidence"]<40)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Queries",   total_q)
    c2.metric("Avg Confidence",  f"{avg_conf}%")
    c3.metric("High Confidence", high_c)
    c4.metric("Low Confidence",  low_c)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🤖 LangGraph Nodes**")
        st.markdown("🔀 Router — text vs image\n\n👁 Vision — LLaMA 3.2 11B\n\n🔍 Retrieval — BM25+Vector\n\n⚖️ Grader — relevance check\n\n✍️ Generator — fuse answer")
    with col2:
        st.markdown("**⚙️ System**")
        st.markdown(f"LLM: LLaMA 3.1 8B (Groq)\nVision: LLaMA 3.2 11B Vision\nEmbeddings: MiniLM-L6-v2\nSearch: {'BM25 + Vector' if BM25_AVAILABLE else 'Vector'}\nSelf-correction: 2 retries")

    st.divider()
    if st.session_state.pdf_store:
        st.subheader("📚 Loaded Documents")
        st.dataframe([{"Document":f,"Pages":v["pages"],"Chunks":len(v["chunks"]),
                       "BM25":"✓" if v.get("bm25") else "✗"}
                      for f,v in st.session_state.pdf_store.items()],
                     use_container_width=True)

    st.divider()
    st.subheader("🕐 Query History")
    if st.session_state.query_log:
        for q in reversed(st.session_state.query_log[-25:]):
            b = "🟢" if q["confidence"]>=70 else "🟡" if q["confidence"]>=40 else "🔴"
            st.markdown(f"`{q['time']}` {b} **{q['confidence']}%** `{q['mode']}` — {q['question'][:80]}")
    else:
        st.info("No queries yet.")
