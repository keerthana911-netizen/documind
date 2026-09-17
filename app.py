import streamlit as st
import os
import tempfile
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ────────────────────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
COLLECTION_NAME = "documind_docs"
MILVUS_HOST     = "localhost"
MILVUS_PORT     = "19530"
EMBED_MODEL     = "all-MiniLM-L6-v2"   # 384-dim, fast
EMBED_DIM       = 384
OLLAMA_MODEL    = "llama3.2"
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 150
TOP_K           = 5

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }
    h1 { color: #00d4ff; font-family: 'Courier New', monospace; }
    h2, h3 { color: #7dd3fc; }
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #00d4ff33;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .status-ok   { color: #4ade80; font-weight: bold; }
    .status-fail { color: #f87171; font-weight: bold; }
    .answer-box  {
        background: #1e293b;
        border-left: 4px solid #00d4ff;
        border-radius: 8px;
        padding: 16px;
        margin-top: 8px;
        color: #e2e8f0;
    }
    .source-box  {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 10px;
        margin: 4px 0;
        font-size: 0.85em;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load the sentence-transformer model once."""
    return SentenceTransformer(EMBED_MODEL)


def connect_milvus():
    """Connect to Milvus (standalone or docker)."""
    try:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        return True, "Connected to Milvus ✓"
    except Exception as e:
        return False, f"Milvus connection failed: {e}"


def get_or_create_collection(drop_existing: bool = False) -> Collection:
    """Return (or create) the Milvus collection."""
    if drop_existing and utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    if utility.has_collection(COLLECTION_NAME):
        col = Collection(COLLECTION_NAME)
        col.load()
        return col

    fields = [
        FieldSchema(name="id",        dtype=DataType.INT64,         is_primary=True, auto_id=True),
        FieldSchema(name="text",      dtype=DataType.VARCHAR,        max_length=4096),
        FieldSchema(name="source",    dtype=DataType.VARCHAR,        max_length=512),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR,   dim=EMBED_DIM),
    ]
    schema = CollectionSchema(fields, description="Document embeddings")
    col = Collection(COLLECTION_NAME, schema)

    # Create IVF_FLAT index for fast ANN search
    index_params = {
        "metric_type": "COSINE",
        "index_type":  "IVF_FLAT",
        "params":      {"nlist": 128},
    }
    col.create_index("embedding", index_params)
    col.load()
    return col


def load_documents(uploaded_files) -> list:
    """Load PDFs and DOCX files into LangChain Document objects."""
    docs = []
    for f in uploaded_files:
        suffix = Path(f.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(tmp_path)
            elif suffix in (".docx", ".doc"):
                loader = Docx2txtLoader(tmp_path)
            else:
                st.warning(f"Unsupported file type: {f.name}")
                continue
            loaded = loader.load()
            for d in loaded:
                d.metadata["source"] = f.name
            docs.extend(loaded)
        finally:
            os.unlink(tmp_path)
    return docs


def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def embed_and_store(chunks: list, embed_model, collection: Collection) -> int:
    """Vectorise chunks and insert into Milvus."""
    texts   = [c.page_content for c in chunks]
    sources = [c.metadata.get("source", "unknown") for c in chunks]

    # Embed in small batches to avoid OOM
    batch_size = 64
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs  = embed_model.encode(batch, normalize_embeddings=True)
        all_embeddings.extend(vecs.tolist())

    data = [texts, sources, all_embeddings]
    collection.insert(data)
    collection.flush()
    return len(texts)


def similarity_search(query: str, embed_model, collection: Collection, top_k: int = TOP_K):
    """Return top-k most relevant chunks for the query."""
    q_vec = embed_model.encode([query], normalize_embeddings=True).tolist()
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 32}}
    results = collection.search(
        data         = q_vec,
        anns_field   = "embedding",
        param        = search_params,
        limit        = top_k,
        output_fields= ["text", "source"],
    )
    hits = []
    for hit in results[0]:
        hits.append({
            "text":   hit.entity.get("text"),
            "source": hit.entity.get("source"),
            "score":  round(hit.score, 4),
        })
    return hits


def build_rag_answer(question: str, context_hits: list) -> str:
    """Run the RAG chain with Ollama Llama 3.2."""
    context = "\n\n---\n\n".join(
        [f"[Source: {h['source']}]\n{h['text']}" for h in context_hits]
    )

    prompt = PromptTemplate.from_template("""You are a helpful assistant that answers questions using only the provided documents.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I couldn't find this information in the uploaded documents."

Context:
{context}

Question: {question}

Answer (be concise and accurate):""")

    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.1)

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ System Status")

    # Milvus status
    milvus_ok, milvus_msg = connect_milvus()
    if milvus_ok:
        st.markdown(f'<span class="status-ok">🟢 Milvus</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="status-fail">🔴 Milvus — {milvus_msg}</span>', unsafe_allow_html=True)
        st.info("Start Milvus: `docker-compose up -d`")

    # Ollama status
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False

    if ollama_ok:
        st.markdown('<span class="status-ok">🟢 Ollama (llama3.2)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-fail">🔴 Ollama not running</span>', unsafe_allow_html=True)
        st.info("Start Ollama: `ollama serve`")

    st.divider()

    st.markdown("## 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF / DOCX files",
        type=["pdf", "docx", "doc"],
        accept_multiple_files=True,
    )

    clear_existing = st.checkbox("🗑️ Clear existing data before indexing", value=False)

    process_btn = st.button("⚡ Process & Index Documents", type="primary", use_container_width=True)

    if "doc_count" in st.session_state:
        st.success(f"✅ {st.session_state.doc_count} chunks indexed")

    st.divider()
    st.markdown("## 🔧 Settings")
    top_k_val = st.slider("Top-K results", 1, 10, TOP_K)
    st.caption(f"Model: `{OLLAMA_MODEL}` | Embed: `{EMBED_MODEL}`")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("# 🧠 DocuMind")
st.markdown("*Powered by **Milvus** · **Ollama Llama 3.2** · **Sentence Transformers***")
st.divider()

# ── Metrics row ────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Vector Store", "Milvus", delta="✓ Connected" if milvus_ok else "✗ Offline")
with col2:
    st.metric("LLM", "Llama 3.2", delta="via Ollama (local)")
with col3:
    st.metric("Embeddings", "MiniLM-L6", delta=f"{EMBED_DIM}d vectors")
with col4:
    chunks_stored = st.session_state.get("doc_count", 0)
    st.metric("Chunks Indexed", chunks_stored)

st.divider()

# ── Document Processing ────────────────────────────────────────────────────────
if process_btn:
    if not uploaded_files:
        st.error("Please upload at least one PDF or DOCX file first.")
    elif not milvus_ok:
        st.error("Cannot index — Milvus is not running. Start it with `docker-compose up -d`")
    else:
        with st.spinner("Loading & chunking documents…"):
            docs   = load_documents(uploaded_files)
            chunks = chunk_documents(docs)

        st.info(f"📄 Loaded **{len(docs)}** pages → **{len(chunks)}** chunks")

        with st.spinner("Embedding & storing in Milvus…"):
            embed_model = load_embedding_model()
            collection  = get_or_create_collection(drop_existing=clear_existing)
            n = embed_and_store(chunks, embed_model, collection)

        st.session_state.doc_count = n
        st.success(f"✅ Successfully indexed **{n}** chunks into Milvus!")
        st.balloons()

# ── Q&A Interface ──────────────────────────────────────────────────────────────
st.markdown("## 💬 Ask Your Documents")

question = st.text_input(
    "Type your question here…",
    placeholder="e.g. What does section 3 of the document say about eligibility?",
)

ask_btn = st.button("🔍 Ask", type="primary")

if ask_btn and question:
    if not milvus_ok:
        st.error("Milvus is not running.")
    elif not ollama_ok:
        st.error("Ollama is not running. Start with `ollama serve`")
    else:
        try:
            embed_model = load_embedding_model()
            collection  = get_or_create_collection()

            with st.spinner("Searching Milvus for relevant context…"):
                hits = similarity_search(question, embed_model, collection, top_k=top_k_val)

            if not hits:
                st.warning("No relevant documents found. Please upload and index documents first.")
            else:
                with st.spinner("Generating answer with Llama 3.2…"):
                    answer = build_rag_answer(question, hits)

                st.markdown("### 🤖 Answer")
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                st.markdown("### 📚 Source Chunks Used")
                for i, hit in enumerate(hits, 1):
                    with st.expander(f"Chunk {i} · {hit['source']} · Score: {hit['score']}"):
                        st.write(hit["text"])

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)

elif ask_btn and not question:
    st.warning("Please enter a question.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("🔒 Fully local — your documents never leave your machine.")
