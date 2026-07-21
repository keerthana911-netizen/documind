# 🧠 DocuMind
### Ask questions of your own documents — fully local RAG pipeline, no cloud, no API keys

> **No live demo link** — this depends on a Milvus container stack (Docker) plus a locally-running Ollama LLM server, infrastructure that free hosting tiers (Streamlit Cloud, Hugging Face Spaces) can't run. Clone the repo and run it locally to try it — see [Run it](#run-it) below.

---

## What this is

Upload PDFs or Word documents, and ask questions about them in plain English. DocuMind retrieves the most relevant passages and generates a grounded answer — with the exact source chunks it used shown alongside, so you can verify it isn't making things up.

This is a **retrieval-augmented generation (RAG)** system, and every piece of it — embedding, vector search, and the LLM itself — runs locally. No document ever leaves your machine.

---

## Architecture

```
PDF / DOCX files
      │
      ▼
LangChain loaders (PyPDFLoader / Docx2txtLoader)
      │
      ▼
Recursive text splitter (800-char chunks, 150-char overlap)
      │
      ▼
Sentence-Transformers (all-MiniLM-L6-v2) → 384-dim embeddings
      │
      ▼
Milvus (IVF_FLAT index, COSINE similarity)
      │
      │  ◄── user question, embedded the same way
      ▼
Top-K similarity search → most relevant chunks
      │
      ▼
LangChain RAG chain → Llama 3.2 (via Ollama, local)
      │
      ▼
Grounded answer + source chunk attribution
```

---

## Why the design choices matter

- **Chunking with overlap (800/150)** — prevents a fact from being cleanly severed at a chunk boundary; overlap means near-boundary context survives in both neighboring chunks.
- **COSINE similarity over raw L2** — for text embeddings, cosine similarity is more robust to variation in text length between chunks than Euclidean distance.
- **Strict "answer only from context" prompting** — the LLM is explicitly told to say it doesn't know rather than hallucinate when the retrieved chunks don't contain the answer. This is the difference between a RAG system you can actually trust and one that just sounds confident.
- **Source attribution shown alongside every answer** — every generated answer displays exactly which chunks (and from which file) it was grounded in, so a user isn't just taking the model's word for it.
- **Live system status in the sidebar** — Milvus and Ollama connection health are checked and shown in real time, so failures are diagnosable instead of silent.

---

## Stack

| Component | Role |
|---|---|
| Streamlit | UI |
| LangChain | Document loading, chunking, RAG chain orchestration |
| Milvus | Vector database (IVF_FLAT index, COSINE metric) |
| Sentence-Transformers (`all-MiniLM-L6-v2`) | Text → 384-dim embeddings |
| Ollama + Llama 3.2 | Local LLM inference |

**Note:** document loading uses standard text extraction (`PyPDFLoader`, `Docx2txtLoader`), not OCR — so it works on PDFs with real selectable text, not scanned image-only documents.

---

## Run it

```bash
# 1. Start Milvus (needs the docker-compose.yml from the milvus-vector-db repo,
#    or your own standalone Milvus setup)
docker-compose up -d

# 2. Start Ollama and pull the model
ollama serve
ollama pull llama3.2

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch
streamlit run app.py
```

Opens at `http://localhost:8501`.

1. Check the sidebar — Milvus and Ollama should both show 🟢
2. Upload PDF/DOCX files
3. Click **"⚡ Process & Index Documents"**
4. Ask a question, get an answer with cited source chunks

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Milvus 🔴 | `docker-compose up -d`, wait ~30s for containers to become healthy |
| Ollama 🔴 | Run `ollama serve` in a separate terminal |
| Port 19530 already in use | `docker-compose down` then `up -d` again |
| Model not found | `ollama pull llama3.2` |
| Slow first answer | Normal — model loads into RAM once, then subsequent answers are fast |

---


> "DocuMind is a fully local RAG pipeline — documents get chunked, embedded, and stored in Milvus, then a question gets embedded the same way and matched against the closest chunks by cosine similarity. Llama 3.2 generates the answer strictly from retrieved context, with every answer showing exactly which source chunks it came from, so it's a system you can actually audit rather than just trust."
