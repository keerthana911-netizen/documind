# 🏢 Company RAG System
### Milvus · Ollama Llama 3.2 · Sentence Transformers · Streamlit

---

## ⚡ QUICK START (Run in order, copy-paste each block)

### Step 1 — Start Milvus via Docker
```bash
# Make sure Docker Desktop is running first!
docker-compose up -d

# Wait ~30 seconds, then verify it's healthy:
docker ps
# You should see: milvus-standalone, milvus-minio, milvus-etcd  → STATUS: healthy
```

### Step 2 — Start Ollama (new terminal)
```bash
ollama serve
# Keep this terminal open
```

### Step 3 — Make sure Llama 3.2 is pulled
```bash
# In another terminal:
ollama pull llama3.2
```

### Step 4 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 5 — Run the app
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

---

## 🎯 How to Use in the Demo

1. Open **http://localhost:8501**
2. Check sidebar — Milvus 🟢 and Ollama 🟢 should both be green
3. Upload your company PDF/DOCX files in the sidebar
4. Click **"⚡ Process & Index Documents"** — wait for the ✅
5. Type a question in the main area and click **"🔍 Ask"**
6. See the AI answer + the exact source chunks it used

---

## 🏗️ Architecture

```
PDF / DOCX Files
      ↓
LangChain Loaders + Text Splitter (800 char chunks)
      ↓
Sentence Transformers (all-MiniLM-L6-v2) → 384-dim vectors
      ↓
Milvus Vector Store (IVF_FLAT + COSINE similarity)
      ↓
User Question → Embed → Milvus ANN Search → Top-5 chunks
      ↓
LangChain RAG Chain → Ollama Llama 3.2 (local)
      ↓
Answer + Source Attribution
```

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| Milvus 🔴 | `docker-compose up -d` and wait 30s |
| Ollama 🔴 | `ollama serve` in a terminal |
| Port 19530 in use | `docker-compose down` then `up -d` |
| Model not found | `ollama pull llama3.2` |
| Slow first answer | Normal — model loads into RAM once, then fast |

---

## 📁 File Structure
```
company-rag/
├── app.py              ← Main Streamlit app (ALL logic here)
├── docker-compose.yml  ← Milvus standalone stack
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```
# documind
