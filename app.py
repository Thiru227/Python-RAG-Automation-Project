import os
cache_dir = "/tmp/huggingface"
os.makedirs(cache_dir, exist_ok=True)

os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["HF_HUB_CACHE"] = cache_dir
os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir

# IMPORTANT: Disable ChromaDB telemetry and set to ephemeral mode
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import json
import time
import shutil
import requests
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flask import Flask, request, jsonify, render_template
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import chromadb
from langchain_chroma import Chroma

# CONFIG
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-r1:free"

def ensure_vector_store(embeddings):
    """Create the Chroma DB in ephemeral (in-memory) mode for HF Spaces."""
    
    # Collect documents from env override or from data/ folder
    collected_docs = []
    pdf_path = os.getenv("INGEST_PDF_PATH")
    if pdf_path and os.path.exists(pdf_path):
        collected_docs.extend(PyPDFLoader(pdf_path).load())
    else:
        data_root = os.path.join(os.getcwd(), "data")
        if os.path.isdir(data_root):
            for root, _, files in os.walk(data_root):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    try:
                        if fname.lower().endswith(".pdf"):
                            collected_docs.extend(PyPDFLoader(fpath).load())
                        elif fname.lower().endswith(".txt"):
                            collected_docs.extend(TextLoader(fpath, encoding="utf8").load())
                    except Exception as e:
                        print(f"Warning: failed to load {fpath}: {e}")

    if not collected_docs:
        print("No documents found to ingest in data/. Creating empty vector store.")
        return None
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(collected_docs)
    
    # Use ephemeral client (in-memory) instead of persistent
    client = chromadb.EphemeralClient()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name="rag_collection"
    )
    
    print(f"Ingested {len(chunks)} chunks into ephemeral vector store.")
    return vectorstore

# Load embeddings and create vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = ensure_vector_store(embeddings)

app = Flask(__name__, static_folder='static', template_folder='templates')

def retrieve_context(question, k=3):
    if db is None:
        return "No documents available."
    docs = db.similarity_search(question, k=k)
    return "\n\n".join(d.page_content for d in docs)

def call_openrouter_system(system_prompt, user_prompt):
    if not OPENROUTER_KEY:
        raise RuntimeError("Set OPENROUTER_API_KEY in environment.")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.0
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json"
    }
    # retry on 429/5xx with exponential backoff
    max_retries = 4
    backoff = 1.0
    last_error_text = None
    for attempt in range(max_retries + 1):
        r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
        if r.status_code == 429 or 500 <= r.status_code < 600:
            try:
                err = r.json()
                last_error_text = err.get("error") or err
            except Exception:
                last_error_text = r.text
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2
                continue
        r.raise_for_status()
        jr = r.json()
        return jr["choices"][0]["message"]["content"].strip()
    raise RuntimeError(f"OpenRouter request failed after retries: {last_error_text}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    q = data.get("question", "").strip()
    if not q:
        return jsonify({"error": "question required"}), 400

    context = retrieve_context(q, k=3)
    system = (
        "You are an educational assistant. Answer using ONLY the context between <CONTEXT> tags. "
        "If the answer is not in the context, say 'I don't know from the provided material.'"
    )
    user = f"<CONTEXT>\n{context}\n</CONTEXT>\n\nQuestion: {q}\nAnswer concisely:"
    try:
        answer = call_openrouter_system(system, user)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": "LLM request failed. Please retry in a moment.", "detail": str(e)}), 502

@app.route("/webhook", methods=["POST"])
def webhook():
    req = request.get_json(silent=True, force=True)
    q = None
    if req and "queryResult" in req:
        q = req["queryResult"].get("queryText")
    if not q:
        return jsonify({"fulfillmentText": "Sorry, I couldn't read your question."})
    context = retrieve_context(q, k=3)
    system = (
        "You are an educational assistant. Use ONLY the context between <CONTEXT> tags to answer. "
        "If not in context, say 'I don't know from the provided material.'"
    )
    user = f"<CONTEXT>\n{context}\n</CONTEXT>\n\nQuestion: {q}\nAnswer briefly:"
    try:
        ans = call_openrouter_system(system, user)
        return jsonify({"fulfillmentText": ans})
    except Exception:
        return jsonify({"fulfillmentText": "The model is busy. Please try again shortly."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
