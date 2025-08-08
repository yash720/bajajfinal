#!/usr/bin/env python3
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import re
import json
import tempfile
import requests
import pdfplumber
import pytesseract
import numpy as np
import faiss
import pinecone
import fasttext
from docx import Document
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

app = FastAPI()

EXPECTED_TOKEN = "57f9dc886a5611894d0824b60df338012759dad04387696807ae6e5f287f531f"

# ---------------- Language detection model auto-download ----------------
FASTTEXT_MODEL_PATH = "lid.176.bin"
if not os.path.exists(FASTTEXT_MODEL_PATH):
    print("Downloading FastText language detection model...")
    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(FASTTEXT_MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
lang_detector = fasttext.load_model(FASTTEXT_MODEL_PATH)

# ---------------- Embedding + Translation Models ----------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
translation_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
translation_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

# ---------------- Pinecone / FAISS Init ----------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = "insurance-clauses"

if PINECONE_API_KEY and PINECONE_ENV:
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    if PINECONE_INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(PINECONE_INDEX_NAME, dimension=384)
    pinecone_index = pinecone.Index(PINECONE_INDEX_NAME)
else:
    pinecone_index = None
    faiss_index = faiss.IndexFlatL2(384)
    faiss_meta = []

# ---------------- Helper Functions ----------------
def detect_language(text):
    pred = lang_detector.predict(text.replace("\n", " ")[:2000])[0][0]
    return pred.replace("__label__", "")

def translate_to_english(text, src_lang):
    if src_lang.startswith("en"):
        return text
    translation_tokenizer.src_lang = src_lang
    encoded = translation_tokenizer(text, return_tensors="pt")
    generated = translation_model.generate(**encoded, forced_bos_token_id=translation_tokenizer.get_lang_id("en"))
    return translation_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

def download_file(url):
    r = requests.get(url, stream=True, timeout=60)
    if r.status_code != 200:
        raise ValueError(f"Failed to download: {r.status_code}")
    suffix = os.path.splitext(url.split("?")[0])[1] or ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with open(tmp.name, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    return tmp.name

def ocr_image(img):
    return pytesseract.image_to_string(img)

def parse_pdf(path):
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
                else:
                    img = page.to_image(resolution=300)
                    text += ocr_image(img.original)
    except:
        images = convert_from_path(path)
        for img in images:
            text += ocr_image(img)
    return text

def parse_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def parse_eml(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def parse_document(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text = parse_pdf(path)
    elif ext == ".docx":
        text = parse_docx(path)
    elif ext == ".eml":
        text = parse_eml(path)
    else:
        text = ""
    return extract_clauses(text, os.path.basename(path))

def extract_clauses(text, source):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    clauses = []
    for i, p in enumerate(paras):
        clauses.append((p, {"file": source, "position": i+1}))
    return clauses

def upsert_clauses(clauses):
    texts = [c[0] for c in clauses]
    metas = [c[1] for c in clauses]
    vecs = embedder.encode(texts, convert_to_tensor=False).tolist()
    if pinecone_index:
        pinecone_index.upsert([
            (f"{m['file']}_{m['position']}", v, {**m, "text": t})
            for t, v, m in zip(texts, vecs, metas)
        ])
    else:
        global faiss_index, faiss_meta
        faiss_index.add(np.array(vecs).astype("float32"))
        faiss_meta.extend([{"text": t, **m} for t, m in zip(texts, metas)])

def search_clauses(query, top_k=5):
    qvec = embedder.encode([query], convert_to_tensor=False).astype("float32")
    if pinecone_index:
        res = pinecone_index.query(qvec.tolist()[0], top_k=top_k, include_metadata=True)
        return [((m["text"], {"file": m["file"], "position": m["position"]}), match["score"]) for match in res["matches"]]
    else:
        D, I = faiss_index.search(qvec, top_k)
        return [((faiss_meta[idx]["text"], {"file": faiss_meta[idx]["file"], "position": faiss_meta[idx]["position"]}), float(D[0][rank])) for rank, idx in enumerate(I[0]) if idx != -1]

def evaluate_decision(query, matches):
    answer_text = "Not found"
    if matches:
        combined = " ".join([m[0][0] for m in matches])
        answer_text = combined
    return answer_text

# ---------------- API Models & Endpoint ----------------
class RunRequest(BaseModel):
    documents: list | str
    questions: list

@app.post("/api/v1/hackrx/run")
def hackrx_run(payload: RunRequest, authorization: str = Header(None)):
    if not authorization or EXPECTED_TOKEN not in authorization:
        raise HTTPException(status_code=401, detail="Unauthorized")
    docs = payload.documents if isinstance(payload.documents, list) else [payload.documents]
    for d in docs:
        path = download_file(d)
        clauses = parse_document(path)
        upsert_clauses(clauses)
        os.remove(path)
    answers = []
    for q in payload.questions:
        lang = detect_language(q)
        q_en = translate_to_english(q, lang)
        matches = search_clauses(q_en, top_k=5)
        decision = evaluate_decision(q_en, matches)
        answers.append(decision)
    return {"answers": answers}
