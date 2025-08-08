# 🧠 Insurance LLM-Powered Query–Retrieval System

This project is a **FastAPI-based intelligent query–retrieval system** that processes large documents (PDF, DOCX, EML) and answers **natural language questions** with **semantic search and multilingual support**.

It is designed for hackathon submissions like **HackRx** and supports both **Pinecone** and **FAISS** for vector retrieval.

---

## ✨ Features

- 📄 **Multiple Document Types**: PDF, DOCX, EML  
- 🔍 **OCR Support** for scanned documents/images (Tesseract)  
- 🧠 **Semantic Search** with `SentenceTransformer` embeddings (`all-MiniLM-L6-v2`)  
- 🗄 **Vector Database**:
  - Pinecone (cloud, persistent, scalable)  
  - FAISS (offline fallback)  
- 🌐 **Multilingual Support**:
  - FastText language detection (auto-download if missing)  
  - M2M100 translation model for non-English queries  
- 🔐 **Secure API**: Bearer token authentication

---

## 📦 Requirements

### Python
Python 3.10+

### Dependencies
(Installed via `pip` or inside Docker build)

```
fastapi
uvicorn[standard]
requests
pdfplumber
python-docx
pdf2image
pillow
pytesseract
sentence-transformers
transformers
torch
faiss-cpu
pinecone-client
fasttext
pyyaml
```

---

## 🚀 Running the API

### 1. Clone & Setup

```bash
git clone https://github.com/<your-username>/insurance-qa.git
cd insurance-qa
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Without Docker

```bash
uvicorn insurance_api:app --host 0.0.0.0 --port 8000
```

Open Swagger docs at:
```bash
http://localhost:8000/docs
```

### 3. Run With Docker

```bash
docker build -t insurance-api .
docker run -p 8000:8000 \
  -e PINECONE_API_KEY="your_key" \
  -e PINECONE_ENV="your_env" \
  insurance-api
```

---

## 🔑 Authentication

All requests require:
```
Authorization: Bearer 57f9dc886a5611894d0824b60df338012759dad04387696807ae6e5f287f531f
```

---

## 📤 API Usage

### Endpoint
```bash
POST /api/v1/hackrx/run
```

### Request Body
```json
{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "Does this policy cover knee surgery, and what are the conditions?",
    "What is the waiting period for cataract surgery?"
  ]
}
```

**Note**: `"documents"` can be a single URL or an array of URLs.

### Example curl Command
```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Authorization: Bearer 57f9dc886a5611894d0824b60df338012759dad04387696807ae6e5f287f531f" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=...",
    "questions": [
      "Does this policy cover knee surgery, and what are the conditions?",
      "What is the waiting period for cataract surgery?"
    ]
  }'
```

### Example Response
```json
{
  "answers": [
    "Yes, the policy covers knee surgery under certain conditions...",
    "The waiting period for cataract surgery is two years..."
  ]
}
```

---

## 🌐 Environment Variables

| Variable | Description |
|----------|-------------|
| `PINECONE_API_KEY` | Pinecone API Key (optional) |
| `PINECONE_ENV` | Pinecone Environment (optional) |

If Pinecone is not configured, FAISS local indexing will be used.

---

## 📝 Notes

- **FastText** `lid.176.bin` (~126 MB) is auto-downloaded on first run if missing.
- **Models** (`all-MiniLM-L6-v2`, `M2M100`) are also downloaded on first use.
- For production or hackathons, build these models into the Docker image to reduce cold start time.

---

## 🛠 Development Setup

### Prerequisites
- Python 3.10+
- Docker (optional)
- Tesseract OCR installed on system

### Installation Steps
1. Clone the repository
2. Create virtual environment
3. Install dependencies
4. Set environment variables (optional for Pinecone)
5. Run the application

### Testing
Use the provided curl examples or access the interactive API documentation at `/docs` when the server is running.

---

## 🏆 Hackathon Ready

This system is optimized for hackathon submissions with:
- Fast setup and deployment
- Comprehensive document processing
- Multilingual capabilities
- Scalable vector search
- Production-ready API design

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
