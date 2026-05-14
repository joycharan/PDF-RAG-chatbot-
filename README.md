# PDF RAG ChatBot with Gemini API and Gradio

PDFChatBot is a Python-based chatbot designed to answer questions based on the content of uploaded PDF files. It utilizes the Gradio library for a user-friendly web interface and LangChain for Retrieval-Augmented Generation (RAG). This version uses the **Google Gemini API** for language model responses, making it accessible even on machines without GPU support.

---

## Technologies Used 🚀

- **LangChain** (RAG pipeline)
- **Google Gemini API** (LLM)
- **ChromaDB** (vector store)
- **Hugging Face** (embeddings)
- **Gradio** (web UI)
- **PyMuPDF** (PDF parsing)

---

## Features ⭐

- Upload and process PDF files (supports large files and multi-page documents)
- Ask questions about the PDF content using natural language
- Maintains chat history for context-aware answers
- Uses Gemini API for fast, cloud-based LLM responses
- Displays relevant PDF page previews alongside answers
- Handles large PDFs with chunking and memory optimizations

---

## Prerequisites 📋

- Python 3.10+
- Google Gemini API key (get one from [Google AI Studio](https://aistudio.google.com/app/apikey))

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration ⚙️

Edit `config.yaml` to set your API key and model:

```yaml
GOOGLE_API_KEY: "your-gemini-api-key"
LLM_API_TYPE: "google"
GOOGLE_MODEL_NAME: "gemini-2.5-flash-lite" # or another supported Gemini model
modelEmbeddings: "sentence-transformers/all-MiniLM-L6-v2"
```

---

## Usage 📚

1. Upload a PDF file using the "📁 Upload PDF" button.
2. Enter your question in the text box.
3. Click "Send" to get an answer based on the PDF content.
4. View chat history and PDF page previews in the interface.

---

## Running Locally 💻

From the project root, run:

```bash
python src/app.py
```

- To create a public link, set `share=True` in `demo.launch()` in `app.py`.

---

## Notes & Limitations ⚠️

- Free Gemini API keys have usage quotas and may rate-limit heavy use.
- Large PDFs (hundreds of pages) are supported, but extremely large files may be slow or hit memory limits.
- For best results, use clear, specific questions about the document content.

---



