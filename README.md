# 🧠 LibreChat AI

**LibreChat AI** is a Persian AI assistant specifically designed for Linux and Free Software topics.  
It is built using **LangChain + Ollama + Chroma + Streamlit**, and includes Few-shot learning, RAG (Retrieval-Augmented Generation), and chat memory management.

---

## Features

### 1️⃣ Few-shot Learning
- Includes **20 Persian question-answer examples** on Linux and Free Software.
- Ensures **accurate, context-aware Persian responses** even without PDFs or a vector database.
- Few-shot examples are embedded in the model prompt and influence the style of answers.

### 2️⃣ Retrieval-Augmented Generation (RAG)
- If a **Chroma vector store** exists (populated from PDF documents), the model can retrieve relevant information for more accurate responses.
- PDFs can be ingested using the `ingest_pdfs(data_dir)` method.

### 3️⃣ Chat Memory
- Chat history is stored in **Streamlit session state** (`st.session_state`).
- Each new chat has a **unique UUID**, and previous chats can be selected and displayed.
- Supports multi-turn conversations with context.

### 4️⃣ Persian Text Normalization
- Automatically cleans and normalizes Persian text:
  - Converts Arabic letters to Persian equivalents
  - Removes hidden and special characters
  - Strips extra whitespace

### 5️⃣ Ollama LLM Integration
- Works with **Ollama models** (e.g., `gemma3:4b`) for Persian-language generation.
- Streaming responses are enabled for faster feedback.

### 6️⃣ PDF Ingestion (Optional)
- Users can add PDF files to the `data/` folder.
- Text is split into chunks and stored in Chroma for RAG.
- Optional: the model can run without any PDFs.

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/librechat-ai.git
cd librechat-ai

2. **Create a virtual environment**
```
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


3. **Install dependencies**
```
pip install -r requirements.txt
Make sure you have Python 3.10+ and a working Ollama setup.
```

4. **Optional: Ingest PDFs**
```
from model import LibreChatModel
model = LibreChatModel()
num_chunks = model.ingest_pdfs("data")
print(f"{num_chunks} chunks ingested.")
```

5. **Running the Streamlit UI**
```
streamlit run ui.py
```

Open your browser at the provided local URL.
Use the sidebar to start a new chat or select previous chats.
Ask questions in Persian about Linux and Free Software.
Responses will combine Few-shot learning, chat memory, and RAG (if vector DB exists).


**Notes**

Even without PDFs, the model works using Few-shot examples and chat memory.
PDF ingestion is optional but recommended for richer context.
Supports GPU acceleration if available for embeddings.

**License**


This project is released under the MIT License. Feel free to use, modify, and distribute.


