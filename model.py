import os
import re
import unicodedata
import torch

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


class LibreChatModel:
    """
    LibreChat AI model with support for:
    - Persian text normalization
    - Few-shot examples
    - RAG using Chroma VectorStore
    - Chat memory handling
    - Ollama LLM
    """
    def __init__(self, db_directory="./my_local_db", model_name="gemma3:4b"):
        """
        Initialize LibreChatModel.

        Args:
            db_directory (str): Path to store or load vector database.
            model_name (str): Ollama model name.
        """
        self.db_directory = db_directory
        self.model_name = model_name

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Embedding (appropriate for persian)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": device}
        )

        # LLM (Ollama)
        self.llm = OllamaLLM(model=self.model_name, streaming=True)

        # Vector DB (if existed)
        self.vector_store = self._load_vector_db()

        # Few-shot examples 
        self.few_shot_examples = [
            {"input": "لینوکس چیست؟", "output": "لینوکس یک سیستم عامل آزاد و متن‌باز است که روی هسته لینوکس اجرا می‌شود."},
            {"input": "تفاوت لینوکس و ویندوز چیست؟", "output": "لینوکس متن‌باز و رایگان است، ویندوز مالکیتی و پولی است."},
            {"input": "توزیع لینوکس چیست؟", "output": "توزیع لینوکس نسخه‌ای از لینوکس با بسته‌های نرم‌افزاری و مدیریت بسته مخصوص خودش است."},
            {"input": "Ubuntu چیست؟", "output": "Ubuntu یک توزیع محبوب لینوکس است که کاربرپسند و مناسب دسکتاپ و سرور است."},
            {"input": "Debian چیست؟", "output": "Debian یک توزیع پایدار و متن‌باز است که پایه بسیاری از توزیع‌ها مانند Ubuntu است."},
            {"input": "Red Hat چیست؟", "output": "Red Hat یک توزیع تجاری لینوکس است و برای سرورها و محیط‌های سازمانی استفاده می‌شود."},
            {"input": "Kernel لینوکس چیست؟", "output": "Kernel هسته سیستم عامل لینوکس است که مدیریت سخت‌افزار و منابع را بر عهده دارد."},
            {"input": "Shell در لینوکس چیست؟", "output": "Shell رابط خط فرمان است که دستورات کاربر را به سیستم عامل منتقل می‌کند."},
            {"input": "Bash چیست؟", "output": "Bash یک نوع shell لینوکسی است که در بسیاری از توزیع‌ها به صورت پیش‌فرض وجود دارد."},
            {"input": "Package Manager چیست؟", "output": "Package Manager ابزاری برای نصب، حذف و به‌روزرسانی نرم‌افزارها در لینوکس است."},
            {"input": "apt چیست؟", "output": "apt ابزار مدیریت بسته در توزیع‌های Debian و Ubuntu است."},
            {"input": "yum چیست؟", "output": "yum ابزار مدیریت بسته در توزیع‌های Red Hat و CentOS است."},
            {"input": "GPL چیست؟", "output": "GPL یک مجوز نرم‌افزار آزاد است که آزادی استفاده، تغییر و توزیع را تضمین می‌کند."},
            {"input": "نرم‌افزار آزاد چیست؟", "output": "نرم‌افزار آزاد به کاربر آزادی اجرای، مطالعه، تغییر و توزیع نرم‌افزار را می‌دهد."},
            {"input": "تفاوت نرم‌افزار آزاد و متن‌باز چیست؟", "output": "نرم‌افزار آزاد روی آزادی کاربر تأکید دارد و متن‌باز بیشتر روی دسترسی به کد منبع."},
            {"input": "Systemd چیست؟", "output": "Systemd یک سیستم مدیریت سرویس‌ها و بوت لینوکس است."},
            {"input": "Cron چیست؟", "output": "Cron ابزار زمان‌بندی وظایف در لینوکس است."},
            {"input": "SSH چیست؟", "output": "SSH پروتکلی برای دسترسی امن به سرورهای لینوکس از راه دور است."},
            {"input": "Filesystem چیست؟", "output": "Filesystem ساختار سازماندهی و ذخیره فایل‌ها روی دیسک است."},
            {"input": "Root در لینوکس چیست؟", "output": "Root کاربر اصلی با بیشترین سطح دسترسی در لینوکس است."},
            {"input": "استالمن کیست", "output": "خالق لینوکس"}
            
        ]

    # ---------- Vector DB ----------
    def _load_vector_db(self):
        """
        Load the Chroma vector database if exists.

        Returns:
            Chroma or None: Returns Chroma instance or None if DB not found.
        """
        if os.path.exists(self.db_directory) and os.listdir(self.db_directory):
            return Chroma(
                persist_directory=self.db_directory,
                embedding_function=self.embeddings
            )
        return None

    # ---------- Persian Normalization ----------
    def normalize_persian(self, text: str) -> str:
        """
        Normalize Persian text by:
        - Replacing Arabic letters with Persian equivalents
        - Removing hidden and special characters
        - Removing extra whitespaces

        Args:
            text (str): Input Persian text.

        Returns:
            str: Normalized Persian text.
        """
        text = unicodedata.normalize("NFC", text)
        artifacts = ["\u200b", "\u200c", "\u200d", "\u200e", "\u200f", "\u2060", "\ufeff", "Ͷ", "͵"]
        for ch in artifacts:
            text = text.replace(ch, "")
        mapping = str.maketrans({"ي": "ی", "ك": "ک", "إ": "ا", "أ": "ا"})
        text = text.translate(mapping)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ---------- Ingest PDFs ----------
    def ingest_pdfs(self, data_dir="./data"):
        """
        Generate a response to user query using:
        - Few-shot examples
        - Chat history
        - RAG context from vector store (if exists)

        Args:
            query (str): User input question.
            chat_history (list): List of previous messages in format:
                [{"role": "user" or "assistant", "content": "..."}]

        Returns:
            str: LLM response text.
        """
        all_docs = []
        for file in os.listdir(data_dir):
            if file.lower().endswith(".pdf"):
                loader = PyPDFium2Loader(os.path.join(data_dir, file))
                docs = loader.load()
                for d in docs:
                    d.page_content = self.normalize_persian(d.page_content)
                all_docs.extend(docs)

        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = splitter.split_documents(all_docs)

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_directory
        )

        return len(chunks)

    # ---------- Main Response ----------
    def get_response(self, query: str, chat_history: list):
        """
        Generate a response to user query using:
        - Few-shot examples
        - Chat history
        - RAG context from vector store (if exists)

        Args:
            query (str): User input question.
            chat_history (list): List of previous messages in format:
                [{"role": "user" or "assistant", "content": "..."}]

        Returns:
            str: LLM response text.
        """

        # --- Few-shot examples ---
        few_shot_text = ""
        for ex in self.few_shot_examples:
            few_shot_text += f"کاربر: {ex['input']}\nدستیار: {ex['output']}\n"

        # --- User chat history ---
        history_text = ""
        for m in chat_history[-10:]:
            role = "کاربر" if m["role"] == "user" else "دستیار"
            history_text += f"{role}: {m['content']}\n"

        # --- Context from RAG ---
        context = ""
        if self.vector_store:
            docs = self.vector_store.similarity_search(query, k=4)
            context = "\n\n".join(d.page_content for d in docs)

        # --- Final prompt ---
        system_prompt = (
            "تو یک دستیار متخصص لینوکس و نرم‌افزار آزاد هستی.\n"
            "اگر پاسخ در مستندات نبود، از دانش خودت استفاده کن و شفاف بگو که در اسناد نبود.\n"
            "حتماً فارسی جواب بده.\n\n"
            "نمونه‌ها:\n{few_shot}\n"
            "تاریخچه گفتگو:\n{history}\n\n"
            "مستندات مرتبط:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        chain = prompt | self.llm

        return chain.invoke({
            "input": query,
            "few_shot": few_shot_text,
            "history": history_text,
            "context": context
        })
