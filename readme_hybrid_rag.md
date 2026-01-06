# 📘 Hybrid RAG AI Study Companion

A powerful AI-powered study assistant that helps students understand their course materials by answering questions from uploaded PDF documents using **Hybrid Retrieval-Augmented Generation (RAG)**.

## 🌟 Features

- **Hybrid Search**: Combines BM25 (keyword-based) and semantic embeddings for superior retrieval
- **PDF Processing**: Upload and process any study material in PDF format
- **Azure OpenAI Integration**: Powered by GPT models for intelligent answers
- **Interactive UI**: Built with Streamlit for a smooth user experience
- **Context-Aware Answers**: Only answers based on uploaded documents

---

## 📋 Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [How It Works](#how-it-works)
4. [Code Explanation](#code-explanation)
5. [Usage](#usage)
6. [Architecture](#architecture)

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Azure OpenAI account with deployed chat model
- pip (Python package manager)

### Step 1: Clone or Download the Project

```bash
cd Study_Bot
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Package Breakdown:**
- `streamlit`: Web UI framework
- `nltk`: Natural Language Toolkit for text tokenization
- `numpy`: Numerical computing for array operations
- `rank-bm25`: BM25 algorithm implementation
- `langchain-community`: Document loaders and vector stores
- `langchain-text-splitters`: Text chunking utilities
- `faiss-cpu`: Facebook's vector similarity search
- `sentence-transformers`: Embedding models
- `openai`: Azure OpenAI client
- `python-dotenv`: Environment variable management

---

## ⚙️ Configuration

### Step 1: Create `.env` File

Create a file named `.env` in your project root:

```env
AZURE_OPENAI_API_KEY=your-azure-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini
```

### Step 2: Get Azure OpenAI Credentials

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to your Azure OpenAI resource
3. **Keys and Endpoint** section:
   - Copy **Key 1** → `AZURE_OPENAI_API_KEY`
   - Copy **Endpoint** → `AZURE_OPENAI_ENDPOINT`
4. **Deployments** section:
   - Note your chat model deployment name → `AZURE_OPENAI_CHAT_DEPLOYMENT`

### Step 3: Deploy Required Models

In Azure OpenAI Studio, deploy:
- **Chat Model**: gpt-4o-mini or gpt-35-turbo (for answering questions)

---

## 🔍 How It Works

### The Hybrid RAG Pipeline

```
PDF Upload → Text Extraction → Chunking → Dual Indexing → Hybrid Retrieval → Answer Generation
```

1. **Upload PDF**: User uploads study material
2. **Text Extraction**: PyPDFLoader extracts text from PDF
3. **Chunking**: Document split into 800-character chunks with 150-char overlap
4. **Dual Indexing**:
   - **BM25**: Keyword-based index (finds exact term matches)
   - **FAISS**: Semantic embeddings (understands meaning)
5. **Hybrid Retrieval**: Query searches both indices, merges results
6. **Answer Generation**: Azure OpenAI generates answer from retrieved context

### Why Hybrid RAG?

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| **BM25 Only** | Exact keyword matches, fast | Misses paraphrases, synonyms |
| **Embeddings Only** | Understands semantics | May miss specific terms |
| **Hybrid** | ✅ Best of both worlds | Slightly more complex |

---

## 📖 Code Explanation

### Line-by-Line Breakdown

#### **1. Imports Section**

```python
import streamlit as st
import nltk
import numpy as np
import os
from dotenv import load_dotenv
```

**What each import does:**
- `streamlit`: Creates the web interface
- `nltk`: Tokenizes text into words for BM25
- `numpy`: Handles array operations for scoring
- `os`: Reads environment variables
- `dotenv`: Loads `.env` file

```python
from rank_bm25 import BM25Okapi
```
- **BM25Okapi**: Implements BM25 ranking algorithm (like Google's old search)
- Uses term frequency and document frequency for scoring

```python
from langchain_community.document_loaders import PyPDFLoader
```
- **PyPDFLoader**: Extracts text from PDFs page-by-page
- Returns a list of `Document` objects (one per page)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```
- **RecursiveCharacterTextSplitter**: Intelligently splits text into chunks
- Tries to split at natural boundaries (paragraphs, sentences, words)

```python
from langchain_community.vectorstores import FAISS
```
- **FAISS**: Facebook AI Similarity Search
- Stores embeddings and finds nearest neighbors quickly (millions of vectors in milliseconds)

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
```
- **HuggingFaceEmbeddings**: Converts text to numerical vectors
- Uses pre-trained models (we use `all-MiniLM-L6-v2`)

```python
from openai import AzureOpenAI
```
- **AzureOpenAI**: Client to call Azure's GPT models

---

#### **2. Setup Section**

```python
load_dotenv()
```
- **Purpose**: Loads environment variables from `.env` file
- Makes `AZURE_OPENAI_API_KEY` accessible via `os.getenv()`

```python
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt")
```
- **Purpose**: Downloads NLTK's punkt tokenizer (needed for splitting words)
- Only downloads if not already present
- **Punkt**: Pre-trained model for sentence/word tokenization

```python
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-01"
)
```
- **Purpose**: Initializes Azure OpenAI client
- `api_key`: Authenticates your requests
- `azure_endpoint`: Your specific Azure resource URL
- `api_version`: Azure API version to use

```python
st.set_page_config(page_title="Hybrid RAG Study Companion", page_icon="📘")
```
- **Purpose**: Configures Streamlit page settings
- Sets browser tab title and favicon

---

#### **3. PDF Loading Function**

```python
def load_pdf(file_path):
    """Load PDF and return documents."""
    loader = PyPDFLoader(file_path)
    return loader.load()
```

**How it works:**
1. Creates a `PyPDFLoader` object with the PDF path
2. `.load()` extracts text from all pages
3. Returns list of `Document` objects:
   ```python
   [
       Document(page_content="Page 1 text...", metadata={"page": 0}),
       Document(page_content="Page 2 text...", metadata={"page": 1}),
       ...
   ]
   ```

---

#### **4. Chunking Function**

```python
def chunk_docs(docs):
    """Split documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)
```

**Parameters explained:**
- `chunk_size=800`: Each chunk is ~800 characters (about 1-2 paragraphs)
- `chunk_overlap=150`: Adjacent chunks share 150 characters

**Why chunking matters:**
- **Too large**: Embeddings lose specificity, harder to find relevant info
- **Too small**: Loses context, incomplete information
- **Overlap**: Ensures no information is lost at chunk boundaries

**Example:**
```
Original text: "Machine learning is... [800 chars] ...neural networks use..."
                                                    ↓
Chunk 1: "Machine learning is... [800 chars]"
                              ↑ 150 char overlap ↓
Chunk 2:              "...used in practice. [800 chars]"
```

---

#### **5. BM25 Index Building**

```python
def build_bm25(chunks):
    """Build BM25 index for keyword-based retrieval."""
    tokenized_chunks = [
        nltk.word_tokenize(chunk.page_content.lower())
        for chunk in chunks
    ]
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25, tokenized_chunks
```

**Step-by-step:**

1. **Tokenization**: Splits each chunk into words
   ```python
   "Machine learning is powerful" 
   → ['machine', 'learning', 'is', 'powerful']
   ```

2. **Lowercasing**: Ensures case-insensitive matching
   ```python
   "Machine" → "machine"
   ```

3. **BM25Okapi initialization**: Builds inverted index
   - Calculates term frequencies (TF)
   - Calculates inverse document frequencies (IDF)
   - Stores for fast scoring

**BM25 Formula (simplified):**
```
Score = Σ IDF(term) × (TF(term) × (k+1)) / (TF(term) + k)
```
- **IDF**: Rare terms score higher
- **TF**: More occurrences = higher score
- **k**: Dampening factor (prevents over-emphasis)

---

#### **6. BM25 Retrieval**

```python
def bm25_retrieve(bm25, tokenized_chunks, chunks, query, k=3):
    """Retrieve top-k chunks using BM25."""
    tokenized_query = nltk.word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    
    top_idx = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top_idx]
```

**Process:**

1. **Tokenize query**: `"neural networks"` → `['neural', 'networks']`

2. **Score all chunks**: BM25 calculates relevance score for each chunk
   ```python
   scores = [0.5, 2.3, 0.1, 1.8, 0.9, ...]
   ```

3. **Sort and select top-k**:
   ```python
   np.argsort(scores)      # [2, 0, 4, 3, 1, ...]  (indices sorted by score)
   [::-1]                   # [1, 3, 4, 0, 2, ...]  (reverse = highest first)
   [:k]                     # [1, 3, 4]              (top 3 indices)
   ```

4. **Return corresponding chunks**: Uses indices to retrieve original chunks

---

#### **7. FAISS Index Building**

```python
def build_faiss(chunks):
    """Build FAISS vector store for semantic retrieval."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.from_documents(chunks, embeddings)
    return db
```

**What happens:**

1. **Initialize embedding model**:
   - Downloads `all-MiniLM-L6-v2` (384-dimensional vectors)
   - Fast and accurate for semantic search

2. **Embed all chunks**:
   ```python
   "Machine learning algorithms" 
   → [0.12, -0.45, 0.78, ..., 0.33]  (384 numbers)
   ```

3. **Build FAISS index**:
   - Stores all vectors in efficient data structure
   - Creates index for fast nearest-neighbor search

**Embedding Model Details:**
- **all-MiniLM-L6-v2**: 
  - 384 dimensions (smaller = faster)
  - Trained on 1B+ sentence pairs
  - Understands semantic similarity

---

#### **8. Embedding Retrieval**

```python
def embedding_retrieve(db, query, k=3):
    """Retrieve top-k chunks using semantic similarity."""
    return db.similarity_search(query, k=k)
```

**Behind the scenes:**

1. **Query embedding**: Converts query to 384-dim vector

2. **Cosine similarity**: Measures angle between vectors
   ```
   similarity = (query_vec · chunk_vec) / (||query_vec|| × ||chunk_vec||)
   ```
   - 1.0 = identical meaning
   - 0.0 = unrelated
   - -1.0 = opposite meaning

3. **FAISS search**: Efficiently finds top-k nearest neighbors

4. **Returns chunks**: Ordered by semantic relevance

**Example:**
```
Query: "How do neural networks learn?"
Finds chunks containing:
- "backpropagation and gradient descent" ✓ (high similarity)
- "training deep learning models" ✓ (high similarity)
- "the history of computers" ✗ (low similarity)
```

---

#### **9. Hybrid Merge**

```python
def hybrid_retrieve(bm25_docs, embed_docs):
    """Merge BM25 and embedding results, removing duplicates."""
    seen = set()
    merged = []
    
    for doc in bm25_docs + embed_docs:
        if doc.page_content not in seen:
            merged.append(doc)
            seen.add(doc.page_content)
    
    return merged
```

**Algorithm:**

1. **Concatenate results**: `bm25_docs + embed_docs`

2. **Deduplication**:
   - Uses `set()` to track seen content
   - Adds document only if not already present
   - Preserves order (BM25 results first)

**Why this works:**
- BM25 might find: `[chunk_5, chunk_2, chunk_8]`
- Embeddings might find: `[chunk_2, chunk_9, chunk_1]`
- Merged result: `[chunk_5, chunk_2, chunk_8, chunk_9, chunk_1]` (chunk_2 not duplicated)

---

#### **10. Prompt Template**

```python
PROMPT = """You are a study assistant helping students understand their course material.

Answer ONLY using the provided context below.
If the answer is not present in the context, say "I cannot find this information in the uploaded document."

Context:
{context}

Question:
{question}

Provide a clear, concise answer with references to relevant sections if possible."""
```

**Purpose:** Structures the prompt for Azure OpenAI

**Key instructions:**
- `Answer ONLY using the provided context`: Prevents hallucinations
- `If the answer is not present`: Handles unanswerable questions
- `{context}` and `{question}`: Placeholders for dynamic content

---

#### **11. Answer Generation**

```python
def generate_answer(docs, question, deployment_name):
    """Generate answer using Azure OpenAI."""
    context = "\n\n".join([d.page_content for d in docs])
    
    prompt = PROMPT.format(
        context=context,
        question=question
    )
    
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"
```

**Step-by-step:**

1. **Build context**: Joins all retrieved chunks with double newlines
   ```python
   context = "Chunk 1 text...\n\nChunk 2 text...\n\nChunk 3 text..."
   ```

2. **Format prompt**: Inserts context and question into template

3. **API call**:
   - `model=deployment_name`: Your Azure deployment
   - `messages`: Chat format (user role)
   - `temperature=0`: Deterministic output (no creativity)

4. **Extract answer**: Gets text from first choice

**Temperature parameter:**
- `0`: Deterministic, factual (best for Q&A)
- `0.7`: Balanced creativity
- `1.0`: Maximum creativity (not recommended here)

---

#### **12. Streamlit UI**

```python
st.title("📘 Hybrid RAG AI Study Companion")
st.markdown("Upload your study material and ask questions!")
```
- **st.title()**: Large heading
- **st.markdown()**: Supports Markdown formatting

```python
with st.sidebar:
    st.header("⚙️ Settings")
    deployment_name = st.text_input(
        "Azure OpenAI Deployment Name",
        value=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini"),
        help="Enter your Azure OpenAI chat deployment name"
    )
    k_results = st.slider("Number of chunks to retrieve", 1, 10, 3)
```

**UI Components:**
- `st.sidebar`: Creates collapsible sidebar
- `st.text_input()`: Text field with default value
- `st.slider()`: Slider widget (min=1, max=10, default=3)

---

#### **13. Session State**

```python
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'bm25' not in st.session_state:
    st.session_state.bm25 = None
if 'tokenized_chunks' not in st.session_state:
    st.session_state.tokenized_chunks = None
if 'faiss_db' not in st.session_state:
    st.session_state.faiss_db = None
```

**Why session state?**
- Streamlit reruns the entire script on every interaction
- Session state persists data between reruns
- Prevents reprocessing PDF every time user asks a question

**Without session state:**
```
User uploads PDF → Process (30 seconds)
User asks question → Rerun script → Process again (30 seconds) ❌
```

**With session state:**
```
User uploads PDF → Process (30 seconds) → Store in session
User asks question → Rerun script → Use stored data (instant) ✓
```

---

#### **14. PDF Processing**

```python
if uploaded_file:
    if st.session_state.chunks is None:
        with st.spinner("Processing PDF..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            docs = load_pdf("temp.pdf")
            st.session_state.chunks = chunk_docs(docs)
            
            st.session_state.bm25, st.session_state.tokenized_chunks = build_bm25(st.session_state.chunks)
            st.session_state.faiss_db = build_faiss(st.session_state.chunks)
            
            st.success(f"✅ PDF indexed! Created {len(st.session_state.chunks)} chunks")
```

**Process flow:**

1. **Check if already processed**: `if st.session_state.chunks is None`

2. **Save uploaded file**: 
   - `uploaded_file.getbuffer()`: Gets file bytes
   - Writes to temporary file `temp.pdf`

3. **Process pipeline**:
   - Load PDF → Extract text
   - Chunk documents → Split into pieces
   - Build BM25 → Keyword index
   - Build FAISS → Semantic index

4. **Store in session**: Prevents reprocessing

5. **Show success message**: Informs user

---

#### **15. Question Answering**

```python
if question:
    with st.spinner("Searching for answer..."):
        bm25_docs = bm25_retrieve(
            st.session_state.bm25,
            st.session_state.tokenized_chunks,
            st.session_state.chunks,
            question,
            k=k_results
        )
        embed_docs = embedding_retrieve(st.session_state.faiss_db, question, k=k_results)
        
        final_docs = hybrid_retrieve(bm25_docs, embed_docs)
        
        answer = generate_answer(final_docs, question, deployment_name)
        
        st.subheader("💡 Answer")
        st.write(answer)
```

**Execution flow:**

1. **Dual retrieval**:
   - BM25: Keyword-based search
   - FAISS: Semantic search
   - Both retrieve `k_results` chunks (default: 3)

2. **Merge results**: Combine and deduplicate

3. **Generate answer**: Send to Azure OpenAI

4. **Display**: Show formatted answer

```python
with st.expander("📄 View Retrieved Context"):
    for i, doc in enumerate(final_docs, 1):
        st.markdown(f"**Chunk {i}:**")
        st.text(doc.page_content[:300] + "...")
        st.markdown("---")
```

**Expandable section:**
- Shows which chunks were used
- Helps users verify answer accuracy
- Educational: see what the AI "read"

---

## 🎯 Usage

### Running the Application

```bash
streamlit run app.py
```

Your browser will open to `http://localhost:8501`

### Step-by-Step Guide

1. **Upload PDF**: Click "Browse files" and select your study material
2. **Wait for processing**: App will show "✅ PDF indexed!"
3. **Ask questions**: Type your question in the text input
4. **View answer**: AI generates answer based on your document
5. **Check context**: Expand "View Retrieved Context" to see source chunks

### Example Questions

```
"What is the definition of neural networks?"
"Explain the backpropagation algorithm"
"What are the key differences between supervised and unsupervised learning?"
"Summarize Chapter 3"
```

---

## 🏗️ Architecture

### System Diagram

```
┌─────────────┐
│   PDF File  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  PyPDFLoader    │ ← Extracts text page-by-page
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ RecursiveTextSplitter   │ ← Chunks: 800 chars, 150 overlap
└───────────┬─────────────┘
            │
            ├─────────────────┬─────────────────┐
            ▼                 ▼                 ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ BM25 Index   │  │   Embeddings │  │  FAISS Index │
    │ (Keywords)   │  │   (Vectors)  │  │  (Semantic)  │
    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
           │                 │                  │
           └────────┬────────┴────────┬─────────┘
                    ▼                 ▼
              ┌──────────┐      ┌──────────┐
              │ BM25     │      │ Vector   │
              │ Search   │      │ Search   │
              └────┬─────┘      └────┬─────┘
                   │                 │
                   └────────┬────────┘
                            ▼
                   ┌─────────────────┐
                   │ Merge & Dedup   │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ Azure OpenAI    │ ← GPT generates answer
                   │   (GPT-4o)      │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  Final Answer   │
                   └─────────────────┘
```

### Data Flow

```python
# Input
user_question = "What is machine learning?"

# Step 1: BM25 Search
bm25_results = ["chunk_5", "chunk_12", "chunk_3"]  # Keyword matches

# Step 2: Semantic Search
embedding_results = ["chunk_12", "chunk_8", "chunk_1"]  # Meaning matches

# Step 3: Merge
merged = ["chunk_5", "chunk_12", "chunk_3", "chunk_8", "chunk_1"]  # Deduplicated

# Step 4: Generate Answer
context = join(merged)
prompt = f"Context: {context}\n\nQuestion: {user_question}"
answer = azure_openai.generate(prompt)

# Output
"Machine learning is a subset of artificial intelligence that..."
```

---

## 🔧 Customization

### Adjust Chunk Size

```python
def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Larger chunks = more context
        chunk_overlap=200     # More overlap = better continuity
    )
    return splitter.split_documents(docs)
```

### Change Embedding Model

```python
def build_faiss(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Multilingual
        # or "BAAI/bge-small-en-v1.5"  # Better accuracy
    )
    return FAISS.from_documents(chunks, embeddings)
```

### Modify Retrieval Count

In the sidebar, adjust `k_results` slider or change default:

```python
k_results = st.slider("Number of chunks to retrieve", 1, 10, 5)  # Default: 5
```

---

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'langchain.embeddings'`

**Solution:** Update import
```python
# Wrong
from langchain.embeddings import HuggingFaceEmbeddings

# Correct
from langchain_community.embeddings import HuggingFaceEmbeddings
```

### Issue: Azure OpenAI authentication error

**Solution:** Check `.env` file
```bash
# Verify values
echo %AZURE_OPENAI_API_KEY%        # Windows
echo $AZURE_OPENAI_API_KEY         # Linux/Mac
```

### Issue: NLTK data not found

**Solution:** Manual download
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

### Issue: Out of memory

**Solution:** Reduce chunk size or process fewer pages
```python
# Load only first 10 pages
docs = load_pdf("file.pdf")[:10]
```

---

## 📊 Performance Tips

### Optimization Strategies

1. **Cache embeddings**: Save FAISS index to disk
```python
# Save
vector_store.save_local("faiss_index")

# Load
vector_store = FAISS.load_local("faiss_index", embeddings)
```

2. **Batch processing**: Process multiple PDFs at once

3. **GPU acceleration**: Use `faiss-gpu` instead of `faiss-cpu`

4. **Smaller embedding models**: Faster but less accurate

---

## 📝 License

This project is open-source. Feel free to modify and distribute.

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Support for more file formats (DOCX, TXT, PPTX)
- Multi-language support
- Query history and favorites
- Export answers to PDF
- Fine-tuned embedding models

---

## 📧 Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review [Azure OpenAI documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
3. Check [LangChain documentation](https://python.langchain.com/)

---

## 🎓 Educational Use

This tool is designed for:
- ✅ Personal study and learning
- ✅ Understanding course materials
- ✅ Quick reference and lookup
- ❌ Not a replacement for reading/studying
- ❌ Not for academic dishonesty

**Remember:** Use AI as a study aid, not a substitute for learning!

---

**Built with ❤️ using LangChain, Streamlit, and Azure OpenAI**