from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
def build_faiss(chunks):
    """Build FAISS vector store for semantic retrieval."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.from_documents(chunks, embeddings)
    return db

def embedding_retrieve(db, query, k=3):
    """Retrieve top-k chunks using semantic similarity."""
    return db.similarity_search(query, k=k)

# -------------------- MERGE RESULTS --------------------
def hybrid_retrieve(bm25_docs, embed_docs):
    """Merge BM25 and embedding results, removing duplicates."""
    seen = set()
    merged = []
    
    for doc in bm25_docs + embed_docs:
        if doc.page_content not in seen:
            merged.append(doc)
            seen.add(doc.page_content)
    
    return merged
