import streamlit as st
import nltk
import numpy as np
import os
from dotenv import load_dotenv
from ingestion.loader import load_pdf
from ingestion.chunker import chunk_docs
from ingestion.embedder import build_faiss, embedding_retrieve, hybrid_retrieve
from ingestion.bm25_indexer import build_bm25, bm25_retrieve
from langchain_community.document_loaders import PyPDFLoader
from openai import AzureOpenAI
from rag.retriever import generate_answer
from rag.prompt import PROMPT
# -------------------- SETUP --------------------
load_dotenv()


# Download NLTK data (only first time)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

# Azure OpenAI Client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)
# -------------------- STREAMLIT APP --------------------

st.set_page_config(page_title="Hybrid RAG Study Companion", page_icon="📘")
st.title("📘 Hybrid RAG AI Study Companion")
st.markdown("Upload your study material and ask questions!")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    deployment_name = st.text_input(
        "Azure OpenAI Deployment Name",
        value=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1"),
        help="Enter your Azure OpenAI chat deployment name"
    )
    k_results = st.slider("Number of chunks to retrieve", 1, 10, 3)
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This app uses **Hybrid RAG** combining:")
    st.markdown("- 🔍 BM25 (keyword search)")
    st.markdown("- 🧠 Embeddings (semantic search)")

# Main content
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
question = st.text_input("Ask a question about your document")

# Initialize session state
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'bm25' not in st.session_state:
    st.session_state.bm25 = None
if 'tokenized_chunks' not in st.session_state:
    st.session_state.tokenized_chunks = None
if 'faiss_db' not in st.session_state:
    st.session_state.faiss_db = None

# Process uploaded PDF
if uploaded_file:
    # Only process if new file or not yet processed
    if st.session_state.chunks is None:
        with st.spinner("Processing PDF..."):
            # Save uploaded file temporarily
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load and process
            docs = load_pdf("temp.pdf")
            st.session_state.chunks = chunk_docs(docs)
            
            # Build indices
            st.session_state.bm25, st.session_state.tokenized_chunks = build_bm25(st.session_state.chunks)
            st.session_state.faiss_db = build_faiss(st.session_state.chunks)
            
            st.success(f"✅ PDF indexed! Created {len(st.session_state.chunks)} chunks using Hybrid RAG (BM25 + Embeddings)")
    
    # Answer questions
    if question:
        with st.spinner("Searching for answer..."):
            # Retrieve from both methods
            bm25_docs = bm25_retrieve(
                st.session_state.bm25,
                st.session_state.tokenized_chunks,
                st.session_state.chunks,
                question,
                k=k_results
            )
            embed_docs = embedding_retrieve(st.session_state.faiss_db, question, k=k_results)
            
            # Merge results
            final_docs = hybrid_retrieve(bm25_docs, embed_docs)
            
            # Generate answer
            answer = generate_answer(final_docs, question, PROMPT, client, deployment_name)
            
            # Display results
            st.subheader("💡 Answer")
            st.write(answer)
            
            # Show retrieved chunks (expandable)
            with st.expander("📄 View Retrieved Context"):
                for i, doc in enumerate(final_docs, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(doc.page_content[:300] + "...")
                    st.markdown("---")

# Footer
st.markdown("---")
st.markdown("*Powered by Azure OpenAI, LangChain, and Hybrid RAG*")