from rank_bm25 import BM25Okapi
import nltk
import numpy as np

def build_bm25(chunks):
    """Build BM25 index for keyword-based retrieval."""
    tokenized_chunks = [
        nltk.word_tokenize(chunk.page_content.lower())
        for chunk in chunks
    ]
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25, tokenized_chunks

def bm25_retrieve(bm25, tokenized_chunks, chunks, query, k=3):
    """Retrieve top-k chunks using BM25."""
    tokenized_query = nltk.word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    
    top_idx = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top_idx]