# Study Bot - Hybrid RAG AI Study Companion

A modern AI-powered study companion that uses Hybrid RAG (Retrieval-Augmented Generation) combining BM25 keyword search and semantic embeddings to help students understand their course material.

## 🏗️ Architecture

- **Frontend**: React with Vite
- **Backend**: FastAPI
- **AI**: Azure OpenAI with Hybrid RAG (BM25 + FAISS embeddings)

## 📁 Project Structure

```
Study_Bot/
├── backend/
│   ├── main.py              # FastAPI server
│   └── requirements.txt     # Backend dependencies
├── frontend/
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── services/        # API service layer
│   │   ├── App.jsx          # Main app component
│   │   └── main.jsx         # Entry point
│   ├── package.json
│   └── vite.config.js
├── ingestion/               # Data processing modules
│   ├── loader.py
│   ├── chunker.py
│   ├── embedder.py
│   └── bm25_indexer.py
└── rag/                     # RAG components
    ├── retriever.py
    └── prompt.py
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 16+
- Azure OpenAI API credentials

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your Azure OpenAI credentials:
```env
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint_here
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4.1
```

5. Run the FastAPI server:
```bash
# From the backend directory
python main.py
```

The backend will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

## 📖 Usage

1. **Start both servers**: Make sure both the FastAPI backend (port 8000) and React frontend (port 5173) are running.

2. **Upload a PDF**: Click the file upload area and select a PDF document with your study material.

3. **Configure settings** (optional): Adjust the Azure OpenAI deployment name and number of chunks to retrieve in the sidebar.

4. **Ask questions**: Type your question about the document and click "Ask" to get an AI-generated answer based on the retrieved context.

5. **View context**: Expand the "View Retrieved Context" section to see the document chunks that were used to generate the answer.

## 🔌 API Endpoints

### Backend API (FastAPI)

- `GET /` - Health check
- `GET /status` - Get application status
- `POST /upload` - Upload and process a PDF file
- `POST /ask` - Ask a question about the uploaded document
- `DELETE /reset` - Reset application state

### Example API Usage

```bash
# Upload a PDF
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"

# Ask a question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic?",
    "k_results": 3,
    "deployment_name": "gpt-4.1"
  }'
```

## 🎨 Features

- **Hybrid RAG**: Combines BM25 keyword search with semantic embeddings for optimal retrieval
- **Real-time processing**: Fast PDF processing and indexing
- **Interactive UI**: Modern, responsive React interface
- **Configurable**: Adjust retrieval parameters and AI model settings
- **Context viewing**: View the document chunks used to generate answers
- **CORS enabled**: Easy integration with other frontends

## 🛠️ Technology Stack

### Backend
- FastAPI
- LangChain
- FAISS (vector database)
- BM25 (keyword search)
- Azure OpenAI
- Sentence Transformers

### Frontend
- React 18
- Vite
- Axios
- CSS3

## 📝 Development

### Building for Production

**Frontend**:
```bash
cd frontend
npm run build
```

**Backend**:
The backend is production-ready but consider:
- Using a production ASGI server like Gunicorn with Uvicorn workers
- Implementing proper state management (Redis, database)
- Adding authentication and rate limiting

## 🤝 Contributing

Feel free to fork this project and submit pull requests!

## 📄 License

MIT License

## 🙏 Acknowledgments

- Azure OpenAI for powering the AI responses
- LangChain for the RAG framework
- FastAPI for the excellent Python web framework
- React and Vite for the modern frontend stack
