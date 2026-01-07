# Quick Setup Guide

## Step-by-Step Instructions

### 1. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
cd ..
```

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

> Requires Node.js 18+

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```env
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4.1
```

#### Frontend (.env in ./frontend)

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4.1
VITE_CHUNKS_TO_RETRIEVE=3
```

### 4. Run the Application

#### Option A: Use the startup script (Windows)
```bash
start.bat
```

#### Option B: Use the startup script (macOS/Linux)
```bash
chmod +x start.sh
./start.sh
```

#### Option C: Manual start

**Terminal 1 - Backend:**
```bash
cd backend
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

#### Option D: Frontend production build and preview
```bash
cd frontend
npm run build
npm run preview
# Preview runs on http://localhost:4173 by default
```

### 5. Access the Application

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Troubleshooting

### Backend Issues

**NLTK Data Error:**
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

**Module Import Error:**
- Make sure you're in the correct directory
- Verify virtual environment is activated
- Reinstall dependencies: `pip install -r backend/requirements.txt`

**Azure OpenAI Error:**
- Check your .env file credentials
- Verify deployment name matches your Azure resource
- Ensure API endpoint URL is correct

### Frontend Issues

**Node Modules Error:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

**CORS Error:**
- Ensure backend is running on port 8000
- Check CORS settings in backend/main.py

**Port Already in Use:**
- Change port in frontend/vite.config.js
- Or kill the process using the port

**Vite env not loaded:**
- Restart the dev server after changing `.env`
- Env keys must start with `VITE_`
- Ensure `.env` is inside `frontend/`

**Preview Port:**
- `npm run preview` uses port 4173 by default
- Change port via `vite.config.js` if needed

## First Time Usage

1. Upload a PDF document
2. Wait for processing (should take a few seconds)
3. Ask a question about the content
4. View the AI-generated answer and source context

## Configuration Options

### Frontend Settings (Sidebar)
- **Azure OpenAI Deployment**: Model deployment name (default: gpt-4.1)
- **Chunks to retrieve**: Number of context chunks (1-10, default: 3)

### Backend Configuration (backend/main.py)
- CORS origins
- Port number
- State management

Enjoy using your Study Bot! 🎓📚
