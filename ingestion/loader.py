from langchain_community.document_loaders import PyPDFLoader
def load_pdf(path):
    try:
        loader = PyPDFLoader(path)
        documents = loader.load()
        return documents
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found at: {path}")
    except Exception as e:
        raise Exception(f"Error loading PDF: {str(e)}")