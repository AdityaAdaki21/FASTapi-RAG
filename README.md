# FASTapi-RAG

## Overview
FASTapi-RAG is a Retrieval-Augmented Generation (RAG) system that allows users to query PDF documents using an AI-powered chatbot. It leverages Ollama for language generation and ChromaDB for document indexing and retrieval.

## Features
- Upload and process PDF documents
- Query documents using natural language
- Retrieve relevant document chunks based on similarity search
- Supports multiple PDF collections
- Interactive web interface with chat functionality
- Configurable models for LLM and embedding

## Project Structure
```
├── main.py                  # FastAPI backend
├── ollamarag_core.py        # Core RAG system functionality 
├── ollamarag_document.py    # Document processing functionality
├── embedding_functions.py   # Custom embedding functions
├── requirements.txt         # Python dependencies
├── static/
│   ├── index.html           # Web UI
```

## Installation

### Prerequisites
- Python 3.9+
- Ollama installed and running (`ollama serve`)
- Required models pulled in Ollama (e.g., `ollama pull tinyllama` and `ollama pull mxbai-embed-large`)

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### 1. Start the API Server
```bash
python main.py
```
The server will run on `http://localhost:8000`.

### 2. Access the Web Interface
Open `http://localhost:8000/static/index.html` in your browser.

### 3. Common Workflow
1. Upload a PDF using the web interface
2. Wait for processing to complete (watch the terminal for progress)
3. Enter questions about your document in the chat interface
4. Switch between collections for different documents

## API Examples

### Health Check
```bash
curl http://localhost:8000/
```

### Upload PDF
```bash
curl -X POST -F "file=@/path/to/document.pdf" http://localhost:8000/upload-pdf
```

### Query Document
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "collection": "my_document", "top_k": 5}' \
  http://localhost:8000/query
```

### List Collections
```bash
curl http://localhost:8000/collections
```

### Switch Collection
```bash
curl -X POST http://localhost:8000/switch-collection/my_document
```

### Update Configuration
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"llm_model": "llama3", "embedding_model": "mxbai-embed-large", "context_window": 4096}' \
  http://localhost:8000/config
```

### Delete Collection
```bash
curl -X DELETE http://localhost:8000/collections/my_document
```

### Get Chunks
```bash
curl "http://localhost:8000/chunks/my_document?limit=10&offset=0"
```

### Available Models
```bash
curl http://localhost:8000/available-models
```

### Test Connection
```bash
curl http://localhost:8000/test-connection
```

### Reindex Collection
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"chunk_size": 1000, "chunk_overlap": 200, "max_workers": 4}' \
  http://localhost:8000/reindex/my_document
```

## Configuration Options

### LLM Models
- Any Ollama-compatible model (llama3, mistral, etc.)
- Default: "tinyllama"

### Embedding Models
- Supported models: mxbai-embed-large, nomic-embed, all-minilm, etc.
- Default: "mxbai-embed-large"

### Chunking Parameters
- `chunk_size`: Size of text chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 500)
- `max_workers`: Number of parallel workers for processing (default: 4)

## Troubleshooting
- Ensure Ollama is running with `ollama serve`
- Check required models are installed with `ollama list`
- For embedding issues, verify the embedding model exists in Ollama
- If no results are returned, try reindexing the collection with different parameters

## Dependencies
refer [requirement.txt](https://github.com/AdityaAdaki21/FASTapi-RAG/blob/6564ce3d43c765c8d8a90735d57f66b5a552741f/requirements.txt)

## Contributing
Feel free to open issues or pull requests to improve the project!
