# FASTapi-RAG

## Overview
FASTapi-RAG is a powerful Retrieval-Augmented Generation (RAG) system that enables intelligent document querying through an AI-powered chatbot. Built with FastAPI and leveraging Ollama for language generation and ChromaDB for efficient document indexing and retrieval, this system provides a seamless way to interact with PDF documents using natural language.

## Key Features
- **Smart Document Processing**: Upload and process PDF documents with automatic text extraction and semantic chunking
- **Natural Language Querying**: Ask questions about your documents in plain English
- **Advanced Retrieval**: Uses similarity search to find the most relevant document sections
- **Interactive UI**: Modern web interface with real-time chat functionality
- **Configurable AI Models**: Choose from various LLM and embedding models
- **RESTful API**: Comprehensive API for integration with other systems
- **Scalable Architecture**: Designed for efficient processing of large documents

## Architecture

### Components
- **FastAPI Backend** (`main.py`): RESTful API server handling all requests
- **RAG Core** (`ollamarag_core.py`): Core logic for document retrieval and AI interaction
- **Document Processor** (`ollamarag_document.py`): PDF processing and chunking logic
- **Embedding Handler** (`embedding_functions.py`): Custom embedding functions for document indexing
- **Streamlit Interface** (`streamlit_ollamarag.py`): Modern, interactive web UI for system interaction

### Data Flow
1. Document Upload → Text Extraction → Semantic Chunking → Vector Embedding → ChromaDB Storage
2. Query Input → Query Embedding → Similarity Search → Context Retrieval → LLM Processing → Response

## Installation

### System Requirements
- Python 3.9 or higher
- 8GB RAM minimum (16GB recommended for large documents)
- 2GB free disk space
- CUDA-compatible GPU (optional, for faster processing)

### Prerequisites
1. Install Python 3.9+
2. Install and start Ollama:
   ```bash
   # For Linux/macOS
   curl https://ollama.ai/install.sh | sh
   ollama serve

   # For Windows
   # Download from https://ollama.ai/download
   ```
3. Pull required models:
   ```bash
   ollama pull tinyllama
   ollama pull mxbai-embed-large
   ```

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/AdityaAdaki21/FASTapi-RAG.git
   cd FASTapi-RAG
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # For Windows
   .\venv\Scripts\activate
   # For Linux/macOS
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Guide

### Starting the System
1. Start the FastAPI backend server:
   ```bash
   python main.py
   ```
   API server runs at `http://localhost:8000`

2. Start the Streamlit interface in a new terminal:
   ```bash
   streamlit run streamlit_ollamarag.py
   ```
   The web interface will automatically open in your default browser

### Document Processing Workflow
1. **Upload Document**:
   - Use the web interface's upload button
   - Or use the API endpoint with cURL:
     ```bash
     curl -X POST -F "file=@/path/to/document.pdf" http://localhost:8000/upload-pdf
     ```

2. **Monitor Processing**:
   - Watch the terminal for progress indicators
   - Processing includes: text extraction, chunking, and embedding generation

3. **Query Documents**:
   - Enter questions in the chat interface
   - Or use the API endpoint:
     ```bash
     curl -X POST -H "Content-Type: application/json" \
       -d '{"query": "What is RAG?", "collection": "my_document", "top_k": 5}' \
       http://localhost:8000/query
     ```

4. **Manage Collections**:
   - Switch between documents using the collection selector
   - Delete unused collections when needed

## API Reference

### Document Management

#### Upload PDF
POST `/upload-pdf`
```bash
curl -X POST -F "file=@/path/to/document.pdf" http://localhost:8000/upload-pdf
```

#### List Collections
GET `/collections`
```bash
curl http://localhost:8000/collections
```

#### Switch Collection
POST `/switch-collection/{collection_name}`
```bash
curl -X POST http://localhost:8000/switch-collection/my_document
```

#### Delete Collection
DELETE `/collections/{collection_name}`
```bash
curl -X DELETE http://localhost:8000/collections/my_document
```

### Querying and Retrieval

#### Query Document
POST `/query`
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "collection": "my_document", "top_k": 5}' \
  http://localhost:8000/query
```

#### Get Chunks
GET `/chunks/{collection_name}`
```bash
curl "http://localhost:8000/chunks/my_document?limit=10&offset=0"
```

### System Management

#### Update Configuration
POST `/config`
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"llm_model": "llama3", "embedding_model": "mxbai-embed-large", "context_window": 4096}' \
  http://localhost:8000/config
```

#### Reindex Collection
POST `/reindex/{collection_name}`
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"chunk_size": 1000, "chunk_overlap": 200, "max_workers": 4}' \
  http://localhost:8000/reindex/my_document
```

#### Health Checks
- GET `/`: Basic health check
- GET `/test-connection`: Test Ollama connection
- GET `/available-models`: List available AI models

## Configuration

### Model Selection

#### LLM Models
- Supports any Ollama-compatible model
- Recommended models:
  - tinyllama (default): Fast, lightweight
  - llama3: Better quality, more resources
  - mistral: Good balance of speed and quality

#### Embedding Models
- Available options:
  - mxbai-embed-large (default): Best quality
  - nomic-embed: Faster processing
  - all-minilm: Lightweight option

### Performance Tuning

#### Chunking Parameters
- `chunk_size`: Text segment size (default: 1000)
  - Larger: Better context, slower processing
  - Smaller: Faster, might miss context
- `chunk_overlap`: Overlap between chunks (default: 500)
  - Larger: Better context preservation
  - Smaller: Less storage usage
- `max_workers`: Parallel processing threads (default: 4)
  - Adjust based on CPU cores available

#### Memory Usage
- Adjust `context_window` based on available RAM
- Default: 4096 tokens
- Reduce for memory-constrained systems

## Troubleshooting

### Common Issues

#### Server Won't Start
1. Check if Ollama is running:
   ```bash
   ollama serve
   ```
2. Verify port 8000 is available
3. Check Python version compatibility

#### Model Loading Errors
1. Verify models are installed:
   ```bash
   ollama list
   ```
2. Check model compatibility with Ollama version
3. Ensure sufficient system resources

#### Poor Query Results
1. Try reindexing the collection with different chunk parameters
2. Verify document processing completed successfully
3. Check if the query is well-formed

#### Performance Issues
1. Monitor system resources
2. Adjust worker count and chunk size
3. Consider using a lighter model

## Development

### Setting Up Dev Environment
1. Fork the repository
2. Install dev dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
3. Set up pre-commit hooks

### Code Style
- Document new features

## Dependencies
Refer to [requirements.txt](https://github.com/AdityaAdaki21/FASTapi-RAG/blob/main/requirements.txt) for a complete list of dependencies.

Key dependencies:
- fastapi: Web framework
- chromadb: Vector database
- PyPDF2: PDF processing
- pydantic: Data validation
- requests: HTTP client

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit changes with descriptive messages

## Support
- Create an issue for bugs or feature requests
