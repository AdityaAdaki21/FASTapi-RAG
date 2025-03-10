# FASTapi-RAG

## Overview
FASTapi-RAG is a Retrieval-Augmented Generation (RAG) system that allows users to query PDF documents using an AI-powered chatbot. It leverages Ollama for language generation and ChromaDB for document indexing and retrieval.

## Features
- Upload and process PDF documents.
- Query documents using natural language.
- Retrieve relevant document chunks based on similarity search.
- Supports multiple PDF collections.
- Interactive web interface with chat functionality.
- Configurable models for LLM and embedding.

## Project Structure
```
├── main.py                 # FastAPI backend
├── ollamarag.py            # Core logic for RAG system
├── embedding_functions.py  # Custom embedding functions
├── requirements.txt        # Python dependencies
├── static/
│   ├── index.html          # Web UI
```

## Installation

### Prerequisites
- Python 3.9
- Ollama installed and running (`ollama serve`)

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

### 3. Upload a PDF
Use the web interface or send a POST request to `/upload-pdf` endpoint.

### 4. Query the Document
Use the chat interface or send a POST request to `/query` with your question.

## API Endpoints

### Health Check
```http
GET /
```
Returns API status.

### Upload PDF
```http
POST /upload-pdf
```
Uploads a PDF for processing.

### Query Document
```http
POST /query
```
Queries the document and retrieves relevant responses.

## Configuration
You can change the LLM model, embedding model, and context window via the `/config` endpoint or the web interface.

## Dependencies
Refer to `requirements.txt`:

## Contributing
Feel free to open issues or pull requests to improve the project!