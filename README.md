# F1-AI: Formula 1 RAG Application

F1-AI is a Retrieval-Augmented Generation (RAG) application specifically designed for Formula 1 information. It allows users to scrape Formula 1-related content from the web, store it in a vector database, and ask questions about the stored information using natural language.

## Features

![Example](image.png)

- Web scraping of Formula 1 content with automatic content extraction
- Vector database storage using Pinecone for efficient similarity search
- Multiple LLM provider options (HuggingFace, Ollama)
- RAG-powered question answering with contextual understanding
- Command-line interface for automation and scripting
- User-friendly Streamlit web interface
- Asynchronous data ingestion for improved performance

## Architecture

F1-AI is built on a modern tech stack:

- **LangChain**: Orchestrates the RAG pipeline and manages interactions between components
- **Pinecone**: Vector database for storing and retrieving embeddings
- **HuggingFace**: Primary LLM provider with Mixtral-8x7B-Instruct model
- **Ollama**: Alternative local LLM provider for text generation and embeddings
- **Playwright**: Handles web scraping with JavaScript support
- **Streamlit**: Provides the web interface

## Prerequisites

- Python 3.8 or higher
- HuggingFace API key (set as HUGGINGFACE_API_KEY environment variable)
- Pinecone API key (set as PINECONE_API_KEY environment variable)
- 8GB RAM minimum (16GB recommended)
- Internet connection for web scraping
- Ollama installed (optional, for local LLM support)
  - Download from [Ollama's website](https://ollama.ai)
  - Pull the required model: `ollama pull mixtral`

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd FASTapi-RAG
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Playwright browsers:
   ```bash
   playwright install
   ```

4. Set up environment variables:
   Create a .env file with:
   ```
   HUGGINGFACE_API_KEY=your_api_key_here  # Required for HuggingFace provider
   PINECONE_API_KEY=your_api_key_here      # Required for vector storage
   OLLAMA_HOST=http://localhost:11434 
   ```

## Usage

### Command Line Interface

1. Ingest data from URLs:
   ```bash
   python f1_ai.py ingest --urls <url1> <url2> --max-chunks 100 --provider huggingface
   ```
   Options:
   - `--urls`: Space-separated list of URLs to scrape
   - `--max-chunks`: Maximum number of text chunks to process (default: 100)
   - `--provider`: LLM provider to use (huggingface, ollama, huggingface-openai)

2. Ask questions about Formula 1:
   ```bash
   python f1_ai.py ask "Who won the 2023 F1 World Championship?" --provider huggingface
   ```
   Options:
   - `--provider`: Choose LLM provider (huggingface, ollama, huggingface-openai)

### Streamlit Interface

Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

This will open a web interface where you can:
- Input URLs for data ingestion
- Ask questions about Formula 1
- View the responses in a chat-like interface
- Monitor the ingestion progress
- View source citations for answers

## Project Structure

- `f1_ai.py`: Core RAG application implementation
  - Handles data ingestion, chunking, and embeddings
  - Manages vector database operations
  - Implements question-answering logic
- `llm_manager.py`: LLM provider management
  - Supports multiple LLM providers (HuggingFace, Ollama)
  - Handles embeddings generation
  - Manages API interactions
- `streamlit_app.py`: Streamlit web interface
  - Provides user-friendly UI
  - Manages async operations

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Submit a Pull Request
