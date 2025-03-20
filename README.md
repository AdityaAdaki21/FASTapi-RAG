# F1-AI: Formula 1 RAG Application

F1-AI is a Retrieval-Augmented Generation (RAG) application specifically designed for Formula 1 information. It features an intelligent web scraper that automatically discovers and extracts Formula 1-related content from the web, stores it in a vector database, and enables natural language querying of the stored information.

## Features

![Example](image.png)

- Web scraping of Formula 1 content with automatic content extraction
- Vector database storage using Pinecone for efficient similarity search
- OpenRouter integration for advanced LLM capabilities
- RAG-powered question answering with contextual understanding and source citations
- Command-line interface for automation and scripting
- User-friendly Streamlit web interface with chat history
- Asynchronous data ingestion and processing for improved performance

## Architecture

F1-AI is built on a modern tech stack:

- **LangChain**: Orchestrates the RAG pipeline and manages interactions between components
- **Pinecone**: Vector database for storing and retrieving embeddings
- **OpenRouter**: Primary LLM provider with Mistral-7B-Instruct model
- **Ollama**: Alternative local LLM provider for embeddings
- **Playwright**: Handles web scraping with JavaScript support
- **BeautifulSoup4**: Processes HTML content and extracts relevant information
- **Streamlit**: Provides an interactive web interface with chat functionality

## Prerequisites

- Python 3.8 or higher
- OpenRouter API key (set as OPENROUTER_API_KEY environment variable)
- Pinecone API key (set as PINECONE_API_KEY environment variable)
- 8GB RAM minimum (16GB recommended)
- Internet connection for web scraping
- Ollama installed (optional, for local embeddings)
  - Download from [Ollama's website](https://ollama.ai)
  - Pull the required model: `ollama pull all-minilm-l6-v2`

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
   OPENROUTER_API_KEY=your_api_key_here    # Required for LLM functionality
   PINECONE_API_KEY=your_api_key_here      # Required for vector storage
   ```

## Usage

### Command Line Interface

1. Scrape and ingest F1 content:
   ```bash
   python f1_scraper.py --start-urls https://www.formula1.com/ --max-pages 100 --depth 2 --ingest
   ```
   Options:
   - `--start-urls`: Space-separated list of URLs to start crawling from
   - `--max-pages`: Maximum number of pages to crawl (default: 100)
   - `--depth`: Maximum crawl depth (default: 2)
   - `--ingest`: Flag to ingest discovered content into RAG system
   - `--max-chunks`: Maximum chunks per URL for ingestion (default: 50)
   - `--llm-provider`: Choose LLM provider (openrouter, ollama)

2. Ask questions about Formula 1:
   ```bash
   python f1_ai.py ask "Who won the 2023 F1 World Championship?"
   ```

### Streamlit Interface

Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

This will open a web interface where you can:
- Ask questions about Formula 1
- View responses in a chat-like interface
- See source citations for answers
- Track conversation history
- Get real-time updates on response generation

## Project Structure

- `f1_scraper.py`: Intelligent web crawler implementation
  - Automatically discovers F1-related content
  - Handles content relevance detection
  - Manages crawling depth and limits
- `f1_ai.py`: Core RAG application implementation
  - Handles data ingestion and chunking
  - Manages vector database operations
  - Implements question-answering logic
- `llm_manager.py`: LLM provider management
  - Integrates with OpenRouter for advanced LLM capabilities
  - Handles embeddings generation
  - Manages API interactions
- `streamlit_app.py`: Streamlit web interface
  - Provides chat-based UI
  - Manages conversation history
  - Handles async operations

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Submit a Pull Request