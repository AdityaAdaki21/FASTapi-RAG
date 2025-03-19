# F1-AI: Formula 1 RAG Application

F1-AI is a Retrieval-Augmented Generation (RAG) application specifically designed for Formula 1 information. It allows users to scrape Formula 1-related content from the web, store it in a vector database, and ask questions about the stored information using natural language.

## Features

- Web scraping of Formula 1 content
- Vector database storage using Chroma
- RAG-powered question answering
- Command-line interface
- Streamlit web interface

## Prerequisites

- Python 3.8 or higher
- Ollama installed and running locally

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

3. Make sure Ollama is running and the required models are available:
   - mxbai-embed-large (for embeddings)
   - llama3.2 (for text generation)

## Usage

### Command Line Interface

1. Ingest data from URLs:
   ```bash
   python f1_ai.py ingest --urls <url1> <url2> --max-chunks 100
   ```

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
- Input URLs for data ingestion
- Ask questions about Formula 1
- View the responses in a chat-like interface

## Project Structure

- `f1_ai.py`: Core RAG application implementation
- `streamlit_app.py`: Streamlit web interface
- `requirements.txt`: Project dependencies
- `vectordb/`: Directory containing the Chroma vector database

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.