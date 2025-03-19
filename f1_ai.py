import os
import argparse
import logging
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple
from rich.console import Console
from rich.markdown import Markdown

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

# Load environment variables
load_dotenv()

class F1AI:
    def __init__(self, vector_db_path: str = "./vectordb"):
        """Initialize the F1-AI RAG application."""
        self.vector_db_path = vector_db_path
        
        # Import here to avoid loading modules if not used
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings, OllamaLLM
        
        # Initialize models with persistent instances
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
        self.llm = OllamaLLM(model="phi4-mini:3.8b")
        
        # Initialize vector store if it exists
        if os.path.exists(vector_db_path):
            self.vectordb = Chroma(persist_directory=vector_db_path, embedding_function=self.embeddings)
        else:
            self.vectordb = None
            
    async def scrape(self, url: str, max_chunks: int = 100) -> List[Dict[str, Any]]:
        """Scrape content from a URL and split into chunks with improved error handling."""
        from playwright.async_api import async_playwright, TimeoutError
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from bs4 import BeautifulSoup
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                console.log(f"[blue]Loading {url}...[/blue]")
                
                try:
                    await page.goto(url, timeout=30000)
                    # Get HTML content
                    html_content = await page.content()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Remove unwanted elements
                    for element in soup.find_all(['script', 'style', 'nav', 'footer']):
                        element.decompose()
                    
                    text = soup.get_text(separator=' ', strip=True)
                except TimeoutError:
                    logger.error(f"Timeout while loading {url}")
                    return []
                finally:
                    await browser.close()
            
            console.log(f"[green]Processing text ({len(text)} characters)...[/green]")
            
            # Enhanced text cleaning
            text = ' '.join(text.split())  # Normalize whitespace
            
            # Improved text splitting with semantic boundaries
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", "!", "?", ",", " "],
                length_function=len
            )
            
            docs = splitter.create_documents([text])
            
            # Limit the number of chunks
            limited_docs = docs[:max_chunks]
            console.log(f"[yellow]Using {len(limited_docs)} chunks out of {len(docs)} total chunks[/yellow]")
            
            # Enhanced metadata
            timestamp = datetime.now().isoformat()
            return [{
                "page_content": doc.page_content,
                "metadata": {
                    "source": url,
                    "chunk_index": i,
                    "total_chunks": len(limited_docs),
                    "timestamp": timestamp
                }
            } for i, doc in enumerate(limited_docs)]
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return []

    async def ingest(self, urls: List[str], max_chunks_per_url: int = 100) -> None:
        """Ingest data from URLs into the vector database."""
        from langchain_community.vectorstores import Chroma
        from tqdm import tqdm
        
        # Create empty list to store documents
        all_docs = []
        
        # Scrape and process each URL with progress bar
        for url in tqdm(urls, desc="Scraping URLs"):
            chunks = await self.scrape(url, max_chunks=max_chunks_per_url)
            all_docs.extend(chunks)
            
        # Create or update vector database
        total_docs = len(all_docs)
        print(f"\nCreating vector database with {total_docs} documents...")
        texts = [doc["page_content"] for doc in all_docs]
        metadatas = [doc["metadata"] for doc in all_docs]
        
        print("Starting embedding generation (this might take several minutes)...")
        self.vectordb = Chroma.from_texts(
            texts=texts, 
            embedding=self.embeddings,
            metadatas=metadatas,
            persist_directory=self.vector_db_path
        )
        
        # Persist the database
        print("Saving vector database to disk...")
        self.vectordb.persist()
        print("✅ Vector database created and persisted successfully!")
    
    async def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question and get a response using RAG with enhanced formatting and citations."""
        if not self.vectordb:
            return {"answer": "Error: Vector database not initialized. Please ingest data first.", "sources": []}
        
        from langchain.chains import RetrievalQA
        from langchain_ollama import OllamaLLM
        from langchain.prompts import PromptTemplate
        
        # Enhanced prompt template with source citation instructions
        template = """
        Answer the question based on the provided context. Include relevant citations.
        If you're unsure, acknowledge the uncertainty.
        
        Context: {context}
        Question: {question}
        
        Answer with citations:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        # Retrieve relevant documents with MMR reranking
        retriever = self.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 8,  # Fetch more documents for reranking
                "lambda_mult": 0.7  # Diversity factor
            }
        )
        
        # Create enhanced RAG chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt,
                "verbose": True
            }
        )
        
        try:
            # Get response
            response = await qa_chain.ainvoke({"query": question})
            
            # Format sources
            sources = [{
                "url": doc.metadata["source"],
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "timestamp": doc.metadata.get("timestamp", "")
            } for doc in response["source_documents"]]
            
            # Format response with markdown
            formatted_response = {
                "answer": response["result"],
                "sources": sources
            }
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error while processing your question. Please try again.",
                "sources": []
            }

async def main():
    """Main function to run the application."""
    import asyncio
    
    parser = argparse.ArgumentParser(description="F1-AI: RAG Application for Formula 1 information")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest command
    # In the main function, modify the ingest command to accept max_chunks
    ingest_parser = subparsers.add_parser("ingest", help="Ingest data from URLs")
    ingest_parser.add_argument("--urls", nargs="+", required=True, help="URLs to scrape")
    ingest_parser.add_argument("--max-chunks", type=int, default=100, help="Maximum chunks per URL")
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", help="Question to ask")
    
    args = parser.parse_args()
    
    f1_ai = F1AI()
    
    if args.command == "ingest":
        await f1_ai.ingest(args.urls, max_chunks_per_url=args.max_chunks)
    elif args.command == "ask":
        response = await f1_ai.ask_question(args.question)
        print(response)
    else:
        parser.print_help()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())