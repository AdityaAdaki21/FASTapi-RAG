import os
import argparse
import logging
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple
from rich.console import Console
from rich.markdown import Markdown
from pinecone import Pinecone
from langchain_pinecone import Pinecone as LangchainPinecone

# Import our custom LLM Manager
from llm_manager import LLMManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

# Load environment variables
load_dotenv()

class F1AI:
    def __init__(self, index_name: str = "f12", llm_provider: str = "openrouter"):
        """
        Initialize the F1-AI RAG application.
        
        Args:
            index_name (str): Name of the Pinecone index to use
            llm_provider (str): Provider for LLM. "openrouter" is used by default.
        """
        self.index_name = index_name
        
        # Initialize LLM via manager
        self.llm_manager = LLMManager(provider=llm_provider)
        self.llm = self.llm_manager.get_llm()
        
        # Load Pinecone API Key
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("❌ Pinecone API key missing! Set PINECONE_API_KEY in environment variables.")

        # Initialize Pinecone with v2 client
        self.pc = Pinecone(api_key=pinecone_api_key)

        # Check existing indexes
        existing_indexes = [idx['name'] for idx in self.pc.list_indexes()]

        if index_name not in existing_indexes:
            raise ValueError(f"❌ Pinecone index '{index_name}' does not exist! Please create it first.")

        # Connect to Pinecone index
        index = self.pc.Index(index_name)
        
        # Use the existing pre-configured Pinecone index
        # Note: We're using the embeddings that Pinecone already has configured
        self.vectordb = LangchainPinecone(
            index=index,
            text_key="text",
            embedding=self.llm_manager.get_embeddings()  # This will only be used for new queries
        )

        print(f"✅ Successfully connected to Pinecone index: {index_name}")


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
        
        print("Starting embedding generation and uploading to Pinecone (this might take several minutes)...")
        # Use the existing vectordb to add documents
        self.vectordb.add_texts(
            texts=texts,
            metadatas=metadatas
        )
        
        print("✅ Documents successfully uploaded to Pinecone!")
    
    async def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question and get a response using RAG."""
        if not self.vectordb:
            return {"answer": "Error: Vector database not initialized. Please ingest data first.", "sources": []}
        
        try:
            # Retrieve relevant documents with similarity search
            retriever = self.vectordb.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Get relevant documents
            docs = retriever.get_relevant_documents(question)
            
            if not docs:
                return {
                    "answer": "I couldn't find any relevant information in my knowledge base. Please try a different question or ingest more relevant data.",
                    "sources": []
                }
            
            # Format context from documents
            context = "\n\n".join([f"Document {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])
            
            # Create prompt for the LLM
            prompt = f"""
            Answer the question based on the provided context. Include relevant citations using [1], [2], etc.
            If you're unsure or if the context doesn't contain the information, acknowledge the uncertainty.

            Context:
            {context}
            
            Question: {question}

            Answer with citations:
            """
            
            # Get response from LLM
            response_text = ""
            if hasattr(self.llm, "__call__"):  # Direct inference client wrapped function
                response_text = self.llm(prompt)
                # Debug response
                logger.info(f"Raw LLM response type: {type(response_text)}")
                if not response_text or response_text.strip() == "":
                    logger.error("Empty response from LLM")
                    response_text = "I apologize, but I couldn't generate a response. This might be due to an issue with the language model."
            else:  # LangChain LLM
                response_text = self.llm.invoke(prompt)
            
            # Format sources
            sources = [{
                "url": doc.metadata["source"],
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "timestamp": doc.metadata.get("timestamp", "")
            } for doc in docs]
            
            # Format response
            formatted_response = {
                "answer": response_text,
                "sources": sources
            }
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                "answer": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                "sources": []
            }

async def main():
    """Main function to run the application."""
    import asyncio
    
    parser = argparse.ArgumentParser(description="F1-AI: RAG Application for Formula 1 information")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest data from URLs")
    ingest_parser.add_argument("--urls", nargs="+", required=True, help="URLs to scrape")
    ingest_parser.add_argument("--max-chunks", type=int, default=100, help="Maximum chunks per URL")
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", help="Question to ask")
    
    # Provider argument
    parser.add_argument("--provider", choices=["ollama", "openrouter"], default="openrouter",
                        help="Provider for LLM (default: openrouter)")
    
    args = parser.parse_args()
    
    f1_ai = F1AI(llm_provider=args.provider)
    
    if args.command == "ingest":
        await f1_ai.ingest(args.urls, max_chunks_per_url=args.max_chunks)
    elif args.command == "ask":
        response = await f1_ai.ask_question(args.question)
        console.print("\n[bold green]Answer:[/bold green]")
        # Format as markdown to make it prettier
        console.print(Markdown(response['answer']))
        
        console.print("\n[bold yellow]Sources:[/bold yellow]")
        for i, source in enumerate(response['sources']):
            console.print(f"[{i+1}] {source['url']}")
    else:
        parser.print_help()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())