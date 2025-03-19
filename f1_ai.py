import os
import argparse
from dotenv import load_dotenv
from typing import List, Dict, Any

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
        self.llm = OllamaLLM(model="llama3.2")
        
        # Initialize vector store if it exists
        if os.path.exists(vector_db_path):
            self.vectordb = Chroma(persist_directory=vector_db_path, embedding_function=self.embeddings)
        else:
            self.vectordb = None
            
    async def scrape(self, url: str, max_chunks: int = 100) -> List[Dict[str, Any]]:
        """Scrape content from a URL and split into chunks."""
        from playwright.async_api import async_playwright
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            print(f"Loading {url}...")
            await page.goto(url)
            text = await page.inner_text("body")
            await browser.close()
        
        print(f"Processing text ({len(text)} characters)...")
        # Clean text
        text = text.replace("\n", " ")
        
        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=100,
        )
        
        docs = splitter.create_documents([text])
        
        # Limit the number of chunks
        limited_docs = docs[:max_chunks]
        print(f"Using {len(limited_docs)} chunks out of {len(docs)} total chunks")
        
        return [{"page_content": doc.page_content, "metadata": {"source": url}} for doc in limited_docs]

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
    
    async def ask_question(self, question: str) -> str:
        """Ask a question and get a response using RAG."""
        if not self.vectordb:
            return "Error: Vector database not initialized. Please ingest data first."
        
        from langchain.chains import RetrievalQA
        from langchain_ollama import OllamaLLM
        
        # Retrieve relevant documents
        retriever = self.vectordb.as_retriever(search_kwargs={"k": 5})
        
        # Create RAG chain using cached LLM
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        # Get response
        response = await qa_chain.ainvoke({"query": question})
        
        return response["result"]

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