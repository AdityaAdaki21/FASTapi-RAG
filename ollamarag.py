# ollamarag.py
import os
import requests
import json
import PyPDF2
import numpy as np
from typing import List, Dict, Any, Tuple
import concurrent.futures
import time
import sys
import threading
import re
from datetime import datetime
import chromadb
import logging

from embedding_functions import OllamaEmbeddingFunction

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ollamarag")

class OllamaRAG:
    
    def __init__(self, llm_model="tinyllama", embedding_model="mxbai-embed-large"):
        """
        Initialize the RAG system with specified models and ChromaDB.
        
        Args:
            llm_model: The Ollama LLM model to use for generation
            embedding_model: The Ollama embedding model to use
        """
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.current_pdf = None
        self.model_loaded = False
        self.context_window = 4096  # Default context window size, adjust for your model
        self.base_url = "http://localhost:11434/api"
        
        # Create a directory for ChromaDB persistence
        os.makedirs("chroma_db", exist_ok=True)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        
        # Custom Ollama embedding function for ChromaDB
        self.ollama_ef = OllamaEmbeddingFunction(
            base_url=self.base_url,
            model_name=self.embedding_model
        )
        
        # Log initialization 
        logger.info(f"Initialized OllamaRAG with LLM model: {llm_model}, embedding model: {embedding_model}")
        
        # Verify that the models are available in Ollama
        self._verify_models()
        
        # Preload models
        self._preload_models()
    
    def _verify_models(self) -> None:
        """Verify that the required models are available."""
        logger.info("Checking available models...")
        try:
            response = requests.get(f"{self.base_url}/tags")
            available_models = [model["name"] for model in response.json()["models"]]
            
            missing_models = []
            if f"{self.llm_model}:latest" not in available_models:
                missing_models.append(f"{self.llm_model}:latest")
            if f"{self.embedding_model}:latest" not in available_models:
                missing_models.append(f"{self.embedding_model}:latest")
            
            if missing_models:
                logger.warning(f"The following models are not available: {', '.join(missing_models)}")
                logger.warning(f"Please pull these models using: ollama pull {' '.join(missing_models)}")
            else:
                logger.info("All required models are available")
        except Exception as e:
            logger.error(f"Error connecting to Ollama server: {e}")
            logger.warning("Make sure the Ollama server is running with 'ollama serve'")
            raise Exception(f"Cannot connect to Ollama server: {e}")
    
    def _preload_models(self) -> None:
        """Preload models into memory to keep them hot for fast inference."""
        logger.info("Preloading models into memory...")
        
        # Define a simple prompt for model loading
        warmup_prompt = "Hello, this is a warmup prompt to load the model into memory."
        
        # Preload embedding model
        try:
            logger.info(f"Loading embedding model '{self.embedding_model}'...")
            start_time = time.time()
            # Just do a single embedding to load the model
            _ = self.get_embedding("This is a test.")
            duration = time.time() - start_time
            logger.info(f"Embedding model loaded in {duration:.2f}s")
        except Exception as e:
            logger.warning(f"Failed to preload embedding model: {e}")
        
        # Preload LLM model
        try:
            logger.info(f"Loading LLM model '{self.llm_model}'...")
            start_time = time.time()
            # Send a simple prompt to load the model
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "model": self.llm_model,
                    "prompt": warmup_prompt,
                    "stream": False,
                    "keep_alive": "5m"  # Keep model loaded for 5 minutes
                }
            )
            
            # Check if we got a successful response
            if response.status_code == 200:
                duration = time.time() - start_time
                logger.info(f"LLM model loaded in {duration:.2f}s")
                self.model_loaded = True
            else:
                logger.warning(f"Failed to preload LLM model. Status code: {response.status_code}")
        except Exception as e:
            logger.warning(f"Failed to preload LLM model: {e}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embeddings for a piece of text.
        
        Args:
            text: The text to embed
            
        Returns:
            List of embedding values
        """
        response = requests.post(
            f"{self.base_url}/embeddings",
            json={
                "model": self.embedding_model,
                "prompt": text,
                "keep_alive": "5m"  # Keep model loaded for 5 minutes
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Error getting embeddings: {response.text}")
        
        return response.json()["embedding"]
    
    def get_or_create_collection(self, collection_name: str) -> chromadb.Collection:
        """
        Get a collection or create it if it doesn't exist.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            chromadb.Collection object
        """
        # First check if collection already exists
        try:
            collections = self.chroma_client.list_collections()
            collection_names = [coll.name for coll in collections]
            if collection_name in collection_names:
                logger.info(f"Using existing collection: {collection_name}")
                return self.chroma_client.get_collection(name=collection_name)
        except Exception as e:
            logger.debug(f"Error checking existing collections: {str(e)}")
        
        # If collection doesn't exist, create a new one
        logger.info(f"Creating fresh collection: {collection_name}")
        
        try:
            # Try to get the collection first
            try:
                collection = self.chroma_client.get_collection(name=collection_name)
                logger.info("Using existing collection")
                return collection
            except Exception:
                # Collection doesn't exist, so create it
                collection = self.chroma_client.create_collection(name=collection_name)
                logger.info("Created collection with default embeddings")
                return collection
        except Exception as e:
            logger.warning(f"Error creating collection with default embeddings: {e}")
            
            # Try with custom embedding function
            try:
                collection = self.chroma_client.create_collection(
                    name=collection_name,
                    embedding_function=self.ollama_ef
                )
                logger.info("Created collection with custom embeddings")
                return collection
            except Exception as e2:
                logger.warning(f"Error creating collection with custom embeddings: {e2}")
                
                # Last resort: try with no arguments
                try:
                    collection = self.chroma_client.create_collection(name=collection_name)
                    logger.info("Created basic collection")
                    return collection
                except Exception as e3:
                    logger.error("All collection creation methods failed!")
                    raise e3
                
    def _process_chunk(self, chunk_info: Dict) -> Dict:
        """Process a single chunk (for parallel processing)"""
        chunk, pdf_path, chunk_num, total_chunks = chunk_info["chunk"], chunk_info["pdf_path"], chunk_info["chunk_num"], chunk_info["total_chunks"]
        
        try:
            # We don't need to compute embeddings here, ChromaDB will handle it
            return {
                "text": chunk, 
                "source": pdf_path, 
                "id": f"{os.path.basename(pdf_path)}_chunk_{chunk_num}",
                "success": True
            }
        except Exception as e:
            return {
                "text": chunk, 
                "source": pdf_path, 
                "id": f"{os.path.basename(pdf_path)}_chunk_{chunk_num}",
                "success": False, 
                "error": str(e)
            }
    
    
    def ingest_pdf(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 500, 
                   max_workers: int = 4) -> None:
        """
        Ingest a PDF document, chunk it, and store in ChromaDB.
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            max_workers: Maximum number of parallel workers for processing
        """
        self.current_pdf = os.path.basename(pdf_path)
        collection_name = os.path.splitext(self.current_pdf)[0].replace(" ", "_")

        
        logger.info(f"Ingesting PDF: {pdf_path}")
        start_time = time.time()
        
        # Get or create a collection for this PDF
        collection = self.get_or_create_collection(collection_name)
        
        # Extract text from PDF
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                total_pages = len(reader.pages)
                
                logger.info(f"Extracting text from {total_pages} pages...")
                for i, page in enumerate(reader.pages):
                    text += page.extract_text() + " "
                    if (i + 1) % 5 == 0 or i + 1 == total_pages:
                        logger.info(f"Extracted {i + 1}/{total_pages} pages")
                
                logger.info("Text extraction complete")
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise Exception(f"Error reading PDF: {e}")
        
        # Create chunks with overlap
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) > 50:  # Only keep chunks with meaningful content
                chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from the PDF")
        
        # Prepare chunk information for parallel processing
        chunk_infos = [
            {"chunk": chunk, "pdf_path": pdf_path, "chunk_num": i+1, "total_chunks": len(chunks)}
            for i, chunk in enumerate(chunks)
        ]
        
        # Process chunks in parallel
        logger.info("Processing chunks in parallel...")
        successful_chunks = 0
        processed_chunks = []
        
        # Process chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_chunk, chunk_info): i for i, chunk_info in enumerate(chunk_infos)}
            
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                result = future.result()
                if result["success"]:
                    processed_chunks.append(result)
                    successful_chunks += 1
                
                if (i + 1) % 10 == 0 or i + 1 == len(chunks):
                    logger.info(f"Processed {i + 1}/{len(chunks)} chunks")
        
        # Add chunks to ChromaDB collection
        logger.info("Adding chunks to ChromaDB...")
        batch_size = 100  # ChromaDB recommends batching for better performance
        
        for i in range(0, len(processed_chunks), batch_size):
            batch = processed_chunks[i:i+batch_size]
            
            # Prepare batch for ChromaDB
            ids = [chunk["id"] for chunk in batch]
            texts = [chunk["text"] for chunk in batch]
            metadatas = [{"source": chunk["source"]} for chunk in batch]
            
            # Add to collection
            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Added batch {i//batch_size + 1}/{(len(processed_chunks) + batch_size - 1)//batch_size}")
        
        # Display summary
        total_time = time.time() - start_time
        
        logger.info(f"Ingestion complete:")
        logger.info(f"Collection name: {collection_name}")
        logger.info(f"Successful chunks: {successful_chunks}/{len(chunks)} ({successful_chunks/len(chunks)*100:.1f}%)")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Time per chunk: {total_time/len(chunks):.2f}s")
    
    def find_similar_chunks(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Find chunks most similar to the query using ChromaDB.
        
        Args:
            query: The query text
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing similar chunks and their metadata
        """
        if not self.current_pdf:
            return []
            
        collection_name = os.path.splitext(self.current_pdf)[0].replace(" ", "_")
        
        try:
            collection = self.get_or_create_collection(collection_name)
        except Exception as e:
            logger.error(f"Error accessing collection: {e}")
            return []
            
        logger.info(f"Finding relevant context (top {top_k}) for query: {query[:50]}...")
        
        # Query the collection
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted_results.append({
                    "text": doc,
                    "source": results["metadatas"][0][i]["source"],
                    "similarity": 1 - results["distances"][0][i]  # Convert distance to similarity
                })
        
        logger.info(f"Found {len(formatted_results)} relevant chunks")
        return formatted_results
    
    def _keep_models_alive(self) -> None:
        """Background thread to keep models loaded in memory."""
        while True:
            time.sleep(240)  # Send keep-alive every 4 minutes (since timeout is 5m)
            try:
                # Keep LLM alive
                response = requests.post(
                    f"{self.base_url}/generate",
                    json={
                        "model": self.llm_model,
                        "prompt": "keep alive",
                        "stream": False,
                        "keep_alive": "5m"
                    }
                )
                
                # Keep embedding model alive
                response = requests.post(
                    f"{self.base_url}/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": "keep alive",
                        "keep_alive": "5m"
                    }
                )
                logger.debug("Sent keep-alive signals to models")
            except Exception:
                # Silently ignore errors during keep-alive
                pass
    
    def start_keep_alive_thread(self) -> None:
        """Start a background thread to keep models loaded."""
        keep_alive_thread = threading.Thread(target=self._keep_models_alive, daemon=True)
        keep_alive_thread.start()
        logger.info("Started background thread to keep models loaded")
    
    def generate_response(self, query: str, system_prompt: str = None, top_k: int = 5) -> str:
        """
        Generate a response to the query using the LLM and retrieved context.
        
        Args:
            query: The user query
            system_prompt: Optional system prompt to guide the LLM
            top_k: Number of relevant chunks to include in context
            
        Returns:
            The generated response
        """
        # Find relevant chunks with increased top_k for better context
        relevant_chunks = self.find_similar_chunks(query, top_k=top_k)
        
        if not relevant_chunks:
            return "No relevant information found. Please ingest a PDF document first."
        
        # Sort chunks by similarity score in descending order
        relevant_chunks.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Create context from retrieved chunks with chunk markers
        context_parts = []
        for i, chunk in enumerate(relevant_chunks):
            chunk_text = chunk["text"].strip()
            # Add chunk markers to help the model distinguish between sources
            context_parts.append(f"[Document {i+1}] {chunk_text}")
        
        context = "\n\n".join(context_parts)
        
        # Improved system prompt with better instructions
        if not system_prompt:
            system_prompt = (
                "You are a precise and helpful research assistant. Your task is to answer the user's question "
                "based solely on the provided context information. Follow these guidelines:\n"
                "1. Only use information explicitly stated in the context.\n"
                "2. If the context doesn't contain the answer, say 'Based on the provided information, I cannot answer this question.'\n"
                "3. Do not introduce external knowledge or assumptions.\n"
                "4. If information is partially available, provide what you can find and acknowledge the limitations.\n"
                "5. If the user asks for a specific format (list, table, etc.), follow that format.\n"
                "6. Be concise but thorough. Aim for comprehensive accuracy.\n"
                "7. If the context contains conflicting information, acknowledge the conflict and present both perspectives.\n"
                "8. Cite the document numbers ([Document X]) when referring to specific information."
            )
        
        # Improved prompt template with query-focused instructions
        prompt = f"""## Context Information
{context}

## Question
{query}

## Instructions
- Answer the question thoroughly based ONLY on the context provided above.
- If you need to make an inference from the context, clearly indicate it as such.
- Format your answer clearly and logically.
- Include relevant details from the context to support your answer.
- Cite specific document sections when appropriate using [Document X] notation.
"""
        
        # Ensure we don't exceed the context window
        if len(prompt) > self.context_window:
            # More strategic truncation - keep high similarity chunks intact
            current_length = len(prompt)
            target_length = self.context_window - 500  # Leave room for other parts
            
            # Start removing lower-ranked chunks until we fit
            while current_length > target_length and len(context_parts) > 1:
                # Remove the last (lowest similarity) chunk
                removed_chunk = context_parts.pop()
                # Regenerate the context
                context = "\n\n".join(context_parts)
                # Recalculate the prompt
                prompt = f"""## Context Information
{context}

## Question
{query}

## Instructions
- Answer the question thoroughly based ONLY on the context provided above.
- If you need to make an inference from the context, clearly indicate it as such.
- Format your answer clearly and logically.
- Include relevant details from the context to support your answer.
- Cite specific document sections when appropriate using [Document X] notation.

Note: Some less relevant context was removed to fit within the model's context window.
"""
                current_length = len(prompt)
        
        start_time = time.time()
        
        # Send the request to Ollama with temperature control
        logger.info("Generating response with LLM...")
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "model": self.llm_model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "temperature": 0.2,  # Lower temperature for more factual responses
                "top_p": 0.9,        # Better quality generation
                "keep_alive": "5m"   # Keep model loaded for 5 minutes
            }
        )
        
        if response.status_code != 200:
            error_msg = f"Error generating response: {response.text}"
            logger.error(error_msg)
            return error_msg
        
        response_text = response.json()["response"]
        
        # Post-process the response
        response_text = self._post_process_response(response_text)
        
        # Record stats
        total_time = time.time() - start_time
        
        # Log retrieval statistics
        logger.info(f"Response generated in {total_time:.2f} seconds")
        for i, chunk in enumerate(relevant_chunks[:3]):  # Log top 3 chunks
            preview = chunk["text"][:50].replace("\n", " ") + "..."
            logger.info(f"Chunk {i+1}: Similarity {chunk['similarity']:.3f} - {preview}")
        
        return response_text

    # Fix for the batch_size variable in the ingest_pdf method
    # Replace the line with:
    batch_size = 100  # ChromaDB recommends batching for better performance

    # Completed _post_process_response method
    def _post_process_response(self, text: str) -> str:
        """
        Post-process the model's response for better formatting and clarity.
        
        Args:
            text: The raw response text from the LLM
            
        Returns:
            Processed response text
        """
        # Remove any preamble like "Based on the context provided,"
        text = re.sub(r"^(Based on (the|your) (context|information|document|documents) provided,?\s*)", "", text)
        
        # Remove redundant document citations if they're overly repetitive
        doc_citation_count = len(re.findall(r"\[Document \d+\]", text))
        if doc_citation_count > 10:
            # If there are too many citations, clean them up
            text = re.sub(r"\s*\[Document \d+\]\s*", " ", text)
        
        # Fix any double spaces
        text = re.sub(r'\s{2,}', ' ', text)
        
        # Ensure proper paragraph breaks
        text = re.sub(r'(\.\s)([A-Z])', r'.\n\n\2', text)
        
        # Format lists properly
        text = re.sub(r'(\d+\.\s*[^\n]+)(?=\s+\d+\.)', r'\1\n', text)
        
        return text.strip()

    # Add a method to delete collections
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from ChromaDB.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            Boolean indicating success
        """
        try:
            logger.info(f"Attempting to delete collection: {collection_name}")
            self.chroma_client.delete_collection(collection_name)
            logger.info(f"Successfully deleted collection: {collection_name}")
            
            # If this was the current collection, reset current_pdf
            if self.current_pdf and os.path.splitext(self.current_pdf)[0].replace(" ", "_") == collection_name:
                self.current_pdf = None
                logger.info("Reset current_pdf as the active collection was deleted")
                
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            return False

    # Add a method to get specific chunks for debugging
    def get_chunks(self, collection_name: str, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get chunks from a specific collection for inspection.
        
        Args:
            collection_name: Name of the collection
            limit: Maximum number of chunks to return
            offset: Number of chunks to skip
            
        Returns:
            List of chunk documents with their IDs and metadata
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            result = collection.get(limit=limit, offset=offset)
            
            chunks = []
            if result and "documents" in result:
                for i, doc in enumerate(result["documents"]):
                    chunk_data = {
                        "id": result["ids"][i],
                        "text": doc,
                    }
                    
                    if "metadatas" in result and i < len(result["metadatas"]):
                        chunk_data["metadata"] = result["metadatas"][i]
                        
                    chunks.append(chunk_data)
                    
            return chunks
        except Exception as e:
            logger.error(f"Error getting chunks from collection {collection_name}: {e}")
            return []