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
                try:
                    collection = self.chroma_client.get_collection(
                        name=collection_name,
                        embedding_function=self.ollama_ef  # Always specify embedding function
                    )
                    logger.info("Using existing collection with custom embeddings")
                    return collection
                except Exception:
                    # Collection doesn't exist, so create it
                    collection = self.chroma_client.create_collection(
                        name=collection_name,
                        embedding_function=self.ollama_ef  # Always specify embedding function
                    )
                    logger.info("Created collection with custom embeddings")
                    return collection
            except Exception as e:
                logger.error(f"Error accessing collection: {e}")
                raise e
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

    def _keep_models_alive(self) -> None:
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
    
    # Method to delete collections
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

    # Method to get specific chunks for debugging
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