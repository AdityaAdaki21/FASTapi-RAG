import os
from typing import List, Dict, Any
from huggingface_hub import InferenceClient
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

class LLMManager:
    """
    Manager class for handling different LLM and embedding models.
    Uses HuggingFace's InferenceClient directly for HuggingFace models.
    """
    def __init__(self, provider: str = "huggingface"):
        """
        Initialize the LLM Manager.
        
        Args:
            provider (str): The provider for LLM and embeddings. 
                           Options: "ollama", "huggingface", "huggingface-openai"
        """
        self.provider = provider
        self.llm_client = None
        self.embedding_client = None
        
        # Initialize models based on the provider
        if provider == "ollama":
            self._init_ollama()
        elif provider == "huggingface" or provider == "huggingface-openai":
            self._init_huggingface()
        else:
            raise ValueError(f"Unsupported provider: {provider}. Choose 'ollama', 'huggingface', or 'huggingface-openai'")
    
    def _init_ollama(self):
        """Initialize Ollama models."""
        self.llm = OllamaLLM(model="phi4-mini:3.8b")
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
        
    def _init_huggingface(self):
        """Initialize HuggingFace models using InferenceClient directly."""
        # Get API key from environment
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise ValueError("HuggingFace API key not found. Set HUGGINGFACE_API_KEY in environment variables.")
        
        llm_endpoint = "microsoft/Phi-3-mini-4k-instruct"
        embedding_endpoint = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Initialize InferenceClient for LLM
        self.llm_client = InferenceClient(
            model=llm_endpoint,
            token=api_key
        )
        
        # Initialize InferenceClient for embeddings - removed trust_remote_code parameter
        self.embedding_client = InferenceClient(
            model=embedding_endpoint,
            token=api_key
        )
        
        # Store generation parameters
        self.generation_kwargs = {
            "temperature": 0.7,
            "max_new_tokens": 1024,
            "repetition_penalty": 1.2,
            "do_sample": True
        }
    
    # LLM methods for compatibility with LangChain
    def get_llm(self):
        """
        Return a callable object that mimics LangChain LLM interface.
        For huggingface providers, this returns a function that calls the InferenceClient.
        """
        if self.provider == "ollama":
            return self.llm
        else:
            # Return a function that wraps the InferenceClient for LLM
            def llm_function(prompt, **kwargs):
                params = {**self.generation_kwargs, **kwargs}
                response = self.llm_client.text_generation(
                    prompt,
                    **params
                )
                return response
            
            # Add async capability
            async def allm_function(prompt, **kwargs):
                params = {**self.generation_kwargs, **kwargs}
                response = await self.llm_client.text_generation(
                    prompt,
                    **params,
                    stream=False
                )
                return response
            
            llm_function.ainvoke = allm_function
            return llm_function
    
    # Embeddings methods for compatibility with LangChain
    def get_embeddings(self):
        """
        Return a callable object that mimics LangChain Embeddings interface.
        For huggingface providers, this returns an object with embed_documents and embed_query methods.
        """
        if self.provider == "ollama":
            return self.embeddings
        else:
            # Create a wrapper object that has the expected methods
            class EmbeddingsWrapper:
                def __init__(self, client):
                    self.client = client
                
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    """Embed multiple documents."""
                    embeddings = []
                    # Process in batches to avoid overwhelming the API
                    batch_size = 8
                    
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i+batch_size]
                        batch_embeddings = self.client.feature_extraction(batch)
                        # Convert to standard Python list format
                        batch_results = [list(map(float, embedding)) for embedding in batch_embeddings]
                        embeddings.extend(batch_results)
                    
                    return embeddings
                
                def embed_query(self, text: str) -> List[float]:
                    """Embed a single query."""
                    embedding = self.client.feature_extraction(text)
                    if isinstance(embedding, list) and len(embedding) > 0:
                        # If it returns a batch (list of embeddings) for a single input
                        return list(map(float, embedding[0]))
                    # If it returns a single embedding
                    return list(map(float, embedding))
                
                # Make the class callable to fix the TypeError
                def __call__(self, texts):
                    """Make the object callable for compatibility with LangChain."""
                    if isinstance(texts, str):
                        return self.embed_query(texts)
                    elif isinstance(texts, list):
                        return self.embed_documents(texts)
                    else:
                        raise ValueError(f"Unsupported input type: {type(texts)}")
            
            return EmbeddingsWrapper(self.embedding_client)