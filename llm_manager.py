import os
import json
import requests
from typing import List, Dict, Any
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class LLMManager:
    """
    Manager class for handling Ollama embeddings and OpenRouter LLM.
    """
    def __init__(self, provider: str = "openrouter"):
        """
        Initialize the LLM Manager.
        
        Args:
            provider (str): "ollama" for embeddings, OpenRouter is used for LLM regardless
        """
        self.provider = provider
        
        # Initialize Ollama embeddings
        self.embeddings = OllamaEmbeddings(model="tazarov/all-minilm-l6-v2-f32:latest")
        
        # Initialize OpenRouter client
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY in environment variables.")
        
        # Set up OpenRouter API details
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        self.openrouter_model = "mistralai/mistral-7b-instruct:free"
        self.openrouter_headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://f1-ai.app",  # Replace with your app's URL
            "X-Title": "F1-AI Application"  # Replace with your app's name
        }
    
    # LLM methods for compatibility with LangChain
    def get_llm(self):
        """
        Return a callable function that serves as the LLM interface.
        """
        def llm_function(prompt, **kwargs):
            try:
                logger.info(f"Sending prompt to OpenRouter (length: {len(prompt)})")
                
                # Format the messages for OpenRouter API
                messages = [{"role": "user", "content": prompt}]
                
                # Set up request payload
                payload = {
                    "model": self.openrouter_model,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 1024),
                    "top_p": kwargs.get("top_p", 0.9),
                    "stream": False
                }
                
                # Send request to OpenRouter
                response = requests.post(
                    self.openrouter_url,
                    headers=self.openrouter_headers,
                    json=payload,
                    timeout=60
                )
                
                # Process the response
                if response.status_code == 200:
                    response_json = response.json()
                    if "choices" in response_json and len(response_json["choices"]) > 0:
                        generated_text = response_json["choices"][0]["message"]["content"]
                        logger.info(f"Received response from OpenRouter (length: {len(generated_text)})")
                        return generated_text
                    else:
                        logger.warning("Unexpected response format from OpenRouter")
                        return "I couldn't generate a proper response based on the available information."
                else:
                    logger.error(f"Error from OpenRouter API: {response.status_code} - {response.text}")
                    return f"Error from LLM API: {response.status_code}"
                
            except Exception as e:
                logger.error(f"Error during LLM inference: {str(e)}")
                return f"Error generating response: {str(e)}"
        
        # Add async capability
        async def allm_function(prompt, **kwargs):
            import aiohttp
            
            try:
                # Format the messages for OpenRouter API
                messages = [{"role": "user", "content": prompt}]
                
                # Set up request payload
                payload = {
                    "model": self.openrouter_model,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 1024),
                    "top_p": kwargs.get("top_p", 0.9),
                    "stream": False
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.openrouter_url,
                        headers=self.openrouter_headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        if response.status == 200:
                            response_json = await response.json()
                            if "choices" in response_json and len(response_json["choices"]) > 0:
                                generated_text = response_json["choices"][0]["message"]["content"]
                                return generated_text
                            else:
                                logger.warning("Unexpected response format from OpenRouter")
                                return "I couldn't generate a proper response based on the available information."
                        else:
                            error_text = await response.text()
                            logger.error(f"Error from OpenRouter API: {response.status} - {error_text}")
                            return f"Error from LLM API: {response.status}"
                
            except Exception as e:
                logger.error(f"Error during async LLM inference: {str(e)}")
                return f"Error generating response: {str(e)}"
        
        # Add async method to the function
        llm_function.ainvoke = allm_function
        
        # Add invoke method for compatibility
        llm_function.invoke = llm_function
        
        return llm_function
    
    # Embeddings methods for compatibility with LangChain
    def get_embeddings(self):
        """Return the embeddings instance."""
        return self.embeddings