# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import requests
import tempfile
import shutil
from ollamarag_core import OllamaRAG
from ollamarag_document import OllamaDocumentProcessor  # Import the correct class

app = FastAPI(
    title="OllamaRAG API",
    description="PDF-based RAG using Ollama and ChromaDB",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a global instance of OllamaRAG
rag = None

# Models for request and response
class QueryRequest(BaseModel):
    query: str
    collection: Optional[str] = None
    top_k: Optional[int] = 5

class IngestRequest(BaseModel):
    collection: Optional[str] = None
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 500
    max_workers: Optional[int] = 4

class ModelConfig(BaseModel):
    llm_model: str
    embedding_model: str
    context_window: int

# Initialize the RAG system
def init_rag(llm_model="tinyllama", embedding_model="mxbai-embed-large", context_window=2048):
    global rag
    rag = OllamaDocumentProcessor(llm_model=llm_model, embedding_model=embedding_model)
    rag.context_window = context_window
    rag.start_keep_alive_thread()
    return rag

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

@app.on_event("startup")
async def startup_event():
    init_rag()

@app.get("/")
async def root():
    return {"message": "OllamaRAG API is running. Visit /docs for documentation."}

@app.post("/query")
async def query(request: QueryRequest):
    if rag is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        # Switch collection if provided
        if request.collection:
            try:
                rag.get_or_create_collection(request.collection)
                rag.current_pdf = f"{request.collection}.pdf"
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"Collection not found: {str(e)}")
        
        # Generate response
        response = rag.generate_response(request.query, top_k=request.top_k)
        
        # Get relevant chunks for context
        relevant_chunks = rag.find_similar_chunks(request.query, top_k=request.top_k)
        
        return {
            "response": response,
            "relevant_chunks": relevant_chunks,
            "collection": rag.current_pdf.split('.')[0] if rag.current_pdf else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-pdf")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunk_size: int = Query(1000),
    chunk_overlap: int = Query(500),
    max_workers: int = Query(4)
):
    if rag is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    # Save uploaded file to a temporary location
    try:
        # Generate a temporary file path
        temp_file = f"uploads/{file.filename}"
        
        # Save the upload file
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get collection name from filename
        collection_name = os.path.splitext(file.filename)[0].replace(" ", "_")
        
        # Start ingestion process in the background
        background_tasks.add_task(
            rag.ingest_pdf,
            temp_file,
            chunk_size,
            chunk_overlap,
            max_workers
        )
        
        return {
            "message": f"PDF upload started. Processing in the background.",
            "filename": file.filename,
            "collection": collection_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")

@app.get("/collections")
async def list_collections():
    if rag is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        # list_collections now returns a list of names (strings)
        collection_names = rag.chroma_client.list_collections()
        collection_info = []
        
        for name in collection_names:
            try:
                # Retrieve the full collection object using the name
                coll = rag.chroma_client.get_collection(name=name, embedding_function=rag.ollama_ef)
                count = coll.count()
                active = rag.current_pdf and name == rag.current_pdf.split('.')[0].replace(" ", "_")
                collection_info.append({
                    "name": name,
                    "count": count,
                    "active": active
                })
            except Exception:
                collection_info.append({
                    "name": name,
                    "count": "Error",
                    "active": False
                })
        
        return {"collections": collection_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing collections: {str(e)}")

@app.post("/switch-collection/{collection_name}")
async def switch_collection(collection_name: str):
    if rag is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        # Check if collection exists
        collections = rag.chroma_client.list_collections()
        collection_names = [coll.name for coll in collections]
        
        if collection_name not in collection_names:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
        
        # Switch to the collection
        rag.get_or_create_collection(collection_name)
        
        # Update current_pdf
        rag.current_pdf = f"{collection_name}.pdf"
        
        # Get collection stats
        collection = rag.get_or_create_collection(collection_name)
        count = collection.count()
        
        return {
            "message": f"Switched to collection: {collection_name}",
            "count": count
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error switching collections: {str(e)}")

@app.get("/stats")
async def get_stats():
    if rag is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    stats = {
        "llm_model": rag.llm_model,
        "embedding_model": rag.embedding_model,
        "context_window": rag.context_window,
    }
    
    if rag.current_pdf:
        collection_name = rag.current_pdf.split('.')[0].replace(" ", "_")
        stats["current_collection"] = collection_name
        
        try:
            collection = rag.get_or_create_collection(collection_name)
            stats["document_count"] = collection.count()
            
            # Try to get source documents
            try:
                collection_metadata = collection.get()
                if collection_metadata and "metadatas" in collection_metadata:
                    sources = set()
                    for metadata in collection_metadata["metadatas"]:
                        if "source" in metadata:
                            sources.add(metadata["source"])
                    
                    stats["source_documents"] = len(sources)
            except Exception:
                pass
        except Exception as e:
            stats["error"] = str(e)
    
    return stats

@app.post("/config")
async def update_config(config: ModelConfig):
    try:
        global rag
        # Reinitialize the RAG system with new config
        rag = init_rag(
            llm_model=config.llm_model,
            embedding_model=config.embedding_model,
            context_window=config.context_window
        )
        
        return {
            "message": "Configuration updated successfully",
            "config": {
                "llm_model": rag.llm_model,
                "embedding_model": rag.embedding_model,
                "context_window": rag.context_window
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating configuration: {str(e)}")

# Add these endpoints to main.py

@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    if rag is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        success = rag.delete_collection(collection_name)
        if success:
            return {"message": f"Collection '{collection_name}' successfully deleted"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to delete collection '{collection_name}'")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}")

@app.get("/chunks/{collection_name}")
async def get_chunks(
    collection_name: str,
    limit: int = Query(10, description="Maximum number of chunks to return"),
    offset: int = Query(0, description="Number of chunks to skip")
):
    if rag is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        chunks = rag.get_chunks(collection_name, limit, offset)
        return {
            "collection": collection_name,
            "chunks": chunks,
            "count": len(chunks),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting chunks: {str(e)}")

# Add endpoint to get available Ollama models
@app.get("/available-models")
async def available_models():
    if rag is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        response = requests.get(f"{rag.base_url}/tags")
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch models from Ollama API")
        
        models_data = response.json()
        
        # Extract model names and categorize them
        llm_models = []
        embedding_models = []
        
        known_embedding_models = [
            "mxbai-embed", "nomic-embed", "all-minilm", "e5",
            "bge-", "gte-", "jina-", "mxbai-embed"
        ]
        
        for model in models_data.get("models", []):
            model_name = model.get("name", "").split(":")[0]
            
            # Check if it's likely an embedding model
            is_embedding = False
            for embedding_keyword in known_embedding_models:
                if embedding_keyword in model_name.lower():
                    is_embedding = True
                    break
            
            if is_embedding:
                embedding_models.append(model_name)
            else:
                llm_models.append(model_name)
        
        return {
            "llm_models": sorted(list(set(llm_models))),
            "embedding_models": sorted(list(set(embedding_models)))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching available models: {str(e)}")

# Add an endpoint to test connection to Ollama server
@app.get("/test-connection")
async def test_connection():
    try:
        base_url = "http://localhost:11434/api"
        response = requests.get(f"{base_url}/tags")
        
        if response.status_code == 200:
            return {"status": "connected", "message": "Successfully connected to Ollama server"}
        else:
            return {"status": "error", "message": f"Connected to server but got status code: {response.status_code}"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to connect to Ollama server: {str(e)}"}

# Add reindex endpoint to allow reprocessing documents with different settings
@app.post("/reindex/{collection_name}")
async def reindex_collection(
    collection_name: str,
    background_tasks: BackgroundTasks,
    request: IngestRequest
):
    if rag is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        # Find the corresponding PDF file
        pdf_file = None
        for filename in os.listdir("uploads"):
            if filename.endswith(".pdf") and os.path.splitext(filename)[0].replace(" ", "_") == collection_name:
                pdf_file = f"uploads/{filename}"
                break
        
        if not pdf_file:
            raise HTTPException(status_code=404, detail=f"Source PDF for collection '{collection_name}' not found")
        
        # Delete existing collection
        rag.delete_collection(collection_name)
        
        # Reingest with new settings
        chunk_size = request.chunk_size or 1000
        chunk_overlap = request.chunk_overlap or 500
        max_workers = request.max_workers or 4
        
        # Start ingestion process in the background
        background_tasks.add_task(
            rag.ingest_pdf,
            pdf_file,
            chunk_size,
            chunk_overlap,
            max_workers
        )
        
        return {
            "message": f"Reindexing started for collection '{collection_name}'",
            "settings": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "max_workers": max_workers
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reindexing collection: {str(e)}")
def handle_api_error(func):
    """Decorator for standardizing API error handling"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            endpoint = func.__name__
            raise HTTPException(
                status_code=500, 
                detail=f"Error in {endpoint}: {str(e)}"
            )
    return wrapper

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)