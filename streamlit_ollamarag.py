import streamlit as st
import requests
import pandas as pd
import time
import os
from io import BytesIO

# Set up the API URL
API_URL = "http://localhost:8000"  # Update this if your API is running elsewhere

# Set page configuration
st.set_page_config(
    page_title="OllamaRAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS for chatbot-like interface
st.markdown("""
<style>
    /* Main styles */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1E88E5;
    }
    
    /* Chat messages */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding-bottom: 5rem;
    }
    .user-message {
        background-color: #E3F2FD;
        color: #0D47A1;
        padding: 10px 15px;
        border-radius: 18px 18px 2px 18px;
        margin-left: 20%;
        margin-right: 10px;
        align-self: flex-end;
        max-width: 80%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .bot-message {
        background-color: #F5F5F5;
        color: #212121;
        padding: 10px 15px;
        border-radius: 18px 18px 18px 2px;
        margin-right: 20%;
        margin-left: 10px;
        align-self: flex-start;
        max-width: 80%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .message-time {
        font-size: 0.7rem;
        color: #9E9E9E;
        text-align: right;
        margin-top: 4px;
    }
    .bot-thinking {
        display: flex;
        align-items: center;
        background-color: #F5F5F5;
        color: #9E9E9E;
        padding: 10px 15px;
        border-radius: 18px;
        margin-right: 20%;
        margin-left: 10px;
        align-self: flex-start;
        max-width: fit-content;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Source citation */
    .source-citation {
        background-color: #BBDEFB;
        padding: 5px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 5px 0;
        display: inline-block;
    }
    
    /* Chat input area */
    .chat-input-area {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: white;
        padding: 1rem;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        z-index: 1000;
    }
    
    /* Button styles */
    .custom-button {
        background-color: #1E88E5;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .custom-button:hover {
        background-color: #1565C0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Settings area */
    .settings-container {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    /* Upload file area */
    .upload-area {
        border: 2px dashed #1E88E5;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        background-color: #E3F2FD;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .upload-area:hover {
        background-color: #BBDEFB;
    }
    
    /* Hide default Streamlit elements */
    div.block-container {
        padding-top: 1rem;
        padding-bottom: 4rem;
    }
    
    /* Typing animation */
    .typing-animation {
        display: inline-block;
        width: 20px;
        text-align: left;
    }
    .typing-dot {
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #9E9E9E;
        margin-right: 2px;
        animation: wave 1.3s linear infinite;
    }
    .typing-dot:nth-child(2) {
        animation-delay: -1.1s;
    }
    .typing-dot:nth-child(3) {
        animation-delay: -0.9s;
    }
    @keyframes wave {
        0%, 60%, 100% {
            transform: initial;
        }
        30% {
            transform: translateY(-5px);
        }
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def test_api_connection():
    try:
        response = requests.get(f"{API_URL}/test-connection")
        return response.json()["status"] == "connected"
    except:
        return False

def get_collections():
    try:
        response = requests.get(f"{API_URL}/collections")
        return response.json()["collections"]
    except:
        return []

def get_stats():
    try:
        response = requests.get(f"{API_URL}/stats")
        return response.json()
    except:
        return {}

def get_available_models():
    try:
        response = requests.get(f"{API_URL}/available-models")
        return response.json()
    except:
        return {"llm_models": [], "embedding_models": []}

def switch_collection(collection_name):
    try:
        response = requests.post(f"{API_URL}/switch-collection/{collection_name}")
        return response.json()["message"]
    except Exception as e:
        return f"Error: {str(e)}"

def delete_collection(collection_name):
    try:
        response = requests.delete(f"{API_URL}/collections/{collection_name}")
        return response.json()["message"]
    except Exception as e:
        return f"Error: {str(e)}"

def upload_pdf(pdf_file, chunk_size, chunk_overlap, max_workers):
    try:
        files = {"file": pdf_file}
        params = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "max_workers": max_workers
        }
        response = requests.post(f"{API_URL}/upload-pdf", files=files, params=params)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def query_pdf(question, collection=None, top_k=5):
    try:
        payload = {
            "query": question,
            "collection": collection,
            "top_k": top_k
        }
        response = requests.post(f"{API_URL}/query", json=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def update_config(llm_model, embedding_model, context_window):
    try:
        payload = {
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "context_window": context_window
        }
        response = requests.post(f"{API_URL}/config", json=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def format_time():
    return time.strftime("%I:%M %p")

# Initialize session state
if "collections" not in st.session_state:
    st.session_state.collections = []
if "current_collection" not in st.session_state:
    st.session_state.current_collection = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_connected" not in st.session_state:
    st.session_state.api_connected = test_api_connection()
if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = True
if "thinking" not in st.session_state:
    st.session_state.thinking = False

# Function to handle message submission
def handle_message_submit():
    if st.session_state.user_input and st.session_state.user_input.strip():
        user_message = st.session_state.user_input
        st.session_state.user_input = ""
        
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user", 
            "content": user_message,
            "time": format_time()
        })
        
        # Set thinking state to true
        st.session_state.thinking = True
        

# Main application layout
# Sidebar for settings and controls
with st.sidebar:
    st.markdown('<div class="sub-header">🤖 Chatbot Settings</div>', unsafe_allow_html=True)
    
    # API Connection Status
    if st.session_state.api_connected:
        st.success("✅ Connected to API")
    else:
        st.error("❌ Not connected to API")
        if st.button("🔄 Retry Connection"):
            st.session_state.api_connected = test_api_connection()

    
    # Collections Management
    st.markdown('<div class="sub-header">📚 Knowledge Base</div>', unsafe_allow_html=True)
    
    if st.session_state.api_connected:
        if st.button("🔄 Refresh Collections"):
            st.session_state.collections = get_collections()
        
        if not st.session_state.collections:
            st.session_state.collections = get_collections()
        
        if st.session_state.collections:
            collection_df = pd.DataFrame(st.session_state.collections)
            st.dataframe(collection_df, use_container_width=True)
            
            selected_collection = st.selectbox(
                "Select Collection",
                options=[coll["name"] for coll in st.session_state.collections],
                index=next((i for i, coll in enumerate(st.session_state.collections) if coll.get("active", False)), 0)
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Switch KB"):
                    with st.spinner(f"Switching to {selected_collection}..."):
                        result = switch_collection(selected_collection)
                        st.success(result)
                        st.session_state.current_collection = selected_collection
                        st.session_state.collections = get_collections()
                        
                        # Add system message about collection change
                        st.session_state.chat_history.append({
                            "role": "system",
                            "content": f"Switched to knowledge base: {selected_collection}",
                            "time": format_time()
                        })
            
            with col2:
                if st.button("🗑️ Delete KB"):
                    confirm = st.checkbox("Confirm deletion")
                    if confirm:
                        with st.spinner(f"Deleting {selected_collection}..."):
                            result = delete_collection(selected_collection)
                            st.success(result)
                            st.session_state.collections = get_collections()
                            if st.session_state.current_collection == selected_collection:
                                st.session_state.current_collection = None
        else:
            st.info("No knowledge bases available. Upload a PDF to create one.")
    
    # Advanced Settings
    with st.expander("⚙️ Advanced Settings"):
        if st.session_state.api_connected:
            models = get_available_models()
            stats = get_stats()
            
            with st.form("model_config"):
                llm_model = st.selectbox(
                    "LLM Model",
                    options=models["llm_models"],
                    index=models["llm_models"].index(stats.get("llm_model", models["llm_models"][0])) if stats.get("llm_model") in models["llm_models"] else 0
                )
                
                embedding_model = st.selectbox(
                    "Embedding Model",
                    options=models["embedding_models"],
                    index=models["embedding_models"].index(stats.get("embedding_model", models["embedding_models"][0])) if stats.get("embedding_model") in models["embedding_models"] else 0
                )
                
                context_window = st.number_input(
                    "Context Window",
                    min_value=1024,
                    max_value=32768,
                    value=stats.get("context_window", 4096),
                    step=1024
                )
                
                submit_config = st.form_submit_button("Update Configuration")
            
            if submit_config:
                with st.spinner("Updating configuration..."):
                    result = update_config(llm_model, embedding_model, context_window)
                    if "error" not in result:
                        st.success("Configuration updated successfully!")
                    else:
                        st.error(f"Error updating configuration: {result['error']}")
        
        # PDF Ingestion Settings
        st.markdown("#### PDF Processing Settings")
        top_k = st.slider("Number of relevant chunks", min_value=1, max_value=10, value=5)
        
        # Clear History Button
        if st.button("🧹 Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.show_welcome = True


# Main area - Chat interface
st.markdown('<div class="main-header">🤖 OllamaRAG Chatbot</div>', unsafe_allow_html=True)

# Check current collection
stats = get_stats()
current_collection = stats.get("current_collection", None)

if not current_collection:
    st.warning("No knowledge base selected. Please upload a PDF or select a collection from the sidebar.")

# Upload PDF tab in main area if no collection
if not current_collection:
    st.markdown('<div class="sub-header">📤 Upload a PDF to create your knowledge base</div>', unsafe_allow_html=True)
    
    if not st.session_state.api_connected:
        st.error("Cannot upload PDF: API not connected")
    else:
        with st.form("upload_form"):
            st.markdown('<div class="upload-area">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Drag and drop a PDF here", type="pdf")
            st.markdown('</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                chunk_size = st.number_input("Chunk Size", min_value=100, max_value=5000, value=1000, step=100)
            with col2:
                chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=chunk_size-1, value=min(500, chunk_size-1), step=50)
            with col3:
                max_workers = st.number_input("Max Workers", min_value=1, max_value=8, value=4, step=1)
            
            upload_button = st.form_submit_button("Upload PDF")
        
        if upload_button and uploaded_file is not None:
            with st.spinner("Uploading and processing PDF..."):
                # Save the uploaded file to a temporary file
                pdf_bytes = uploaded_file.read()
                # Call upload_pdf and capture the result
                result = upload_pdf(("temp.pdf", pdf_bytes), chunk_size, chunk_overlap, max_workers)
                
                if "error" not in result:
                    st.success(f"PDF uploaded successfully! Processing in background: {result.get('filename', 'unknown')}")
                    
                    # Add system message about upload
                    st.session_state.chat_history.append({
                        "role": "system",
                        "content": f"Uploaded PDF: {result.get('filename', 'unknown')}. Processing has started.",
                        "time": format_time()
                    })
                    
                    # Wait a bit and refresh collections
                    time.sleep(2)
                    st.session_state.collections = get_collections()
                else:
                    st.error(f"Error uploading PDF: {result['error']}")

# Chat messages area
chat_container = st.container()

with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Welcome message
    if st.session_state.show_welcome and not st.session_state.chat_history:
        welcome_message = "👋 Hello! I'm your PDF knowledge assistant. I can answer questions about your documents. Upload a PDF or select a knowledge base to get started!"
        st.markdown(f'<div class="bot-message">{welcome_message}<div class="message-time">{format_time()}</div></div>', unsafe_allow_html=True)
        st.session_state.show_welcome = False
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}<div class="message-time">{message.get("time", "")}</div></div>', unsafe_allow_html=True)
        elif message["role"] == "system":
            st.markdown(f'<div style="text-align: center; padding: 5px; color: #757575; font-size: 0.9rem; margin: 10px 0;"><i>{message["content"]}</i></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">{message["content"]}<div class="message-time">{message.get("time", "")}</div></div>', unsafe_allow_html=True)
            
            # Add expandable source chunks if available
            if "chunks" in message:
                with st.expander("View source information"):
                    for i, chunk in enumerate(message["chunks"]):
                        st.markdown(f'''
                        <div style="background-color: #F5F5F5; padding: 10px; border-radius: 8px; margin-bottom: 8px;">
                            <div style="font-weight: bold; color: #1E88E5;">Source {i+1}</div>
                            <div style="font-size: 0.8rem; color: #616161;">Relevance: {chunk["similarity"]:.2f}</div>
                            <div style="margin-top: 5px; font-size: 0.9rem;">{chunk["text"][:300]}...</div>
                        </div>
                        ''', unsafe_allow_html=True)
    
    # Show "thinking" animation
    if st.session_state.thinking:
        st.markdown(f'''
        <div class="bot-thinking">
            <span>Thinking</span>
            <div class="typing-animation">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Process thinking state and generate response
# In streamlit_ollamarag.py, modify the process thinking state section

# Process thinking state and generate response
if st.session_state.thinking and st.session_state.api_connected:
    # Get the last user message
    last_user_message = next((msg["content"] for msg in reversed(st.session_state.chat_history) if msg["role"] == "user"), None)
    
    if last_user_message:
        try:
            # Process and generate response
            with st.spinner("Processing your question..."):
                result = query_pdf(last_user_message, current_collection, top_k)
            
            # Add response to chat history
            if "error" not in result:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result.get("response", "I couldn't find an answer to that question."),
                    "chunks": result.get("relevant_chunks", []),
                    "time": format_time()
                })
            else:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Sorry, I encountered an error: {result.get('error', 'Unknown error')}",
                    "time": format_time()
                })
        except Exception as e:
            # Handle any exceptions during response generation
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Sorry, I encountered an unexpected error: {str(e)}",
                "time": format_time()
            })
    
    # Set thinking to False - make sure this always happens even if there's an error
    st.session_state.thinking = False
    st.rerun()  # Force a rerun to update the UI


# Chat input area
st.markdown('<div class="chat-input-area">', unsafe_allow_html=True)
col1, col2 = st.columns([6, 1])
with col1:
    st.text_input(
        "Type your message",
        key="user_input",
        on_change=handle_message_submit,
        placeholder="Ask me about your PDFs..."
    )
with col2:
    st.button("Send", on_click=handle_message_submit, key="send_button")
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown('<div style="text-align: center; color: #666;">OllamaRAG Chatbot | Powered by Ollama, ChromaDB, and FastAPI</div>', unsafe_allow_html=True)