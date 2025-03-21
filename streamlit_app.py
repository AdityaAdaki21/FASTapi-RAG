import streamlit as st
import asyncio
from f1_ai import F1AI

# Initialize session state
if 'f1_ai' not in st.session_state:
    st.session_state.f1_ai = F1AI()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Set page config
st.set_page_config(page_title="F1-AI: Formula 1 RAG Application", layout="wide")

# Title and description
st.title("F1-AI: Formula 1 RAG Application")

# Custom CSS for better styling
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stChatMessage.user {
        background-color: #f0f2f6;
    }
    .stChatMessage.assistant {
        background-color: #ffffff;
    }
    .source-link {
        font-size: 0.8rem;
        color: #666;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)

# Display chat history with enhanced formatting
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and isinstance(message["content"], dict):
            st.markdown(message["content"]["answer"])
            if message["content"]["sources"]:
                st.markdown("---")
                st.markdown("**Sources:**")
                for source in message["content"]["sources"]:
                    st.markdown(f"- [{source['url']}]({source['url']})")
        else:
            st.markdown(message["content"])

# Question input
if question := st.chat_input("Ask a question about Formula 1"):
    # Add user question to chat history
    st.session_state.chat_history.append({"role": "user", "content": question})
    
    # Display user question
    with st.chat_message("user"):
        st.write(question)
    
    # Generate and display response with enhanced formatting
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing Formula 1 knowledge..."):
            response = asyncio.run(st.session_state.f1_ai.ask_question(question))
            st.markdown(response["answer"])
            
            # Display sources if available
            if response["sources"]:
                st.markdown("---")
                st.markdown("**Sources:**")
                for source in response["sources"]:
                    st.markdown(f"- [{source['url']}]({source['url']})")
            
    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})