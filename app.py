import os
import streamlit as st
from rag import RAGChatbot
from datetime import datetime
from typing import List, Dict, Optional

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = RAGChatbot()
if 'selected_urls' not in st.session_state:
    st.session_state.selected_urls = []
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Store API keys securely (e.g., in environment variables) 
# VALID_API_KEYS = [os.environ.get("API_KEY_1"), os.environ.get("API_KEY_2")]
VALID_API_KEYS = ["1234567890", "0987654321"]

def authenticate():
    """
    Prompt the user to enter an API key and validate it against the list of valid API keys.
    """
    st.title("Authentication")
    api_key = st.text_input("Enter your API key:", type="password")
    
    if api_key in VALID_API_KEYS:
        st.success("Authentication successful!")
        return True
    elif api_key:
        st.error("Invalid API key. Please try again.")
    else:
        st.warning("Please enter your API key to access the app.")
    
    return False

def display_chat_message(role: str, content: str, urls: List[str] = None, timestamp: str = None):
    """Display a chat message with appropriate styling"""
    with st.chat_message(role):
        st.write(content)
        if urls:
            st.caption(f"Sources: {', '.join(urls)}")
        if timestamp:
            st.caption(f"Time: {timestamp}")

def main():
    # Check if the user is authenticated
    if not authenticate():
        return

    st.title("📚 Multi-URL Chatbot")
    
    # Sidebar for URL management
    with st.sidebar:
        st.header("URL Management")
        
        # URL input and processing
        url = st.text_input("Enter URL to analyze")
        process_button = st.button("Add URL")
        
        if process_button and url:
            with st.spinner('Processing URL...'):
                success, message = st.session_state.rag_system.process_url(url)
                
                if success:
                    if message == "URL already processed":
                        st.warning(message)
                    else:
                        st.success("URL processed successfully!")
                else:
                    st.error(f"Error processing URL: {message}")

        # URL selection
        st.subheader("Select URLs for Context")
        processed_urls = st.session_state.rag_system.get_processed_urls()
        if processed_urls:
            selected = st.multiselect(
                "Choose URLs to include in the conversation",
                processed_urls,
                default=processed_urls
            )
            st.session_state.selected_urls = selected

            # Remove URL button
            url_to_remove = st.selectbox("Select URL to remove", [""] + processed_urls)
            if st.button("Remove Selected URL") and url_to_remove:
                if st.session_state.rag_system.remove_url(url_to_remove):
                    st.success(f"Removed {url_to_remove}")
                    st.rerun()

    # Main chat interface
    st.header("Chat Interface")
    
    # Display number of loaded URLs
    num_urls = len(st.session_state.rag_system.get_processed_urls())
    st.info(f"📚 {num_urls} URLs loaded | 🎯 {len(st.session_state.selected_urls)} URLs selected for context")
    
    # Display chat messages from session state
    for message in st.session_state.messages:
        display_chat_message(
            message["role"],
            message["content"],
            message.get("urls"),
            message.get("timestamp")
        )
    
    # Chat input
    if question := st.chat_input("Ask a question about the documents..."):
        if not st.session_state.rag_system.is_initialized():
            st.error("Please add at least one URL first!")
        elif not st.session_state.selected_urls:
            st.warning("Please select at least one URL for context in the sidebar!")
        else:
            # Display user message
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.messages.append({
                "role": "user",
                "content": question,
                "timestamp": timestamp
            })
            display_chat_message("user", question, timestamp=timestamp)

            # Display assistant response with loading animation
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer, error = st.session_state.rag_system.get_answer(
                        question,
                        selected_urls=st.session_state.selected_urls
                    )
                    
                    if error:
                        error_message = f"Error generating answer: {error}"
                        st.error(error_message)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_message,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                    else:
                        st.write(answer)
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.caption(f"Sources: {', '.join(st.session_state.selected_urls)}")
                        st.caption(f"Time: {timestamp}")
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "urls": st.session_state.selected_urls,
                            "timestamp": timestamp
                        })

if __name__ == "__main__":
    main()