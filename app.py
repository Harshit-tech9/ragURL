# import streamlit as st
# import time
# from langchain_community.document_loaders import UnstructuredURLLoader
# from sentence_transformers import SentenceTransformer
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_ollama import ChatOllama
# from langchain.prompts import PromptTemplate

# # Initialize session state variables
# if 'vectordb' not in st.session_state:
#     st.session_state.vectordb = None
# if 'llm' not in st.session_state:
#     st.session_state.llm = None

# def load_and_process_url(url):
#     """Load and process a single URL with progress indicators"""
#     # Load documents
#     with st.spinner('Loading document from URL...'):
#         loader = UnstructuredURLLoader(urls=[url])
#         documents = loader.load()

#     # Split documents
#     with st.spinner('Processing document...'):
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
#         chunks = text_splitter.split_documents(documents)

#     # Create embeddings
#     with st.spinner('Creating embeddings...'):
#         model_kwargs = {'device': 'cpu'}
#         encode_kwargs = {'normalize_embeddings': False}
#         embeddings = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-mpnet-base-v2",
#             model_kwargs=model_kwargs,
#             encode_kwargs=encode_kwargs
#         )

#     # Create vector store
#     with st.spinner('Creating vector store...'):
#         vectordb = FAISS.from_documents(chunks, embedding=embeddings)
#         st.session_state.vectordb = vectordb

#     # Initialize LLM
#     with st.spinner('Initializing AI model...'):
#         st.session_state.llm = ChatOllama(
#             model="llama3",
#             temperature=0.8,
#             num_predict=512
#         )

#     return True

# def get_answer(query):
#     """Generate answer for user query"""
#     if not st.session_state.vectordb or not st.session_state.llm:
#         st.error("Please process a URL first!")
#         return

#     with st.spinner('Searching for relevant information...'):
#         # Retrieve relevant documents
#         docs = st.session_state.vectordb.similarity_search(query)
        
#         # Combine document contents
#         context = "\n\n".join([doc.page_content for doc in docs])
        
#         # Construct prompt
#         prompt = f"""Based on the following context, please provide a detailed answer to the question.
        
#         Context: {context}
        
#         Question: {query}
        
#         Please provide a thorough answer and include relevant information from the context."""

#         # Generate response
#         response = st.session_state.llm.call_as_llm(prompt)
        
#         return response

# def main():
#     st.title("📚 URL-based Q&A System")
    
#     # URL input section
#     st.header("1. Enter URL")
#     url = st.text_input("Enter the URL of the document you want to analyze")
    
#     process_button = st.button("Process URL")
    
#     if process_button and url:
#         if load_and_process_url(url):
#             st.success("URL processed successfully! You can now ask questions about the content.")
    
#     # Question input section
#     st.header("2. Ask Questions")
#     question = st.text_input("Enter your question about the document")
    
#     ask_button = st.button("Ask Question")
    
#     if ask_button and question:
#         if st.session_state.vectordb is None:
#             st.error("Please process a URL first before asking questions!")
#         else:
#             answer = get_answer(question)
            
#             # Display the answer in a nice format
#             st.subheader("Answer")
#             st.write(answer)
            
#             # Add a divider for better visual separation
#             st.divider()

# if __name__ == "__main__":
#     main()


# # import streamlit as st

# # def main():
# #     st.title("Test App")
# #     st.write("If you can see this, Streamlit is working!")

# #     url = st.text_input("Enter a URL:")
# #     if url:
# #         st.write(f"You entered: {url}")

# # if __name__ == "__main__":
# #     main() 

# import streamlit as st
# from rag import RAGSystem

# # Initialize session state
# if 'rag_system' not in st.session_state:
#     st.session_state.rag_system = RAGSystem()

# def main():
#     st.title("📚 URL-based Q&A System")
    
#     # URL input section
#     st.header("1. Enter URL")
#     url = st.text_input("Enter the URL of the document you want to analyze")
    
#     process_button = st.button("Process URL")
    
#     if process_button and url:
#         with st.spinner('Processing URL...'):
#             success, error = st.session_state.rag_system.process_url(url)
            
#             if success:
#                 st.success("URL processed successfully! You can now ask questions about the content.")
#             else:
#                 st.error(f"Error processing URL: {error}")
    
#     # Question input section
#     st.header("2. Ask Questions")
#     question = st.text_input("Enter your question about the document")
    
#     ask_button = st.button("Ask Question")
    
#     if ask_button and question:
#         if not st.session_state.rag_system.is_initialized():
#             st.error("Please process a URL first before asking questions!")
#         else:
#             with st.spinner('Generating answer...'):
#                 answer, error = st.session_state.rag_system.get_answer(question)
                
#                 if error:
#                     st.error(f"Error generating answer: {error}")
#                 else:
#                     # Display the answer in a nice format
#                     st.subheader("Answer")
#                     st.write(answer)
                    
#                     # Add a divider for better visual separation
#                     st.divider()

# if __name__ == "__main__":
#     main()   

import streamlit as st
from rag import RAGChatbot
from datetime import datetime
from typing import List, Dict, Optional
import time

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = RAGChatbot()
if 'selected_urls' not in st.session_state:
    st.session_state.selected_urls = []
if 'messages' not in st.session_state:
    st.session_state.messages = []

def display_chat_message(role: str, content: str, urls: List[str] = None, timestamp: str = None):
    """Display a chat message with appropriate styling"""
    with st.chat_message(role):
        st.write(content)
        if urls:
            st.caption(f"Sources: {', '.join(urls)}")
        if timestamp:
            st.caption(f"Time: {timestamp}")

def main():
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
        
        # Chat controls
        if st.session_state.rag_system.is_initialized():
            st.subheader("Chat Controls")
            if st.button("Clear Chat History"):
                st.session_state.rag_system.clear_chat_history()
                st.session_state.messages = []
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