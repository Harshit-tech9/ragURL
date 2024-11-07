from langchain_community.document_loaders import UnstructuredURLLoader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama 
from langchain_groq import ChatGroq
from typing import List, Dict, Tuple, Optional
from datetime import datetime 

import getpass
import os

# os.environ["GROQ_API_KEY"] = getpass.getpass("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") 

api_key = os.getenv("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")

class RAGChatbot:
    def __init__(self):
        self.urls = {}  # Dictionary to store URL:vectordb mappings
        self.llm = None
        self.chat_history: List[Dict[str, str]] = []
        self._initialize_llm()
        
    def _initialize_llm(self):
        """Initialize the language model"""
        # self.llm = ChatOllama(
        #     model="llama3",
        #     temperature=0.8,
        #     num_predict=512
        # ) 

        self.llm = ChatGroq(
            model="llama3-8b-8192",
            temperature=0.8,
            # num_predict=512
        )

        
    def process_url(self, url: str) -> Tuple[bool, str]:
        """
        Process a URL and add it to the system.
        
        Args:
            url (str): URL to process
            
        Returns:
            tuple: (success_status, error_message if any)
        """
        try:
            if url in self.urls:
                return True, "URL already processed"
                
            # Load documents
            loader = UnstructuredURLLoader(urls=[url])
            documents = loader.load()

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)

            # Create embeddings
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': False}
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )

            # Create vector store
            vectordb = FAISS.from_documents(chunks, embedding=embeddings)
            
            # Store URL and its vectordb
            self.urls[url] = {
                'vectordb': vectordb,
                'added_at': datetime.now(),
                'chunks': chunks
            }
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    def get_answer(self, query: str, selected_urls: List[str] = None) -> Tuple[str, str]:
        """
        Generate answer for user query using selected URLs and chat history.
        
        Args:
            query (str): User question
            selected_urls (List[str]): List of URLs to consider for the answer
            
        Returns:
            tuple: (answer, error_message if any)
        """
        if not self.urls:
            return None, "No URLs processed. Please add at least one URL first."
            
        try:
            # Use all URLs if none specified
            urls_to_search = selected_urls if selected_urls else list(self.urls.keys())
            
            # Collect relevant documents from all selected URLs
            all_relevant_docs = []
            for url in urls_to_search:
                if url in self.urls:
                    vectordb = self.urls[url]['vectordb']
                    docs = vectordb.similarity_search(query, k=2)
                    all_relevant_docs.extend(docs)
            
            # Combine document contents with source URLs
            context_parts = []
            for doc in all_relevant_docs:
                source_url = next(url for url in urls_to_search if doc in self.urls[url]['chunks'])
                context_parts.append(f"From {source_url}:\n{doc.page_content}")
            context = "\n\n".join(context_parts)
            
            # Create chat history context
            chat_context = "\n".join([
                f"Human: {msg['human']}\nAssistant: {msg['assistant']}"
                for msg in self.chat_history[-3:]
            ])
            
            # Construct prompt with chat history and source information
            prompt = f"""Based on the following context and chat history, please provide a detailed answer to the question.
            
            Context from documents:
            {context}
            
            Previous conversation:
            {chat_context}
            
            Current question: {query}
            
            Please provide a thorough answer, include relevant information from the context."""

            # Generate response
            response = self.llm.call_as_llm(prompt)
            
            # Update chat history
            self.chat_history.append({
                "human": query,
                "assistant": response,
                "urls": urls_to_search,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            return response, None
            
        except Exception as e:
            return None, str(e)
    
    def get_processed_urls(self) -> List[str]:
        """Return list of processed URLs"""
        return list(self.urls.keys())
    
    def remove_url(self, url: str) -> bool:
        """Remove a URL and its associated data"""
        if url in self.urls:
            del self.urls[url]
            return True
        return False
    
    def is_initialized(self) -> bool:
        """Check if the system has any processed URLs"""
        return len(self.urls) > 0
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Return the chat history"""
        return self.chat_history
    
    def clear_chat_history(self):
        """Clear the chat history"""
        self.chat_history = []

# Essential: make sure the class is available for import
__all__ = ['RAGChatbot']