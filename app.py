import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import pysqlite3
import sys
import os

# Replace sqlite3 module with pysqlite3 for Chroma compatibility
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Load the static dataset from a text file in the repository
def load_static_data():
    with open("data.txt", "r", encoding="utf-8") as file:
        return file.read()

def generate_response(openai_api_key, query_text):
    documents = [load_static_data()]
    # Split documents into manageable chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents(documents)
    
    # Select embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    # Create a vectorstore from documents
    from chromadb.api.client.SharedSystemClient import clear_system_cache
    clear_system_cache()  # Ensure clean slate for Chroma DB
    db = Chroma.from_documents(texts, embeddings)
    
    # Create retriever interface
    retriever = db.as_retriever()
    
    # Create QA chain
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=openai_api_key), 
        chain_type='stuff', 
        retriever=retriever
    )
    return qa.run(query_text)

# Streamlit page title and description
st.set_page_config(page_title='GPT Chatbot with Static Data')
st.title('📄 GPT Chatbot: Static Data')

# User input for query
query_text = st.text_input('Enter your question:', placeholder='Ask a specific question about the document.')

# API Key input
openai_api_key = st.text_input('OpenAI API Key', type='password')

# Generate response when both inputs are provided
if st.button("Submit") and query_text and openai_api_key.startswith('sk-'):
    with st.spinner('Processing your request...'):
        try:
            response = generate_response(openai_api_key, query_text)
            st.success(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
