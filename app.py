import streamlit as st
from langchain.llms import Together
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import pysqlite3
import sys
import os
from PyPDF2 import PdfReader

# Replace sqlite3 module with pysqlite3 for Chroma compatibility
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Function to extract text from the PDF
def extract_text_from_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Load the static dataset from the PDF
def load_static_data():
    pdf_path = "Pellet_mill.pdf"  # Corrected the filename to match your repository
    return extract_text_from_pdf(pdf_path)

def generate_response(query_text):
    documents = [load_static_data()]
    
    # Split documents into manageable chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.create_documents(documents)
    
    # Initialize Chroma vector store with a persistence directory
    db = Chroma.from_documents(
        texts,
        OpenAIEmbeddings(),
        persist_directory=".chroma_data"  # Directory for persistence
    )
    
    # Create retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})  # Return top 5 most relevant chunks
    
    # Create QA chain using Together AI
    llm = Together(api_key=os.getenv("TOGETHER_API_KEY"), model="together-gpt-jumbo")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    return qa.run(query_text)

# Streamlit page title and description
st.set_page_config(page_title='GPT Chatbot with Together AI')
st.title('📄 GPT Chatbot: Together AI')

# User input for query
query_text = st.text_input('Enter your question:', placeholder='Ask a specific question about the document.')

# API Key input for Together AI
together_api_key = st.text_input('Together API Key', type='password')

# Generate response when both inputs are provided
if st.button("Submit") and query_text and together_api_key:
    os.environ["TOGETHER_API_KEY"] = together_api_key  # Set Together API Key in the environment
    with st.spinner('Processing your request...'):
        try:
            response = generate_response(query_text)
            st.success(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
