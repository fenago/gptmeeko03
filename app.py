import streamlit as st
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from chromadb import Client
from chromadb.config import Settings
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
import os
import shutil

# Access the OpenAI API key from environment variables or Streamlit secrets
YOUR_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", st.secrets.get("YOUR_OPENAI_API_KEY", ""))
if not YOUR_OPENAI_API_KEY:
    raise ValueError("OpenAI API Key is not set. Please add it to the environment variables or Streamlit secrets.")

# Function to extract text from the PDF
def extract_text_from_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to clean up Chroma data
def clean_chroma_data():
    chroma_directory = ".chroma_data"
    if os.path.exists(chroma_directory):
        shutil.rmtree(chroma_directory)

# Load the static dataset from the PDF
def load_static_data():
    pdf_path = "Pellet_mill.pdf"  # Ensure this matches your repository's filename
    return extract_text_from_pdf(pdf_path)

def generate_response(query_text):
    # Clean up existing Chroma data to avoid conflicts
    clean_chroma_data()

    # Load and preprocess data
    document_text = load_static_data()

    # Split documents into manageable chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(document_text)

    # Initialize Chroma client with explicit tenant settings
    client = Client(Settings(
        persist_directory=".chroma_data",
        anonymized_telemetry=False
    ))

    # Add documents to Chroma database in bulk
    documents = [{"id": str(i), "content": text, "metadata": {}} for i, text in enumerate(texts)]
    client.add(documents=documents)

    # Retrieve similar documents
    retriever = client.query(query_text=query_text, n_results=5)

    # Create QA chain
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=YOUR_OPENAI_API_KEY, temperature=0),
        chain_type="stuff",
        retriever=retriever
    )
    return qa.run(query_text)

# Streamlit page title and description
st.set_page_config(page_title="GPT Chatbot with PDF Data")
st.title("📄 GPT Chatbot: PDF Data")

# User input for query
query_text = st.text_input("Enter your question:", placeholder="Ask a specific question about the document.")

# Generate response when input is provided
if st.button("Submit") and query_text:
    with st.spinner("Processing your request..."):
        try:
            response = generate_response(query_text)
            st.success(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
