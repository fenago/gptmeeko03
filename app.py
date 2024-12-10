import streamlit as st
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from chromadb import Client
from chromadb.config import Settings
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
import os
import shutil
import logging

# Setup logging
logging.basicConfig(filename="debug.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger()

# Access the OpenAI API key from environment variables or Streamlit secrets
YOUR_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", st.secrets.get("YOUR_OPENAI_API_KEY", ""))
if not YOUR_OPENAI_API_KEY:
    log.error("OpenAI API Key is not set.")
    raise ValueError("OpenAI API Key is not set. Please add it to the environment variables or Streamlit secrets.")

# Function to extract text from the PDF
def extract_text_from_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to clean up Chroma data
def clean_chroma_data(chroma_directory):
    if os.path.exists(chroma_directory):
        log.debug("Cleaning up Chroma directory...")
        shutil.rmtree(chroma_directory)

# Load the static dataset from the PDF
def load_static_data():
    pdf_path = "Pellet_mill.pdf"  # Ensure this matches your repository's filename
    return extract_text_from_pdf(pdf_path)

def initialize_chroma(chroma_directory):
    # Ensure no conflicting Chroma instances are running
    try:
        log.debug("Attempting to initialize Chroma client...")
        client = Client(Settings(
            persist_directory=chroma_directory,
            anonymized_telemetry=False
        ))
        return client
    except Exception as e:
        log.error(f"Chroma initialization error: {e}")
        clean_chroma_data(chroma_directory)
        log.debug("Retrying Chroma initialization after cleanup...")
        client = Client(Settings(
            persist_directory=chroma_directory,
            anonymized_telemetry=False
        ))
        return client

def generate_response(query_text):
    chroma_directory = ".chroma_data"

    # Debug: Check for existing directory
    log.debug(f"Checking Chroma directory: {chroma_directory}")
    if os.path.exists(chroma_directory):
        log.debug("Chroma directory exists.")
    else:
        log.debug("Chroma directory does not exist, creating...")

    # Initialize Chroma client
    client = initialize_chroma(chroma_directory)

    # Load and preprocess data
    log.debug("Loading static data from PDF...")
    document_text = load_static_data()

    # Split documents into manageable chunks
    log.debug("Splitting documents into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(document_text)

    # Check if documents are already in Chroma
    collections = client.list_collections()
    log.debug(f"Existing collections: {collections}")
    if len(collections) == 0:
        log.debug("No collections found, adding documents...")
        documents = [{"id": str(i), "content": text, "metadata": {}} for i, text in enumerate(texts)]
        client.add(documents=documents)
    else:
        log.debug("Documents already exist in the database.")

    # Retrieve similar documents
    log.debug("Querying Chroma for similar documents...")
    retriever = client.query(query_text=query_text, n_results=5)

    # Create QA chain
    log.debug("Creating QA chain...")
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=YOUR_OPENAI_API_KEY, temperature=0),
        chain_type="stuff",
        retriever=retriever
    )
    log.debug("Running QA chain...")
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
            log.error(f"Error during response generation: {e}")
            st.error(f"An error occurred: {e}")
