import streamlit as st
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
import os

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

# Load the static dataset from the PDF
def load_static_data():
    pdf_path = "Pellet_mill.pdf"  # Ensure this matches your repository's filename
    return extract_text_from_pdf(pdf_path)

def generate_response(query_text):
    # Load and preprocess data
    print("Loading static data from PDF...")
    document_text = load_static_data()

    # Split documents into manageable chunks
    print("Splitting documents into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(document_text)

    # Prepare documents for QA chain
    input_documents = [{"content": text} for text in texts]

    # Create QA chain
    print("Creating QA chain...")
    qa_chain = load_qa_chain(llm=OpenAI(openai_api_key=YOUR_OPENAI_API_KEY, temperature=0), chain_type="stuff")
    print("Running QA chain...")
    return qa_chain.run({"input_documents": input_documents, "question": query_text})

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
