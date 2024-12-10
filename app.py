import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
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

def generate_response(openai_api_key, query_text):
    documents = [load_static_data()]
    # Split documents into manageable chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.create_documents(documents)
    
    # Select embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    # Initialize Chroma vector store with an in-memory setup
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=".chroma_data"  # Directory for persistence
    )
    
    # Create retriever interface
    retriever = db.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 most relevant chunks
    
    # Load the retriever with a limit to avoid token overflow
    docs = retriever.get_relevant_documents(query_text)

    # Create QA chain with a custom prompt to reduce verbosity
    prompt_template = """
    You are a helpful assistant. Use the following extracted parts of a document to answer the question.
    If you don't know the answer, just say "I don't know." Do not make up an answer.

    {context}

    Question: {question}
    Answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = load_qa_chain(OpenAI(openai_api_key=openai_api_key), chain_type="stuff", prompt=PROMPT)
    
    # Run the QA chain and return the result
    return qa_chain.run(input_documents=docs, question=query_text)

# Streamlit page title and description
st.set_page_config(page_title='GPT Chatbot with PDF Data')
st.title('📄 GPT Chatbot: PDF Data')

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
