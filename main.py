import os
import pickle
import time
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

embedding_file = "embeddings.pkl"

# Function to load the PDF
def load_pdf(uploaded_file):
    """Load the PDF file."""
    start_time = time.time()
    with open("temp_pdf_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    docs = PyPDFLoader("temp_pdf_file.pdf").load()
    end_time = time.time()
    st.write(f"PDF loading time: {end_time - start_time:.2f} seconds")
    return docs

# Function to split the document into chunks
def split_document(docs):
    """Split the document into chunks."""
    start_time = time.time()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    end_time = time.time()
    st.write(f"Document splitting time: {end_time - start_time:.2f} seconds")
    return splits

# Function to create embeddings and store them locally using FAISS
def create_embeddings(splits):
    """Create and save embeddings if no existing embeddings are found, with a progress bar."""
    start_time = time.time()

    if os.path.exists(embedding_file):
        st.write("Removing old embeddings file...")
        os.remove(embedding_file)

    st.write("Embedding the document...")

    progress_bar = st.progress(0)
    total_chunks = len(splits)

    # Set up HuggingFace BGE Embedding model
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    
    hf_embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    embedded_docs = []

    for i, split in enumerate(splits):
        embedded_docs.append(split)  

        progress_percentage = (i + 1) / total_chunks
        progress_bar.progress(progress_percentage)

    # Create FAISS vectorstore after all chunks are embedded
    vectorstore = FAISS.from_documents(embedded_docs, hf_embeddings)

    # Save embeddings to disk
    with open(embedding_file, "wb") as f:
        pickle.dump(vectorstore, f)
    
    end_time = time.time()
    st.write(f"Embedding time: {end_time - start_time:.2f} seconds")
    return vectorstore

# Function to upload PDF and create embeddings
def handle_uploaded_pdf(uploaded_file):
    """Handle the uploaded PDF file."""
    docs = load_pdf(uploaded_file)
    splits = split_document(docs)
    vectorstore = create_embeddings(splits)
    return vectorstore

# Pinecone integration 
def initialize_pinecone():
    """Initialize Pinecone connection."""
    api_key = os.getenv("PINECONE_API_KEY")

    # Initialize Pinecone client
    pc = Pinecone(api_key=api_key)

    index_name = "pdf-index"

    # Delete the index if it already exists
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)
        st.write(f"Deleted existing index: {index_name}")

    pc.create_index(
        name=index_name,
        dimension=384, 
        metric="euclidean",
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    st.write(f"Created index: {index_name}")

def upload_to_pinecone(vectorstore):
    """Upload the embeddings from FAISS to Pinecone."""
    api_key = os.getenv("PINECONE_API_KEY")
    host = os.getenv("PINECONE_HOST")

    # Initialize Pinecone client
    pc = Pinecone(api_key=api_key)

    index_name = "pdf-index"

    index = pc.Index(index_name)

    # Extract embeddings and document IDs from FAISS
    vectors = []
    for doc_id, embedding in zip(vectorstore.docstore._dict.keys(), vectorstore.index.reconstruct_n(0, vectorstore.index.ntotal)):
        metadata = {"source": "pdf"} 
        vectors.append((doc_id, embedding, metadata))

    # Upload to Pinecone
    try:
        index.upsert(vectors=vectors)
        st.write("Vectors uploaded to Pinecone successfully.")
    except Exception as e:
        st.error(f"Failed to upload vectors to Pinecone: {str(e)}")

# Local LLaMA3.1 model setup with Ollama
def query_llama_model(query, vectorstore):
    """Query the LLaMA3.1 model using Ollama, with context from the document."""
    llm = Ollama(model="llama3.1")  # Ensure this matches your setup

    try:
        # Use the vectorstore to find relevant context
        search_results = vectorstore.similarity_search(query, k=5)
        context = "\n".join([result.page_content for result in search_results])
    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        context = ""

    if not context:
        st.warning("No relevant context found in the document.")
        context = "No relevant context from the uploaded document."

    # Combine context with the user query
    prompt = f"""
    You are a helpful assistant. Use the following context to answer the user's question.

    Context:
    {context}

    User's question:
    {query}
    """

    # Query the model
    try:
        response = llm(prompt)
        if isinstance(response, str):
            return response
        elif isinstance(response, dict) and "content" in response:
            return response["content"]
        else:
            return "Unexpected response format from the LLaMA model."
    except Exception as e:
        return f"Error querying LLaMA3.1 model: {str(e)}"

# Streamlit interface
st.title("PDF Ingestion, Embedding, Vector Store, and Query LLaMA3.1")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    vectorstore = handle_uploaded_pdf(uploaded_file)
    st.write("PDF processed and embeddings created.")
    
    # Optionally, upload embeddings to Pinecone
    if st.checkbox("Upload embeddings to Pinecone"):
        initialize_pinecone()
        upload_to_pinecone(vectorstore)
        st.write("Embeddings uploaded to Pinecone.")

# Allow user to query the local LLaMA3.1 model
user_query = st.text_input("Ask a question to LLaMA3.1")

if user_query:
    response = query_llama_model(user_query, vectorstore)
    st.write(f"Response from LLaMA3.1: {response}")
