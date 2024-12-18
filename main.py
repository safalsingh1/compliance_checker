import streamlit as st
from src.controllers.pdf_loader import load_pdf, split_document
from src.config.embeddings import create_embeddings, load_embeddings
from src.utils.vectorstore import initialize_pinecone, upload_to_pinecone
from src.models.llama_model import query_llama_model

st.title("PDF Ingestion and Query with LLaMA2")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    docs = load_pdf(uploaded_file)
    splits = split_document(docs)
    vectorstore = create_embeddings(splits)
    st.write("PDF processed and embeddings created.")

    if st.checkbox("Upload embeddings to Pinecone"):
        initialize_pinecone()
        upload_to_pinecone(vectorstore)
        st.write("Embeddings uploaded to Pinecone.")

user_query = st.text_input("Ask a question to LLaMA2")
if user_query:
    try:
        vectorstore = load_embeddings()  # Load previously created embeddings
        response = query_llama_model(user_query, vectorstore)
        st.write(f"Response from LLaMA2: {response}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
