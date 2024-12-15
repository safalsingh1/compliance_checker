# PDF Ingestion, Embedding, and Querying with LLaMA3.1

This project allows you to upload a PDF document, split its content into chunks, create embeddings using HuggingFace's BGE embedding model, store those embeddings in a FAISS vector store, and query the document using a locally deployed LLaMA3.1 model for contextual answers.

## Features

- **Upload PDF**: Upload a PDF document for processing.
- **Document Chunking**: The PDF is split into smaller chunks for easier handling.
- **Embedding Creation**: Generate embeddings for document chunks using HuggingFace's BGE embedding model.
- **Vector Storage**: Store embeddings locally using FAISS or upload them to Pinecone (optional).
- **LLaMA3.1 Querying**: Query the document using the LLaMA3.1 model for answers based on the document content.

## Requirements

Before running the app, make sure you have the following dependencies installed:

- Python 3.12 or higher
- Streamlit
- LangChain
- HuggingFace Embeddings
- FAISS
- Pinecone (Optional for cloud-based vector store)
- Ollama (For querying the LLaMA3.1 model)

To install the dependencies, create a virtual environment (optional but recommended) and run:

```bash
pip install -r requirements.txt
