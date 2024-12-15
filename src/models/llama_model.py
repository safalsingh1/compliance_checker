from langchain_community.llms import Ollama

def query_llama_model(query, vectorstore):
    """Query the LLaMA model using context."""
    llm = Ollama(model="llama3.1")

    try:
        search_results = vectorstore.similarity_search(query, k=5)
        context = "\n".join([result.page_content for result in search_results])
    except Exception as e:
        print(f"Error retrieving context: {str(e)}")
        context = "No relevant context from the uploaded document."

    prompt = f"""
    You are a helpful assistant. Use the following context to answer the user's question.

    Context:
    {context}

    User's question:
    {query}
    """
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
