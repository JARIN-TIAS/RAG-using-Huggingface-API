from huggingface_hub import InferenceClient
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 5. Retrieve context (unchanged)
def retrieve_context(query, embedder, index, documents, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding.astype(np.float32), k)
    return [documents[i] for i in indices[0]]

# Function to get the answer using Hugging Face's InferenceClient
def generate_answer_with_huggingface_inference_client(query, context, api_key):
    # Prepare the input context and query
    formatted_context = "\n".join(context)
    messages = [
        {"role": "user", "content": f"Use the following context to answer the question:\n{formatted_context}\nQuestion: {query}"}
    ]
    
    # Initialize InferenceClient
    client = InferenceClient(
        provider="together",
        api_key=api_key
    )
    
    # Get the completion from the model
    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1",
        messages=messages,
        max_tokens=500
    )
    
    # Extract and return the response
    return completion.choices[0].message['content']

# 1. Install required packages
# pip install pypdf langchain sentence-transformers faiss-cpu huggingface_hub

# 2. Load and process PDF document
def load_pdf_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()  # Returns list of Document objects

# 3. Split text into chunks
def split_documents(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(pages)

# 4. Create vector store
def create_vector_store(split_docs):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    document_texts = [doc.page_content for doc in split_docs]
    embeddings = embedder.encode(document_texts)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    return index, document_texts, embedder

# Main workflow (modified to use Hugging Face InferenceClient)
def main(pdf_path, query, api_key):
    # Load and process PDF
    pages = load_pdf_documents(pdf_path)
    split_docs = split_documents(pages)
    
    # Create vector store
    index, document_texts, embedder = create_vector_store(split_docs)
    
    # Retrieve context
    context = retrieve_context(query, embedder, index, document_texts)
    
    # Get answer from Hugging Face InferenceClient
    answer = generate_answer_with_huggingface_inference_client(query, context, api_key)
    return answer

# Example usage
if __name__ == "__main__":
    pdf_path = "test.pdf"
    query = "বাংলাদেশের স্বাধীনতা অর্জনের বছর কী ছিল?"
    api_key = "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"  # Replace with your Hugging Face API key
    
    result = main(pdf_path, query, api_key)
    print("Answer:", result)
