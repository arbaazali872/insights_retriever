import os
import logging
import pickle
import validators
from typing import List
import streamlit as st
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings

# Constants
MODEL_PATH = "sentence-transformers/all-MiniLM-l6-v2"
QA_MODEL_NAME = "bert-large-uncased-whole-word-masking-finetuned-squad"
FILE_PATH = "faiss_store.pkl"
CHUNK_SIZE = 400

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper functions
def validate_urls(urls: List[str]) -> List[str]:
    """Validate and filter out invalid URLs."""
    return [url for url in urls if validators.url(url)]

def load_embeddings():
    """Load HuggingFace embeddings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return HuggingFaceEmbeddings(
        model_name=MODEL_PATH, 
        model_kwargs={'device': str(device)}, 
        encode_kwargs={'normalize_embeddings': False}
    )

def load_and_process_data(urls: List[str], file_path: str):
    """Load data, split into chunks, and save to FAISS."""
    try:
        with st.spinner("Loading and processing data..."):
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()

            # Split the data
            text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=CHUNK_SIZE)
            docs = text_splitter.split_documents(data)

            # Store embeddings in FAISS
            embeddings = load_embeddings()
            vectorstore = FAISS.from_documents(docs, embeddings)

            # Save to file
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore, f)

        st.success("Data processed and FAISS store saved successfully!")
        return vectorstore
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        st.error("Failed to process data. Check the logs for details.")
        return None

def load_vectorstore(file_path: str):
    """Load vectorstore from file."""
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Vectorstore not found. Please process URLs first.")
        return None

def answer_question(query: str, vectorstore, qa_model_name: str):
    """Answer a question using vectorstore and QA model."""
    similar_docs = vectorstore.similarity_search(query)
    if not similar_docs:
        st.info("No relevant documents found.")
        return

    # Combine relevant documents into context
    context = "\n\n".join([doc.page_content for doc in similar_docs])

    # Load QA model and tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
    tokenizer = AutoTokenizer.from_pretrained(qa_model_name)

    # Tokenize input
    inputs = tokenizer(query, context, add_special_tokens=True, return_tensors="pt", max_length=512, truncation=True)
    input_ids = inputs["input_ids"].tolist()[0]

    # Get model predictions
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    st.header("Answer:")
    st.write(answer)

# Main Streamlit app
def main():
    st.title("News Research Tool")
    st.sidebar.title("News Article URLs")

    urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
    process_url_clicked = st.sidebar.button("Process URLs")

    if process_url_clicked:
        valid_urls = validate_urls(urls)
        if not valid_urls:
            st.warning("Please enter at least one valid URL.")
        else:
            load_and_process_data(valid_urls, FILE_PATH)

    query = st.text_input("Question:")
    if query and os.path.exists(FILE_PATH):
        vectorstore = load_vectorstore(FILE_PATH)
        if vectorstore:
            answer_question(query, vectorstore, QA_MODEL_NAME)

if __name__ == "__main__":
    main()
