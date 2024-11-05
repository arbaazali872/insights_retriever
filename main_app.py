import os
import streamlit as st
import pickle
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
import torch

# Set up Streamlit app
st.title("News Research Tool")
st.sidebar.title("News Article URLs")

urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store.pkl"

# Initialize placeholders for status updates
main_placeholder = st.empty()

# Set up HuggingFace embeddings
modelPath = "sentence-transformers/all-MiniLM-l6-v2"
embeddings = HuggingFaceEmbeddings(model_name=modelPath, model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': False})

def load_and_process_data(urls):
    """Load data from URLs and split it into manageable chunks."""
    try:
        with st.spinner("Loading data..."):
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()

        # Split the loaded data into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=400)
        docs = text_splitter.split_documents(data)

        # Store embeddings in FAISS
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Save vectorstore to disk
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

        st.success("Data processed and FAISS store saved successfully!")
        return vectorstore
    except Exception as e:
        st.error(f"Error during data processing: {e}")
        return None

if process_url_clicked and urls:
    if not any(urls):
        st.warning("Please enter at least one URL.")
    else:
        load_and_process_data(urls)

query = main_placeholder.text_input("Question: ")
if query and os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)

        # Perform similarity search using FAISS to retrieve relevant documents
        similar_docs = vectorstore.similarity_search(query)

        if similar_docs:
            # Load BERT-large model and tokenizer for extractive Q&A
            model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Combine relevant documents into a single context
            context = "\n\n".join([doc.page_content for doc in similar_docs])

            # Tokenize the input question and context
            inputs = tokenizer(query, context, add_special_tokens=True, return_tensors="pt", max_length=512, truncation=True)
            input_ids = inputs["input_ids"].tolist()[0]

            # Get model outputs (start and end logits)
            outputs = model(**inputs)
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1

            # Decode the predicted answer
            answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
            st.header("Answer:")
            st.write(answer)

            # Post-process answer if it's too short
            if len(answer.split()) < 3:
                st.write("The answer may be incomplete. Here's the relevant context:")
                st.write(context)

            # Remove duplicate sources
            unique_sources = set()
            for doc in similar_docs:
                source = doc.metadata.get("source", "Unknown")
                unique_sources.add(source)

            # Display the sources
            st.subheader("Sources:")
            for source in unique_sources:
                st.write(source)

        else:
            st.write("No relevant documents found.")
