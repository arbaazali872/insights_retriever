import os
import streamlit as st
import pickle
import time
from transformers import GPT2LMHeadModel, AutoTokenizer
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:  # Validate URL input
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store.pkl"

main_placeholder = st.empty()

if process_url_clicked and urls:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Instead of embeddings, tokenize the text using DistilGPT-2 tokenizer
    main_placeholder.text("Tokenizing Text...Started...✅✅✅")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    texts = [doc.page_content for doc in docs]  # Extract text from the documents
    tokenized_texts = [tokenizer.encode(text) for text in texts]  # Tokenize text

    # Store tokenized texts as vectors in FAISS (note: FAISS expects vectors)
    vectorstore = FAISS.from_texts(texts, lambda docs: tokenized_texts)
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

query = main_placeholder.text_input("Question: ")
if query and os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)

        # Load DistilGPT-2 for question-answer generation
        model = GPT2LMHeadModel.from_pretrained("distilgpt2")
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

        # Simulate retrieval-based QA by generating a response
        inputs = tokenizer(query, return_tensors="pt")
        outputs = model.generate(inputs["input_ids"], max_length=200, num_return_sequences=1)

        # Decode and display the result
        result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.header("Answer")
        st.write(result_text)

        # You can also add a placeholder for displaying "sources" if applicable
        st.subheader("Sources:")
        st.write("Source retrieval is not implemented in this version.")
