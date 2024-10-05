import os
import streamlit as st
import pickle
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings


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

modelPath = "sentence-transformers/all-MiniLM-l6-v2"

# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device':'cpu'}

# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)

model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Example model for embedding
embedding_model = SentenceTransformer(model_name)
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

    # Use sentence-transformers for embedding generation
    main_placeholder.text("Embedding Text...Started...✅✅✅")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Example model for embedding
    embedding_model = SentenceTransformer(model_name)
    # Define the path to the pre-trained model you want to use
    # modelPath = "sentence-transformers/all-MiniLM-l6-v2"

    # # Create a dictionary with model configuration options, specifying to use the CPU for computations
    # model_kwargs = {'device':'cpu'}

    # # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    # encode_kwargs = {'normalize_embeddings': False}

    # # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    # embeddings = HuggingFaceEmbeddings(
    #     model_name=modelPath,     # Provide the pre-trained model's path
    #     model_kwargs=model_kwargs, # Pass the model configuration options
    #     encode_kwargs=encode_kwargs # Pass the encoding options
    # )


    # texts = [doc.page_content for doc in docs]  # Extract text from the documents
    # embeddings = embedding_model.encode(texts, convert_to_numpy=True)  # Convert to NumPy array
    # print("texts are \n", type(texts))
    # print("texts are \n", texts)
    # # Store embeddings in FAISS
    # vectorstore = FAISS.from_texts(texts, embeddings)
    texts = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(texts, embeddings)

    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

query = main_placeholder.text_input("Question: ")
if query and os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)

        # Generate query embedding
        # query_embedding = embedding_model.encode([query], convert_to_numpy=True)

        # # Perform similarity search using FAISS
        # similar_docs = vectorstore.similarity_search_by_vector(query_embedding, k=3)
        # db = FAISS.load_local(folder_path="../database/faiss_db",embeddings=embeddings,index_name="myFaissIndex")
        similar_docs = vectorstore.similarity_search(query)
        if similar_docs:
            st.header("Top Matching Documents")
            for doc in similar_docs:
                st.write(doc.page_content)
        else:
            st.write("No relevant documents found.")

        # Load DistilGPT-2 for text generation
        generation_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

        # Generate response from query (if using the LLM for completion)
        inputs = tokenizer(query, return_tensors="pt")
        outputs = generation_model.generate(inputs["input_ids"], max_length=200, num_return_sequences=1)

        # Decode and display the result
        result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.header("Answer")
        st.write(result_text)

        # Placeholder for displaying sources
        st.subheader("Sources:")
        st.write("Source retrieval is based on document search, shown above.")
