import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv()

st.set_page_config(page_title="RAG Marketing Knowledge Assistant", layout="wide")

st.title("RAG Marketing Knowledge Assistant")
st.write(
    "Upload a marketing document, campaign notes, FAQ, or customer feedback file. "
    "Then ask questions and get AI-generated answers grounded in the document."
)

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

question = st.text_input("Ask a question about the uploaded document")

def save_uploaded_file(uploaded_file):
    os.makedirs("uploaded_docs", exist_ok=True)
    file_path = os.path.join("uploaded_docs", uploaded_file.name)

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return file_path

def load_document(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    return loader.load()

def build_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store

def answer_question(vector_store, question):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain.invoke({"query": question})

    return result

if uploaded_file and question:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please add your OpenAI API key to a .env file before running this project.")
    else:
        with st.spinner("Reading document and generating answer..."):
            file_path = save_uploaded_file(uploaded_file)
            documents = load_document(file_path)
            vector_store = build_vector_store(documents)
            result = answer_question(vector_store, question)

            st.subheader("Answer")
            st.write(result["result"])

            st.subheader("Retrieved Source Context")
            for i, doc in enumerate(result["source_documents"], start=1):
                st.markdown(f"**Source {i}:**")
                st.write(doc.page_content[:700])
