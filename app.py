import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()

st.title("AskMyPDF")
st.caption("Upload your PDF and ask any questions from it.")

with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Get your API key from https://console.groq.com/keys")
    if not groq_api_key:
        st.warning("⚠️ Please enter your Groq API Key to proceed.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    if not groq_api_key:
        st.info("Please add your Groq API key in the sidebar to continue.")
        st.stop()

    # Check if we need to process the uploaded file
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        with st.status("Processing Document...", expanded=True) as status:
            st.write("📥 Saving uploaded file...")
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())

            st.write("📄 Extracting text from PDF...")
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()

            st.write("✂️ Chunking document...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150
            )
            chunks = splitter.split_documents(docs)

            st.write("🧠 Creating vector embeddings...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            db = Chroma.from_documents(chunks, embeddings)

            st.write("🤖 Setting up AI Retriever...")
            retriever = db.as_retriever()

            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                api_key=groq_api_key
            )

            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever
            )
            
            # Store in session state
            st.session_state.qa_chain = qa
            st.session_state.current_file = uploaded_file.name
            st.session_state.messages = []
            
            status.update(label="Document processed successfully!", state="complete", expanded=False)

    # Use the cached QA chain
    qa = st.session_state.qa_chain

    # Initialize chat history if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if question := st.chat_input("Ask a question about the PDF"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(question)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing document and thinking..."):
                answer = qa.run(question)
            st.markdown(answer)
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})