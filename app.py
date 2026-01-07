import streamlit as st
import tempfile
import os

# --- MODERN IMPORTS (Matches langchain>=0.3.0) ---
# These imports require: langchain-community, langchain-huggingface, langchain-chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_classic.chains import RetrievalQA

# Page Config
st.set_page_config(page_title="Local Corporate Brain", layout="centered")
st.title("The Corporate Brain (Offline/Private)")
st.caption("Powered by: LangChain 0.3+, Ollama (Llama 3), and Streamlit")

# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to process PDF
def process_pdf(uploaded_file):
    # Temp file handling (Need a real path for PyPDFLoader)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # 1. Load PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        # 2. Split Text
        # Breaking text into chunks so the AI can digest it
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # 3. Create Embeddings
        # Uses 'sentence-transformers' via langchain-huggingface
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 4. Store in Vector DB (Chroma)
        # Using the new 'langchain_chroma' library
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        
        # 5. Create Retrieval Chain
        # Connects to your running Ollama instance
        llm = ChatOllama(model="llama3", temperature=0)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        
        return qa_chain

    finally:
        # Clean up the temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- Sidebar ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    # Only process if a file is uploaded and we haven't already processed it
    if uploaded_file and "qa_chain" not in st.session_state:
        with st.spinner("Reading Document... (This might take a minute)"):
            try:
                st.session_state.qa_chain = process_pdf(uploaded_file)
                st.success("Brain Ready! Ask away.")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")

    # Reset button
    if st.button("Clear History"):
        st.session_state.messages = []
        if "qa_chain" in st.session_state:
            del st.session_state["qa_chain"]
        st.rerun()

# --- Chat Interface ---
# 1. Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Handle User Input
if prompt := st.chat_input("Ask a question about your PDF..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if PDF is loaded
    if "qa_chain" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Run the chain
                    response = st.session_state.qa_chain.invoke({"query": prompt})
                    result = response["result"] # Extract just the answer
                    st.markdown(result)
                    
                    # Add AI response to history
                    st.session_state.messages.append({"role": "assistant", "content": result})
                except Exception as e:
                    st.error(f"Error generating response: {e}")
    else:
        # If no PDF is uploaded yet
        with st.chat_message("assistant"):
            st.markdown("Please upload a PDF first so I can answer your question!")