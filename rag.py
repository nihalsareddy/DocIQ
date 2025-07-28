from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv() # Load environment variables from .env file

# Define the general-purpose prompt template
prompt_template = """You are a helpful AI assistant tasked with answering questions about the provided document.
Use only the information available in the document excerpt to answer the question.
If the answer is not present in the document, state that you cannot find the information.
Provide concise and direct answers.
Anaylize the document for key information and context.

Document Excerpt:
{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# --- Streamlit UI Components and Logic ---

st.set_page_config(page_title="PDF Q&A Assistant", layout="wide")
st.title("📄 PDF Q&A Assistant")

# Initialize session state variables if they don't exist
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False

# File uploader
uploaded_file = st.file_uploader("Upload your PDF file here", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location to be processed by PyPDFLoader
    # Streamlit's file_uploader provides a BytesIO object, PyPDFLoader needs a path
    temp_file_path = os.path.join("./temp_pdf_storage", uploaded_file.name)
    os.makedirs("./temp_pdf_storage", exist_ok=True)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info(f"Processing PDF: {uploaded_file.name}")

    try:
        # Load document
        doc_loader = PyPDFLoader(temp_file_path)
        document_pages = doc_loader.load()
        st.success(f"Successfully loaded {len(document_pages)} pages from the PDF.")

        # Text Splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""]
        )
        chunks = text_splitter.split_documents(document_pages)
        st.info(f"Split the document into {len(chunks)} text chunks for analysis.")

        # Ensure OPENAI_API_KEY is set
        if os.getenv("OPENAI_API_KEY") is None:
            st.error("Error: OPENAI_API_KEY environment variable not set. Please set it in your .env file.")
            st.session_state.document_loaded = False
        else:
            # Embeddings and Vector Store
            with st.spinner("Creating embeddings and vector store..."):
                embeddings = OpenAIEmbeddings()
                st.session_state.vector_store = Chroma.from_documents(chunks, embeddings)
            st.success("Vector store created successfully.")

            # Retrieval QA Chain
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 5}),
                chain_type_kwargs={"prompt": PROMPT},
                chain_type="stuff"
            )
            st.session_state.document_loaded = True
            st.success("PDF processed and ready for questions!")

    except Exception as e:
        st.error(f"An error occurred during PDF processing: {e}")
        st.error("Please ensure the file is a valid PDF and your OpenAI API key is correctly set.")
        st.session_state.document_loaded = False

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            # Optionally remove the temporary directory if empty
            if not os.listdir(os.path.dirname(temp_file_path)):
                os.rmdir(os.path.dirname(temp_file_path))


# Question input and answer display
if st.session_state.document_loaded:
    st.subheader("Ask a Question about the PDF")
    query = st.text_area(
        "Enter your question here:",
        
        height=100
    )

    if st.button("Get Answer"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Getting answer..."):
                try:
                    response = st.session_state.qa_chain.invoke({"query": query})
                    st.markdown("---")
                    st.subheader("Answer:")
                    st.write(response["result"])
                except Exception as e:
                    st.error(f"An error occurred while getting the answer: {e}")
                    st.error("Please try rephrasing your question or check your internet connection.")
else:
    st.info("Upload a PDF file above to start asking questions.")

st.markdown("---")
st.caption("Powered by Langchain, OpenAI, and Streamlit")
