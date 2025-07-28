import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()


# Initialize components
def initialize_rag():
    # Set OpenAI API Key
      # Replace with your key
    
    # 1. Load Document (PDF example)
    print("Enter the path to your PDF file:")
    loader = PyPDFLoader(input())
    documents = loader.load()
    
    # 2. Split Documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    # 3. Create Vector Store
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    
    # 4. Initialize QA Chain
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_store.as_retriever(),
        chain_type="stuff"
    )
    qa_chain.get_graph().print_ascii()
    
    return qa_chain

# Initialize RAG system

qa_system = initialize_rag()

# Test query
print("Enter your query:")
query = input()
response = qa_system.invoke({"query": query})
print("Response:", response["result"])
