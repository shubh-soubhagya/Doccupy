from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate


def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} document chunks successfully.")
    return docs


def split_documents(docs, chunk_size=500, chunk_overlap=0):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"✅ Split into {len(chunks)} chunks.")
    return chunks

def get_embeddings_model(model_path):
    return HuggingFaceEmbeddings(model_name=model_path)


def load_or_create_faiss(documents, embeddings_model, faiss_path="faiss_index"):
    if os.path.exists(faiss_path):
        vectordb = FAISS.load_local(faiss_path, embeddings_model, allow_dangerous_deserialization=True)
        print("✅ Loaded FAISS DB from local storage.")
    else:
        vectordb = FAISS.from_documents(documents, embeddings_model)
        vectordb.save_local(faiss_path)
        print("✅ Created and saved FAISS DB.")
    return vectordb