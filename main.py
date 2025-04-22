import os
import logging
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate


def load_environment():
    load_dotenv()
    logging.getLogger("langchain").setLevel(logging.ERROR)
    return os.getenv('GROQ_API_KEY')


# def load_pdf(pdf_path):
#     loader = PyPDFLoader(pdf_path)
#     docs = loader.load()
#     print(f"✅ Loaded {len(docs)} document chunks successfully.")
#     return docs


# def split_documents(docs, chunk_size=500, chunk_overlap=0):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     chunks = splitter.split_documents(docs)
#     print(f"✅ Split into {len(chunks)} chunks.")
#     return chunks


# def get_embeddings_model(model_path):
#     return HuggingFaceEmbeddings(model_name=model_path)


# def load_or_create_faiss(documents, embeddings_model, faiss_path="faiss_index"):
#     if os.path.exists(faiss_path):
#         vectordb = FAISS.load_local(faiss_path, embeddings_model, allow_dangerous_deserialization=True)
#         print("✅ Loaded FAISS DB from local storage.")
#     else:
#         vectordb = FAISS.from_documents(documents, embeddings_model)
#         vectordb.save_local(faiss_path)
#         print("✅ Created and saved FAISS DB.")
#     return vectordb


def setup_agent(llm, retriever):
    pdf_tool = create_retriever_tool(retriever, "pdf_search", "Search for PDF information only!")
    tools = [pdf_tool]
    prompt = ChatPromptTemplate.from_template("""
        Answer the questions based on the provided PDF context only.
        Provide accurate and detailed responses strictly from the PDF content.
        <context>
        {context}
        <context>
        Questions: {input}
        {agent_scratchpad}
    """)
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    return agent_executor


def query_loop(agent_executor):
    while True:
        query = input("\nInput your query here: ")
        if query.lower() in ["exit", "quit", "q"]:
            print("Exiting... Goodbye!")
            break

        start_time = time.time()
        try:
            response = agent_executor.invoke({
                "input": query,
                "context": "",
                "agent_scratchpad": ""
            })
            print(f"\n🟩 Final Output:\n{response['output']}")
            print(f"⏱️ Total Response Time: {time.time() - start_time:.2f} seconds")

        except Exception as e:
            print(f"❗ Error: {e}")


def main():
    groq_api_key = load_environment()
    pdf_path = "vul_scanner.pdf"
    model_path = r"C:\Users\hp\Desktop\ps_sol\models\all-MiniLM-L6-v2"

    docs = load_pdf(pdf_path)
    chunks = split_documents(docs)
    embeddings_model = get_embeddings_model(model_path)
    vectordb = load_or_create_faiss(chunks, embeddings_model)

    retriever = vectordb.as_retriever()
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
    agent_executor = setup_agent(llm, retriever)

    query_loop(agent_executor)


if __name__ == "__main__":
    main()
