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

# Load environment variables
load_dotenv()
logging.getLogger("langchain").setLevel(logging.ERROR)  # Suppress unnecessary logs
groq_api_key = os.getenv('GROQ_API_KEY')

# === Step 1: Load a specific PDF file ===
pdf_path = input("📄 Enter the full path of the PDF document: ")
loader = PyPDFLoader(pdf_path)
docs = loader.load()
print(f"✅ Loaded {len(docs)} document chunks successfully.")

# === Step 2: Split documents ===
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
documents = text_splitter.split_documents(docs)
print(f"✅ Split into {len(documents)} chunks.")

# === Step 3: Embedding model ===
embeddings_model = HuggingFaceEmbeddings(model_name=r"C:\Users\hp\Desktop\ps_sol\models\all-MiniLM-L6-v2")

# === Step 4: Check if FAISS DB exists ===
faiss_path = r"faiss_index"
# if os.path.exists(faiss_path):
#     vectordb = FAISS.load_local(faiss_path, embeddings_model, allow_dangerous_deserialization=True)
#     print("✅ Loaded FAISS DB from local storage.")
# else:
vectordb = FAISS.from_documents(documents, embeddings_model)
vectordb.save_local(faiss_path)
print("✅ Created and saved FAISS DB.")

retriever = vectordb.as_retriever()
pdf_tool = create_retriever_tool(retriever, "pdf_search", "Search for PDF information only!")
tools = [pdf_tool]

# === Step 5: Load LLaMA 3 from GROQ ===
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

# === Step 6: Prompt Template ===
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided PDF context only.
    Provide accurate and detailed responses strictly from the PDF content.
    <context>
    {context}
    <context>
    Questions: {input}
    {agent_scratchpad}
    """
)

# === Step 7: Agent and Executor ===
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# === Step 8: Query loop ===
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
