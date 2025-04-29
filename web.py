import os
import logging
import time
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from werkzeug.utils import secure_filename
from langchain_community.chat_models import ChatOllama


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger("langchain").setLevel(logging.ERROR)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Global Variables
agent_executor = None
current_pdf_path = None
retriever_global = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_pdf(pdf_path):
    """Load the uploaded PDF and create the agent_executor."""
    global agent_executor

    try:
        logger.info(f"Processing PDF: {pdf_path}")

        # Step 1: Load PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        logger.info(f"✅ Loaded {len(docs)} document chunks successfully.")

        # Step 2: Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        documents = text_splitter.split_documents(docs)
        logger.info(f"✅ Split into {len(documents)} chunks.")

        # Step 3: Load embeddings from local model
        embeddings_model = HuggingFaceEmbeddings(
            model_name=r"C:\Users\hp\Desktop\ps_sol\models\all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Step 4: Create FAISS vector DB
        vectordb = FAISS.from_documents(documents, embeddings_model)
        base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        faiss_path = os.path.join('faiss_indexes', f"faiss_index_{base_filename}")
        os.makedirs('faiss_indexes', exist_ok=True)
        vectordb.save_local(faiss_path)
        logger.info(f"✅ Created and saved FAISS DB at {faiss_path}")

        retriever = vectordb.as_retriever()

        global retriever_global
        retriever_global = retriever

        # Step 5: Create retriever tool
        pdf_tool = create_retriever_tool(
            retriever,
            "pdf_search",
            "Search information inside the PDF"
        )
        tools = [pdf_tool]

        # Step 6: Load Ollama LLM (Gemma 2B)
        llm = ChatOllama(
            model="gemma2:2b",
            base_url="http://localhost:11434",
            temperature=0.7
        )

        # Step 7: Create prompt
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

        # Step 8: Create agent and executor
        agent = create_openai_tools_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

        return True

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_pdf_path

    if 'pdf_file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded.'})

    file = request.files['pdf_file']

    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected.'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        current_pdf_path = filepath

        if process_pdf(filepath):
            return jsonify({'status': 'success', 'message': 'PDF uploaded and processed successfully.'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to process PDF.'})

    return jsonify({'status': 'error', 'message': 'Invalid file format. Only PDFs are allowed.'})


@app.route('/ask', methods=['POST'])
def ask_question():
    global agent_executor
    global retriever_global

    if not agent_executor:
        return jsonify({'status': 'error', 'message': 'No PDF processed. Please upload a PDF first.'})

    if not retriever_global:
        return jsonify({'status': 'error', 'message': 'Retriever not ready.'})

    data = request.json
    query = data.get('query', '').strip()

    if not query:
        return jsonify({'status': 'error', 'message': 'Query is empty.'})

    try:
        start_time = time.time()

        # ✅ Use the global retriever
        docs = retriever_global.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        response = agent_executor.invoke({
            "input": query,
            "context": context,
            "agent_scratchpad": ""
        })

        answer = response.get('output', 'No output generated.')
        response_time = time.time() - start_time

        return jsonify({
            'status': 'success',
            'answer': answer,
            'response_time': f"{response_time:.2f}"
        })

    except Exception as e:
        logger.error(f"Error answering question: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'Backend Error: {str(e)}'})

# @app.route('/ask', methods=['POST'])
# def ask_question():
#     global agent_executor

#     if not agent_executor:
#         return jsonify({'status': 'error', 'message': 'No PDF processed. Please upload a PDF first.'})

#     data = request.json
#     query = data.get('query', '').strip()

#     if not query:
#         return jsonify({'status': 'error', 'message': 'Query is empty.'})

#     try:
#         start_time = time.time()

#         response = agent_executor.invoke({
#             "input": query,
#             "context": context,
#             "agent_scratchpad": ""
#         })

#         answer = response.get('output', 'No output generated.')
#         response_time = time.time() - start_time

#         return jsonify({
#             'status': 'success',
#             'answer': answer,
#             'response_time': f"{response_time:.2f}"
#         })

#     except Exception as e:
#         logger.error(f"Error answering question: {str(e)}", exc_info=True)
#         return jsonify({'status': 'error', 'message': f'Backend Error: {str(e)}'})

# @app.route('/ask', methods=['POST'])
# def ask_question():
#     global agent_executor
#     global retriever_global


#     if not agent_executor:
#         return jsonify({'status': 'error', 'message': 'No PDF processed. Please upload a PDF first.'})

#     data = request.json
#     query = data.get('query', '').strip()

#     if not query:
#         return jsonify({'status': 'error', 'message': 'Query is empty.'})

#     try:
#         start_time = time.time()

#         # ✅ New: Retrieve context from the retriever
#         retriever_tool = agent_executor.tools[0]
#         retriever = retriever_tool.args['retriever']
#         docs = retriever.get_relevant_documents(query)
#         context = "\n\n".join([doc.page_content for doc in docs])

#         # ✅ Now pass real context
#         response = agent_executor.invoke({
#             "input": query,
#             "context": context,
#             "agent_scratchpad": ""
#         })

#         answer = response.get('output', 'No output generated.')
#         response_time = time.time() - start_time

#         return jsonify({
#             'status': 'success',
#             'answer': answer,
#             'response_time': f"{response_time:.2f}"
#         })

#     except Exception as e:
#         logger.error(f"Error answering question: {str(e)}", exc_info=True)
#         return jsonify({'status': 'error', 'message': f'Backend Error: {str(e)}'})


if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(debug=True, use_reloader=False)
















# import os
# import logging
# import time
# from flask import Flask, render_template, request, jsonify
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain_community.llms import Ollama
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.tools.retriever import create_retriever_tool
# from langchain.agents import create_openai_tools_agent, AgentExecutor
# from langchain_core.prompts import ChatPromptTemplate
# from werkzeug.utils import secure_filename

# # Load environment variables
# load_dotenv()

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)
# logging.getLogger("langchain").setLevel(logging.ERROR)

# # Load GROQ API key
# groq_api_key = os.getenv('GROQ_API_KEY')

# # Initialize Flask app
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
# app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# # Global Variables
# agent_executor = None
# current_pdf_path = None

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# def process_pdf(pdf_path):
#     """Load the uploaded PDF and create the agent_executor."""
#     global agent_executor

#     try:
#         logger.info(f"Processing PDF: {pdf_path}")
        
#         # Step 1: Load PDF
#         loader = PyPDFLoader(pdf_path)
#         docs = loader.load()
#         logger.info(f"✅ Loaded {len(docs)} document chunks successfully.")

#         # Step 2: Split documents
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
#         documents = text_splitter.split_documents(docs)
#         logger.info(f"✅ Split into {len(documents)} chunks.")

#         # Step 3: Load embeddings from local model
#         embeddings_model = HuggingFaceEmbeddings(
#             model_name=r"C:\Users\hp\Desktop\ps_sol\models\all-MiniLM-L6-v2",
#             model_kwargs={'device': 'cpu'}
#         )

#         # Step 4: Create FAISS vector DB
#         vectordb = FAISS.from_documents(documents, embeddings_model)
#         base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
#         faiss_path = os.path.join('faiss_indexes', f"faiss_index_{base_filename}")
#         os.makedirs('faiss_indexes', exist_ok=True)
#         vectordb.save_local(faiss_path)
#         logger.info(f"✅ Created and saved FAISS DB at {faiss_path}")

#         retriever = vectordb.as_retriever(search_kwargs={"k": 3})

#         # Step 5: Create retriever tool
#         pdf_tool = create_retriever_tool(
#             retriever,
#             "pdf_search",
#             "Search information inside the PDF"
#         )
#         tools = [pdf_tool]

#         # Step 6: Load ChatGroq LLM
#         # llm = ChatGroq(
#         #     groq_api_key=groq_api_key,
#         #     model_name="llama3-70b-8192",
#         #     max_tokens=512,  # 👈 limit output size
#         #     temperature=0.7
#         # )

#         llm = Ollama(
#             model="gemma2:2b",
#             base_url="http://localhost:11434",
#             temperature=0.7
#         )

#         # Step 7: Create prompt
#         prompt = ChatPromptTemplate.from_template(
#             """
#             Answer the questions based on the provided PDF context only.
#             Provide accurate and detailed responses strictly from the PDF content.
#             <context>
#             {context}
#             <context>
#             Questions: {input}
#             {agent_scratchpad}
#             """
#         )

#         # Step 8: Create agent and executor
#         agent = create_openai_tools_agent(llm, tools, prompt)
#         agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

#         return True

#     except Exception as e:
#         logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
#         return False

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     global current_pdf_path

#     if 'pdf_file' not in request.files:
#         return jsonify({'status': 'error', 'message': 'No file uploaded.'})

#     file = request.files['pdf_file']

#     if file.filename == '':
#         return jsonify({'status': 'error', 'message': 'No file selected.'})

#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
#         current_pdf_path = filepath

#         if process_pdf(filepath):
#             return jsonify({'status': 'success', 'message': 'PDF uploaded and processed successfully.'})
#         else:
#             return jsonify({'status': 'error', 'message': 'Failed to process PDF.'})

#     return jsonify({'status': 'error', 'message': 'Invalid file format. Only PDFs are allowed.'})

# @app.route('/ask', methods=['POST'])
# def ask_question():
#     global agent_executor

#     if not agent_executor:
#         return jsonify({'status': 'error', 'message': 'No PDF processed. Please upload a PDF first.'})

#     data = request.json
#     query = data.get('query', '').strip()

#     if not query:
#         return jsonify({'status': 'error', 'message': 'Query is empty.'})

#     try:
#         start_time = time.time()

#         # 👇 Correct invocation structure
#         response = agent_executor.invoke({
#             "input": query,
#             "context": "",
#             "agent_scratchpad": ""
#         })

#         answer = response.get('output', 'No output generated.')
#         response_time = time.time() - start_time

#         return jsonify({
#             'status': 'success',
#             'answer': answer,
#             'response_time': f"{response_time:.2f}"
#         })

#     except Exception as e:
#         logger.error(f"Error answering question: {str(e)}", exc_info=True)
#         return jsonify({'status': 'error', 'message': f'Backend Error: {str(e)}'})

# if __name__ == '__main__':
#     logger.info("Starting Flask server...")
#     app.run(debug=True, use_reloader=False)
