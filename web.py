import os
import logging
import time
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()
logging.getLogger("langchain").setLevel(logging.ERROR)  # Suppress unnecessary logs
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create a session variable to store the current PDF path
current_pdf_path = None
agent_executor = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_pdf(pdf_path):
    """Process the PDF and set up the agent executor"""
    global agent_executor
    
    # === Step 1: Load PDF file ===
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} document chunks successfully.")
    
    # === Step 2: Split documents ===
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    documents = text_splitter.split_documents(docs)
    print(f"✅ Split into {len(documents)} chunks.")
    
    # === Step 3: Embedding model ===
    embeddings_model = HuggingFaceEmbeddings(model_name=r"all-MiniLM-L6-v2")
    
    # === Step 4: Create FAISS DB ===
    vectordb = FAISS.from_documents(documents, embeddings_model)
    
    # Save with a unique name based on the filename
    base_filename = os.path.basename(pdf_path).split('.')[0]
    faiss_path = f"faiss_index_{base_filename}"
    vectordb.save_local(faiss_path)
    print(f"✅ Created and saved FAISS DB to {faiss_path}.")
    
    retriever = vectordb.as_retriever()
    pdf_tool = create_retriever_tool(retriever, "pdf_search", "Search for PDF information only!")
    tools = [pdf_tool]
    
    # === Step 5: Load LLM from GROQ ===
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
    
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_pdf_path
    
    if 'pdf_file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})
    
    file = request.files['pdf_file']
    
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        current_pdf_path = filepath
        
        try:
            success = process_pdf(filepath)
            if success:
                return jsonify({'status': 'success', 'message': 'PDF uploaded and processed successfully'})
            else:
                return jsonify({'status': 'error', 'message': 'Failed to process PDF'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Error processing PDF: {str(e)}'})
    
    return jsonify({'status': 'error', 'message': 'Invalid file format. Please upload a PDF.'})

@app.route('/ask', methods=['POST'])
def ask_question():
    global agent_executor
    
    if not agent_executor:
        return jsonify({'status': 'error', 'message': 'Please upload a PDF first'})
    
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'status': 'error', 'message': 'Query is empty'})
    
    try:
        start_time = time.time()
        response = agent_executor.invoke({
            "input": query,
            "context": "",
            "agent_scratchpad": ""
        })
        answer = response['output']
        response_time = time.time() - start_time
        
        return jsonify({
            'status': 'success',
            'answer': answer,
            'response_time': f"{response_time:.2f}"
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)