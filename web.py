import os
import pypdf
import logging
import time
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logging.getLogger("langchain").setLevel(logging.ERROR)  # Suppress unnecessary logs
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize Flask app
app = Flask(__name__)
# Configure CORS properly
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"]
    }
})

# Ensure StoreRoom directory exists
STORE_ROOM = 'StoreRoom'
os.makedirs(STORE_ROOM, exist_ok=True)

app.config['UPLOAD_FOLDER'] = os.path.join(STORE_ROOM, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Create a session variable to store the current PDF path
current_pdf_path = None
agent_executor = None
retriever = None  # Add this global variable at the top

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_pdf(pdf_path):
    """Process the PDF and set up the agent executor"""
    global agent_executor
    global retriever  # Add this line
    
    try:
        logger.info(f"Starting to process PDF: {pdf_path}")
        
        # Check if file exists
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found at path: {pdf_path}")
            return False
            
        # Check file size
        file_size = os.path.getsize(pdf_path)
        logger.info(f"PDF file size: {file_size} bytes")
        
        # === Step 1: Load PDF file ===
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            if not docs:
                logger.error("No content found in PDF")
                return False
            logger.info(f"✅ Loaded {len(docs)} document chunks successfully.")
        except Exception as e:
            logger.error(f"Error loading PDF: {str(e)}", exc_info=True)
            return False
        
        # === Step 2: Split documents ===
        try:
            # Use smaller chunk size and overlap for better processing
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=200,  # Smaller chunks
                chunk_overlap=20,  # Small overlap
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            documents = text_splitter.split_documents(docs)
            if not documents:
                logger.error("No documents generated after splitting")
                return False
            logger.info(f"✅ Split into {len(documents)} chunks.")
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}", exc_info=True)
            return False
        
        # === Step 3: Create embeddings ===
        try:
            # Use a simpler model for embeddings
            embeddings_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",  # Smaller model
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("✅ Created embeddings model")
        except Exception as e:
            logger.error(f"Error creating embeddings model: {str(e)}", exc_info=True)
            return False
        
        # === Step 4: Create FAISS DB ===
        try:
            # First, get embeddings for a sample document to verify
            sample_text = documents[0].page_content
            sample_embedding = embeddings_model.embed_query(sample_text)
            if not sample_embedding:
                logger.error("Failed to generate embeddings for sample text")
                return False
                
            # Now create the FAISS DB with specific parameters
            vectordb = FAISS.from_documents(
                documents, 
                embeddings_model,
                normalize_L2=True  # Normalize vectors for better similarity search
            )
            base_filename = os.path.basename(pdf_path).split('.')[0]
            faiss_path = os.path.join(STORE_ROOM, f"faiss_index_{base_filename}")
            vectordb.save_local(faiss_path)
            logger.info(f"✅ Created and saved FAISS DB to {faiss_path}.")
        except Exception as e:
            logger.error(f"Error creating FAISS DB: {str(e)}", exc_info=True)
            return False
        
        # === Step 5: Create retriever and tools ===
        try:
            retriever = vectordb.as_retriever(
                search_kwargs={"k": 3}  # Get top 3 most relevant chunks
            )
            pdf_tool = create_retriever_tool(
                retriever, 
                "pdf_search", 
                "Search for PDF information only! Use this tool to find relevant information in the PDF."
            )
            tools = [pdf_tool]
            logger.info("✅ Created retriever and tools")
        except Exception as e:
            logger.error(f"Error creating retriever and tools: {str(e)}", exc_info=True)
            return False
        
        # === Step 6: Load LLM from GROQ ===
        try:
            if not groq_api_key:
                logger.error("GROQ API key not found in environment variables")
                return False
            llm = ChatGroq(
                groq_api_key=groq_api_key, 
                model_name="llama3-8b-8192",
                temperature=0.7  # Add some creativity to responses
            )
            logger.info("✅ Created LLM instance")
        except Exception as e:
            logger.error(f"Error creating LLM: {str(e)}", exc_info=True)
            return False
        
        # === Step 7: Create prompt template ===
        try:
            prompt = ChatPromptTemplate.from_template(
                """
You are a helpful AI assistant. Answer the user's question using ONLY the provided PDF context. If the answer is not in the context, say 'I cannot find that information in the PDF.' Do not keep searching or rephrasing. Always answer as soon as you find relevant information, or say you cannot find it and stop.
Context: {context}
Question: {input}
Agent Scratchpad: {agent_scratchpad}
"""
            )
            logger.info("✅ Created prompt template")
        except Exception as e:
            logger.error(f"Error creating prompt template: {str(e)}", exc_info=True)
            return False
        
        # === Step 8: Create agent and executor ===
        try:
            agent = create_openai_tools_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent, 
                tools=tools, 
                verbose=True,  # Enable verbose mode for debugging
                max_iterations=1,  # Only one step to avoid loops
                handle_parsing_errors=True,  # Handle parsing errors gracefully
                return_intermediate_steps=True  # Return intermediate steps for debugging
            )
            logger.info("✅ Created agent and executor")
        except Exception as e:
            logger.error(f"Error creating agent and executor: {str(e)}", exc_info=True)
            return False
        
        return True
    except Exception as e:
        logger.error(f"Unexpected error in process_pdf: {str(e)}", exc_info=True)
        return False

def get_pdf_title(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            title = reader.metadata.title
            return title if title else "No title found in PDF metadata."
    except Exception as e:
        return f"Error reading PDF metadata: {e}"

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'success', 'message': 'Server is working!'})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        return '', 200
        
    global current_pdf_path
    
    try:
        logger.info("Received file upload request")
        if 'pdf_file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'status': 'error', 'message': 'No file part'}), 400
        
        file = request.files['pdf_file']
        logger.info(f"Received file: {file.filename}")
        
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            logger.error(f"Invalid file format: {file.filename}")
            return jsonify({'status': 'error', 'message': 'Invalid file format. Please upload a PDF.'}), 400
        
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.info(f"Saving file to: {filepath}")
            file.save(filepath)
            current_pdf_path = filepath
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}", exc_info=True)
            return jsonify({'status': 'error', 'message': f'Error saving file: {str(e)}'}), 500
        
        try:
            success = process_pdf(filepath)
            if success:
                logger.info("PDF processed successfully")
                return jsonify({'status': 'success', 'message': 'PDF uploaded and processed successfully'})
            else:
                logger.error("Failed to process PDF")
                return jsonify({'status': 'error', 'message': 'Failed to process PDF. Please check the server logs for details.'}), 500
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
            return jsonify({'status': 'error', 'message': f'Error processing PDF: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in upload_file: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'Unexpected error: {str(e)}'}), 500

@app.route('/ask', methods=['POST', 'OPTIONS'])
def ask_question():
    if request.method == 'OPTIONS':
        return '', 200

    global agent_executor

    try:
        if agent_executor is None or retriever is None:
            if not current_pdf_path or not process_pdf(current_pdf_path):
                logger.error("No PDF has been uploaded or failed to process PDF.")
                return jsonify({'status': 'error', 'message': 'Please upload a PDF first or re-upload if you encounter issues.'}), 400

        if not groq_api_key:
            logger.error("GROQ API key not found in environment variables")
            return jsonify({'status': 'error', 'message': 'GROQ API key is not configured. Please check your .env file.'}), 500

        data = request.json
        if not data:
            logger.error("No JSON data in request")
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400

        query = data.get('query', '')
        if not query:
            logger.error("Empty query received")
            return jsonify({'status': 'error', 'message': 'Query is empty'}), 400

        logger.info(f"Processing question: {query}")

        # === SPECIAL CASE: PDF TITLE ===
        if "title" in query.lower():
            title = get_pdf_title(current_pdf_path)
            return jsonify({'status': 'success', 'answer': title})

        try:
            start_time = time.time()
            response = agent_executor.invoke({
                "input": query,
                "context": "",
                "agent_scratchpad": ""
            })

            # Extract the final answer from the response
            if isinstance(response, dict) and 'output' in response:
                answer = response['output']
            else:
                answer = str(response)

            response_time = time.time() - start_time

            logger.info(f"Successfully generated answer in {response_time:.2f} seconds")

            # Fallback for empty or not found answers
            if not answer or "I cannot find" in answer or "max iterations" in answer.lower():
                logger.info("Agent did not find a good answer, using retriever fallback.")
                results = retriever.get_relevant_documents(query)
                if results:
                    return jsonify({'status': 'success', 'answer': results[0].page_content, 'response_time': f"{response_time:.2f}", 'note': 'Direct retriever fallback used.'})
                else:
                    return jsonify({'status': 'error', 'message': 'Could not find the answer in the PDF. Try asking about the content or check if your question is specific enough.'}), 200

            return jsonify({
                'status': 'success',
                'answer': answer,
                'response_time': f"{response_time:.2f}"
            })
        except Exception as e:
            logger.error(f"Agent or LLM error: {str(e)}. Using retriever fallback.")
            results = retriever.get_relevant_documents(query)
            if results:
                return jsonify({'status': 'success', 'answer': results[0].page_content, 'note': 'Direct retriever fallback used after agent error.'})
            return jsonify({'status': 'error', 'message': 'There was a problem with the AI function call and no fallback answer was found.'}), 500

    except Exception as e:
        logger.error(f"Unexpected error in ask_question: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'Unexpected error: {str(e)}'}), 500

# Add this endpoint for direct retriever testing
@app.route('/test_retriever', methods=['POST'])
def test_retriever():
    global retriever
    try:
        data = request.json
        query = data.get('query', 'summary')
        if retriever is None:
            return jsonify({'status': 'error', 'message': 'No retriever available. Upload and process a PDF first.'}), 400
        results = retriever.get_relevant_documents(query)
        return jsonify({'status': 'success', 'results': [doc.page_content for doc in results]})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    # Disable debug mode and use_reloader to prevent server restarts
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)