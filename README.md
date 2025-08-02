# DOCCUPY - Your personal Document Chatbot!

**Doccupy** is an advanced **RAG (Retrieval-Augmented Generation) chatbot** that allows you to **chat with your documents instantly**. By combining the power of **Large Language Models (LLMs)**, **LangChain**, and **Vector Databases**, Doccupy extracts the most relevant context from your files and provides **clear, structured, and Markdown-formatted responses** for effortless understanding.

Whether you are a **researcher, business professional, or developer**, Doccupy helps you **save time, boost productivity, and make your documents interactive**. Simply **upload any document**, and within seconds, you have your **own AI-powered assistant** ready to answer queries.

---

## ✨ Key Features

- 📥 **Instant Document Upload** – Upload PDF files and start chatting immediately.  
- 🤖 **RAG-Powered Context** – Delivers **accurate, context-aware responses** based on document content.  
- ⚡ **Vector Database Integration** – Enables **fast semantic search** and precise retrieval.  
- 📝 **Structured Markdown Answers** – Ensures clear, readable, and presentation-ready outputs.  
- 🔗 **LangChain Integration** – Manages document chunking, embedding, and intelligent response generation.  

---

## 🖥️ Modes of Operation

**Doccupy** provides two flexible ways to interact with your documents:

1. **`web.py`** – Interactive **web interface** powered by **Ollama Gemma2**.  
   - Supports **unlimited token length and unlimited requests**.  
   - **Clean UI** for effortless document exploration.
   - Delivers well-structured **Markdown responses** for easy readability and precise understanding.

2. **`doccupy.py`** – **Command-Line Interface (CLI)** powered by **Groq API**.  
   - **Ultra-fast data retrieval and answering**.  
   - Lightweight for **developers and automation scripts**.
     
**Doccupy transforms static files into dynamic knowledge companions**, helping you understand and utilize your documents faster than ever before.

---

# ⚡ How to Run Doccupy

Follow these steps to set up and run **Doccupy** on your system:

### 1️⃣ Install Python Requirements
Ensure Python 3.8+ is installed, then install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2️⃣ Install Ollama and Pull Gemma2:2b Model
- Download and install Ollama from the official website: [Ollama](https://ollama.com/download)
- Pull the Gemma2:2b model:
```bash
ollama pull gemma2:2b
```

### 3️⃣ Install Embedding Model – all-MiniLM-L6-v2
- Download the all-MiniLM-L6-v2 model from Hugging Face: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- Save or move the model into the `model/` directory in your project.

4️⃣ Run Doccupy
You can run Doccupy in two modes:

🌐 Web Interface (Ollama Gemma2 Model):
```bash
python web.py
```
   
💻 Command-Line Interface (Groq API)
```bash
python doccupy.py
```

---

## 🤝 Contribution
Contributions are welcomed to make Doccupy smarter and more reliable. Whether it’s fixing bugs, optimizing the RAG pipeline, enhancing the web or CLI interface, or improving documentation, every effort counts. To contribute, fork the repository, create a feature branch, commit your changes, and submit a pull request with a clear explanation. Please ensure your code is clean, well-documented, and that new dependencies are added to requirements.txt. Together, we can make Doccupy a more efficient and user-friendly document chatbot.

## 📜 License
Doccupy is licensed under the MIT License, allowing free use, modification, and distribution for personal and commercial purposes. Users must retain the original license and attribution in all copies or derivatives. See the LICENSE file for complete details.

## 📧 Contact

For questions, suggestions, or collaboration opportunities, feel free to reach out:   
📩 Email: soubhagyasrivastava240@gmail.com 
🌐 LinkedIn: [Soubhagya Srivastava](https://www.linkedin.com/in/soubhagya-srivastava-611408267/)  
We’d love to hear your feedback and ideas to make **Doccupy** even better!
