document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const pdfFileInput = document.getElementById('pdf-file');
    const fileNameDisplay = document.getElementById('file-name');
    const uploadStatus = document.getElementById('upload-status');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const messagesContainer = document.getElementById('messages');
    
    // Handle file selection display
    pdfFileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            fileNameDisplay.textContent = this.files[0].name;
        } else {
            fileNameDisplay.textContent = 'Choose a PDF file';
        }
    });
    
    // Handle file upload
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData();
        const fileInput = document.getElementById('pdf-file');
        
        if (fileInput.files.length === 0) {
            showUploadStatus('Please select a PDF file first', 'error');
            return;
        }
        
        formData.append('pdf_file', fileInput.files[0]);
        
        // Show loading status
        showUploadStatus('Uploading and processing your PDF...', 'loading');
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                showUploadStatus(data.message, 'success');
                enableChat();
                addSystemMessage('PDF processed successfully! You can now ask questions about the document.');
            } else {
                showUploadStatus(data.message, 'error');
            }
        })
        .catch(error => {
            showUploadStatus('Error: ' + error.message, 'error');
        });
    });
    
    // Handle chat submission
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const userMessage = userInput.value.trim();
        if (!userMessage) return;
        
        // Add user message to chat
        addUserMessage(userMessage);
        userInput.value = '';
        
        // Show typing indicator
        const typingIndicator = addTypingIndicator();
        
        // Send message to backend
        fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: userMessage })
        })
        .then(response => response.json())
        .then(data => {
            // Remove typing indicator
            typingIndicator.remove();
            
            if (data.status === 'success') {
                addBotMessage(data.answer, data.response_time);
            } else {
                addBotMessage('Error: ' + data.message);
            }
        })
        .catch(error => {
            // Remove typing indicator
            typingIndicator.remove();
            addBotMessage('Error: Failed to get a response. Please try again.');
        });
    });
    
    // Helper Functions
    function showUploadStatus(message, type) {
        uploadStatus.textContent = message;
        uploadStatus.style.display = 'block';
        
        uploadStatus.className = '';
        if (type === 'success') {
            uploadStatus.classList.add('status-success');
        } else if (type === 'error') {
            uploadStatus.classList.add('status-error');
        } else {
            uploadStatus.classList.add('status-loading');
        }
    }
    
    function enableChat() {
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.placeholder = "Ask a question about your document...";
        userInput.focus();
    }
    
    function addUserMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message user-message';
        
        messageElement.innerHTML = `
            <div class="message-content">
                <p>${escapeHtml(message)}</p>
            </div>
            <div class="message-meta">
                ${getCurrentTime()}
            </div>
        `;
        
        messagesContainer.appendChild(messageElement);
        scrollToBottom();
    }
    
    function addBotMessage(message, responseTime = null) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message bot-message';
        
        let metaContent = getCurrentTime();
        if (responseTime) {
            metaContent += ` • ${responseTime}s`;
        }
        
        messageElement.innerHTML = `
            <div class="message-content">
                <p>${formatMessage(message)}</p>
            </div>
            <div class="message-meta">
                ${metaContent}
            </div>
        `;
        
        messagesContainer.appendChild(messageElement);
        scrollToBottom();
    }
    
    function addSystemMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message system-message';
        
        messageElement.innerHTML = `
            <div class="message-content">
                <p>${message}</p>
            </div>
        `;
        
        messagesContainer.appendChild(messageElement);
        scrollToBottom();
    }
    
    function addTypingIndicator() {
        const indicatorElement = document.createElement('div');
        indicatorElement.className = 'message bot-message typing-indicator-container';
        
        indicatorElement.innerHTML = `
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        
        messagesContainer.appendChild(indicatorElement);
        scrollToBottom();
        
        return indicatorElement;
    }
    
    function getCurrentTime() {
        const now = new Date();
        return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    
    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    function escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
    
    function formatMessage(text) {
        // Convert line breaks to <br>
        text = text.replace(/\n/g, '<br>');
        
        // Simple markdown-like formatting
        // Bold: **text** -> <strong>text</strong>
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Italic: *text* -> <em>text</em>
        text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        return text;
    }
});