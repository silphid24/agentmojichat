/**
 * MOJI WebChat Client
 * WebSocket-based chat client for MOJI AI Assistant
 */

class MojiWebChat {
    constructor(config) {
        this.config = {
            wsUrl: config.wsUrl || `ws://${window.location.host}/api/v1/adapters/webchat/ws`,
            userId: config.userId || this.generateUserId(),
            userName: config.userName || 'Guest User',
            reconnectInterval: 5000,
            maxReconnectAttempts: 5,
            ...config
        };
        
        this.ws = null;
        this.reconnectAttempts = 0;
        this.messageQueue = [];
        this.isConnected = false;
        
        // DOM elements
        this.elements = {
            messages: document.getElementById('chat-messages'),
            input: document.getElementById('message-input'),
            sendButton: document.getElementById('send-button'),
            status: document.getElementById('connection-status'),
            typingIndicator: document.getElementById('typing-indicator')
        };
        
        this.setupEventListeners();
    }
    
    generateUserId() {
        return `user_${Math.random().toString(36).substr(2, 9)}`;
    }
    
    connect() {
        try {
            this.ws = new WebSocket(this.config.wsUrl);
            
            this.ws.onopen = () => this.onOpen();
            this.ws.onmessage = (event) => this.onMessage(event);
            this.ws.onclose = () => this.onClose();
            this.ws.onerror = (error) => this.onError(error);
        } catch (error) {
            console.error('WebSocket connection error:', error);
            this.updateStatus('연결 실패', 'error');
        }
    }
    
    onOpen() {
        console.log('WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        
        // Send authentication
        this.ws.send(JSON.stringify({
            user_id: this.config.userId,
            user_name: this.config.userName
        }));
        
        // Update UI
        this.updateStatus('연결됨', 'connected');
        this.enableInput();
        
        // Send queued messages
        while (this.messageQueue.length > 0) {
            const message = this.messageQueue.shift();
            this.sendMessage(message);
        }
    }
    
    onMessage(event) {
        try {
            const data = JSON.parse(event.data);
            console.log('Received message:', data);
            
            switch (data.type) {
                case 'text':
                    this.addMessage(data.text, 'bot', data.timestamp);
                    break;
                case 'system':
                    this.addMessage(data.text, 'system', data.timestamp);
                    break;
                case 'card':
                    this.addCard(data);
                    break;
                case 'buttons':
                    this.addButtons(data);
                    break;
                case 'typing':
                    this.showTypingIndicator(data.show);
                    break;
                default:
                    console.warn('Unknown message type:', data.type);
            }
        } catch (error) {
            console.error('Error parsing message:', error);
        }
    }
    
    onClose() {
        console.log('WebSocket disconnected');
        this.isConnected = false;
        this.updateStatus('연결 끊김', 'disconnected');
        this.disableInput();
        
        // Attempt to reconnect
        if (this.reconnectAttempts < this.config.maxReconnectAttempts) {
            this.reconnectAttempts++;
            this.updateStatus(`재연결 시도 중... (${this.reconnectAttempts}/${this.config.maxReconnectAttempts})`, 'reconnecting');
            setTimeout(() => this.connect(), this.config.reconnectInterval);
        } else {
            this.updateStatus('연결 실패 - 페이지를 새로고침하세요', 'error');
        }
    }
    
    onError(error) {
        console.error('WebSocket error:', error);
        this.addMessage('연결 오류가 발생했습니다.', 'system');
    }
    
    sendMessage(text) {
        if (!text || !text.trim()) return;
        
        if (!this.isConnected) {
            this.messageQueue.push(text);
            this.addMessage('연결이 끊겼습니다. 재연결 중...', 'system');
            return;
        }
        
        // Add user message to chat
        this.addMessage(text, 'user');
        
        // Send to server
        const message = {
            type: 'text',
            text: text,
            timestamp: new Date().toISOString()
        };
        
        this.ws.send(JSON.stringify(message));
        
        // Clear input
        this.elements.input.value = '';
        this.elements.input.focus();
    }
    
    addMessage(text, sender = 'bot', timestamp = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const textDiv = document.createElement('div');
        textDiv.textContent = text;
        messageDiv.appendChild(textDiv);
        
        if (timestamp && sender !== 'system') {
            const timeDiv = document.createElement('div');
            timeDiv.className = 'message-time';
            timeDiv.textContent = this.formatTime(timestamp);
            messageDiv.appendChild(timeDiv);
        }
        
        this.elements.messages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    addCard(data) {
        const cards = data.cards || [data];
        
        cards.forEach(card => {
            const cardDiv = document.createElement('div');
            cardDiv.className = 'message bot card';
            
            if (card.image) {
                const img = document.createElement('img');
                img.src = card.image;
                img.style.width = '100%';
                img.style.borderRadius = '8px';
                img.style.marginBottom = '10px';
                cardDiv.appendChild(img);
            }
            
            if (card.title) {
                const title = document.createElement('h3');
                title.textContent = card.title;
                title.style.marginBottom = '8px';
                cardDiv.appendChild(title);
            }
            
            if (card.subtitle) {
                const subtitle = document.createElement('p');
                subtitle.textContent = card.subtitle;
                subtitle.style.fontSize = '14px';
                subtitle.style.opacity = '0.8';
                subtitle.style.marginBottom = '8px';
                cardDiv.appendChild(subtitle);
            }
            
            if (card.text) {
                const text = document.createElement('p');
                text.textContent = card.text;
                cardDiv.appendChild(text);
            }
            
            if (card.buttons && card.buttons.length > 0) {
                const buttonContainer = document.createElement('div');
                buttonContainer.style.marginTop = '12px';
                buttonContainer.style.display = 'flex';
                buttonContainer.style.gap = '8px';
                buttonContainer.style.flexWrap = 'wrap';
                
                card.buttons.forEach(button => {
                    const btn = document.createElement('button');
                    btn.textContent = button.text;
                    btn.style.padding = '8px 16px';
                    btn.style.border = '1px solid #2563eb';
                    btn.style.borderRadius = '6px';
                    btn.style.background = 'white';
                    btn.style.color = '#2563eb';
                    btn.style.cursor = 'pointer';
                    btn.onclick = () => this.sendMessage(button.value);
                    buttonContainer.appendChild(btn);
                });
                
                cardDiv.appendChild(buttonContainer);
            }
            
            this.elements.messages.appendChild(cardDiv);
        });
        
        this.scrollToBottom();
    }
    
    addButtons(data) {
        const buttonDiv = document.createElement('div');
        buttonDiv.className = 'message bot';
        
        if (data.text) {
            const textDiv = document.createElement('div');
            textDiv.textContent = data.text;
            buttonDiv.appendChild(textDiv);
        }
        
        const buttonContainer = document.createElement('div');
        buttonContainer.style.marginTop = '12px';
        buttonContainer.style.display = 'flex';
        buttonContainer.style.gap = '8px';
        buttonContainer.style.flexWrap = 'wrap';
        
        data.buttons.forEach(button => {
            const btn = document.createElement('button');
            btn.textContent = button.text;
            btn.style.padding = '8px 16px';
            btn.style.border = '1px solid #2563eb';
            btn.style.borderRadius = '6px';
            btn.style.background = 'white';
            btn.style.color = '#2563eb';
            btn.style.cursor = 'pointer';
            btn.onclick = () => this.sendMessage(button.value);
            buttonContainer.appendChild(btn);
        });
        
        buttonDiv.appendChild(buttonContainer);
        this.elements.messages.appendChild(buttonDiv);
        this.scrollToBottom();
    }
    
    showTypingIndicator(show = true) {
        if (this.elements.typingIndicator) {
            this.elements.typingIndicator.style.display = show ? 'block' : 'none';
            if (show) {
                this.scrollToBottom();
            }
        }
    }
    
    updateStatus(text, type = 'info') {
        if (this.elements.status) {
            this.elements.status.textContent = text;
            this.elements.status.className = `status ${type}`;
        }
    }
    
    enableInput() {
        if (this.elements.input) {
            this.elements.input.disabled = false;
            this.elements.input.focus();
        }
        if (this.elements.sendButton) {
            this.elements.sendButton.disabled = false;
        }
    }
    
    disableInput() {
        if (this.elements.input) {
            this.elements.input.disabled = true;
        }
        if (this.elements.sendButton) {
            this.elements.sendButton.disabled = true;
        }
    }
    
    scrollToBottom() {
        if (this.elements.messages) {
            this.elements.messages.scrollTop = this.elements.messages.scrollHeight;
        }
    }
    
    formatTime(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleTimeString('ko-KR', {
            hour: '2-digit',
            minute: '2-digit'
        });
    }
    
    setupEventListeners() {
        // Send button click
        if (this.elements.sendButton) {
            this.elements.sendButton.addEventListener('click', () => {
                this.sendMessage(this.elements.input.value);
            });
        }
        
        // Enter key press
        if (this.elements.input) {
            this.elements.input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage(this.elements.input.value);
                }
            });
        }
    }
    
    // Static initialization method for widget embedding
    static init(config) {
        // Create widget container if not exists
        let container = document.getElementById('moji-webchat-widget');
        if (!container) {
            container = document.createElement('div');
            container.id = 'moji-webchat-widget';
            document.body.appendChild(container);
        }
        
        // Create chat UI
        container.innerHTML = `
            <div class="moji-chat-widget">
                <div class="moji-chat-toggle">
                    <button id="moji-toggle-button">💬</button>
                </div>
                <div class="moji-chat-window" id="moji-chat-window" style="display: none;">
                    <div class="chat-header">
                        <h3>${config.title || 'MOJI Assistant'}</h3>
                        <button id="moji-close-button">✕</button>
                    </div>
                    <div class="chat-messages" id="chat-messages"></div>
                    <div class="typing-indicator" id="typing-indicator" style="display: none;">
                        <span></span><span></span><span></span>
                    </div>
                    <div class="chat-input-container">
                        <input type="text" id="message-input" placeholder="${config.placeholder || 'Type a message...'}" />
                        <button id="send-button">Send</button>
                    </div>
                </div>
            </div>
        `;
        
        // Add widget styles
        const style = document.createElement('style');
        style.textContent = `
            .moji-chat-widget {
                position: fixed;
                ${config.position === 'bottom-left' ? 'left: 20px;' : 'right: 20px;'}
                bottom: 20px;
                z-index: 1000;
            }
            .moji-chat-toggle button {
                width: 60px;
                height: 60px;
                border-radius: 50%;
                background: #2563eb;
                border: none;
                font-size: 24px;
                cursor: pointer;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .moji-chat-window {
                position: absolute;
                ${config.position === 'bottom-left' ? 'left: 0;' : 'right: 0;'}
                bottom: 80px;
                width: 350px;
                height: 500px;
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                display: flex;
                flex-direction: column;
            }
        `;
        document.head.appendChild(style);
        
        // Initialize chat instance
        const chat = new MojiWebChat(config);
        
        // Setup toggle functionality
        document.getElementById('moji-toggle-button').addEventListener('click', () => {
            const window = document.getElementById('moji-chat-window');
            if (window.style.display === 'none') {
                window.style.display = 'flex';
                chat.connect();
            } else {
                window.style.display = 'none';
            }
        });
        
        document.getElementById('moji-close-button').addEventListener('click', () => {
            document.getElementById('moji-chat-window').style.display = 'none';
        });
        
        return chat;
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MojiWebChat;
}