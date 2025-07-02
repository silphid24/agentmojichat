/**
 * MOJI WebChat V2 - Modern UI
 * React-style chat interface built with Vanilla JavaScript + Tailwind CSS
 */

class MojiWebChatV2 {
    constructor(config) {
        this.config = {
            wsUrl: config.wsUrl || `ws://${window.location.host}/api/v1/adapters/webchat/ws`,
            userId: config.userId || this.generateUserId(),
            userName: config.userName || 'Guest User',
            reconnectInterval: 5000,
            maxReconnectAttempts: 5,
            ...config
        };
        
        // State management (React-like)
        this.state = {
            isOpen: false,
            isMinimized: false,
            showWelcomeMessage: true,
            messages: [],
            inputValue: '',
            isLoading: false,
            isConnected: false,
            currentProvider: 'openai',
            currentModel: 'gpt-3.5-turbo',
            temperature: 0.7,
            useRag: true
        };
        
        // WebSocket connection
        this.ws = null;
        this.reconnectAttempts = 0;
        this.messageQueue = [];
        
        // DOM elements cache
        this.elements = {};
        
        // Welcome message timer
        this.welcomeTimer = null;
        
        // Model providers configuration
        this.providers = {
            'openai': { name: 'OpenAI', icon: '🤖', models: [] },
            'anthropic': { name: 'Anthropic', icon: '🧠', models: [] },
            'custom': { name: 'Workstation LLM', icon: '🖥️', models: [] },
            'deepseek': { name: 'DeepSeek', icon: '🚀', models: [] },
            'deepseek-local': { name: 'DeepSeek (Local)', icon: '💻', models: [] },
            'exaone-local': { name: 'EXAONE (Local)', icon: '🔮', models: [] }
        };
        
        this.init();
    }
    
    generateUserId() {
        return `user_${Math.random().toString(36).substr(2, 9)}`;
    }
    
    init() {
        this.cacheElements();
        this.setupEventListeners();
        this.startWelcomeMessageTimer();
        this.initializeWelcomeMessages();
    }
    
    cacheElements() {
        this.elements = {
            chatWidget: document.getElementById('chat-widget'),
            welcomeMessage: document.getElementById('welcome-message'),
            chatWindow: document.getElementById('chat-window'),
            chatContent: document.getElementById('chat-content'),
            messagesContainer: document.getElementById('messages-container'),
            typingIndicator: document.getElementById('typing-indicator'),
            messageForm: document.getElementById('message-form'),
            messageInput: document.getElementById('message-input'),
            sendBtn: document.getElementById('send-btn'),
            chatToggle: document.getElementById('chat-toggle'),
            chatIcon: document.getElementById('chat-icon'),
            minimizeBtn: document.getElementById('minimize-btn'),
            closeBtn: document.getElementById('close-btn'),
            connectionStatus: document.getElementById('connection-status'),
            quickReplyBtns: document.querySelectorAll('.quick-reply-btn'),
            providerSelector: document.getElementById('provider-selector'),
            modelSelector: document.getElementById('model-selector'),
            ragToggle: document.getElementById('rag-toggle'),
            temperatureSlider: document.getElementById('temperature-slider'),
            tempValue: document.getElementById('temp-value')
        };
    }
    
    setupEventListeners() {
        // Chat toggle button
        this.elements.chatToggle.addEventListener('click', () => this.toggleChat());
        
        // Window controls
        this.elements.closeBtn.addEventListener('click', () => this.closeChat());
        this.elements.minimizeBtn.addEventListener('click', () => this.toggleMinimize());
        
        // Message form
        this.elements.messageForm.addEventListener('submit', (e) => this.handleSendMessage(e));
        
        // Input field
        this.elements.messageInput.addEventListener('input', (e) => {
            this.setState({ inputValue: e.target.value });
        });
        
        // Quick reply buttons
        this.elements.quickReplyBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const text = btn.getAttribute('data-text');
                this.handleQuickReply(text);
            });
        });
        
        // Provider selector
        if (this.elements.providerSelector) {
            this.elements.providerSelector.addEventListener('change', async (e) => {
                await this.switchProvider(e.target.value);
            });
        }
        
        // Model selector
        if (this.elements.modelSelector) {
            this.elements.modelSelector.addEventListener('change', (e) => {
                this.setState({ currentModel: e.target.value });
                this.updateModelInfo();
            });
        }
        
        // RAG toggle
        if (this.elements.ragToggle) {
            this.elements.ragToggle.addEventListener('change', (e) => {
                this.setState({ useRag: e.target.checked });
                const ragStatus = e.target.checked ? 'ON' : 'OFF';
                this.addSystemMessage(`RAG 모드: ${ragStatus}`);
            });
        }
        
        // Temperature slider
        if (this.elements.temperatureSlider) {
            this.elements.temperatureSlider.addEventListener('input', (e) => {
                const temperature = parseFloat(e.target.value);
                this.setState({ temperature });
                if (this.elements.tempValue) {
                    this.elements.tempValue.textContent = temperature.toFixed(1);
                }
            });
            
            this.elements.temperatureSlider.addEventListener('change', (e) => {
                const tempValue = parseFloat(e.target.value);
                let tempDescription = '';
                if (tempValue <= 0.3) tempDescription = '매우 정확';
                else if (tempValue <= 0.7) tempDescription = '균형적';
                else if (tempValue <= 1.2) tempDescription = '창의적';
                else tempDescription = '매우 창의적';
                
                this.addSystemMessage(`창의성 설정: ${tempValue} (${tempDescription})`);
            });
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.state.isOpen) {
                this.closeChat();
            }
        });
        
        // Handle window resize for mobile responsiveness
        window.addEventListener('resize', () => {
            if (this.state.isOpen) {
                this.updateMinimizedState();
            }
        });
    }
    
    startWelcomeMessageTimer() {
        this.welcomeTimer = setTimeout(() => {
            this.setState({ showWelcomeMessage: false });
            this.hideWelcomeMessage();
        }, 3000);
    }
    
    initializeWelcomeMessages() {
        // Show welcome message initially
        if (this.state.showWelcomeMessage) {
            this.showWelcomeMessage();
        }
    }
    
    // React-like state management
    setState(newState) {
        const prevState = { ...this.state };
        this.state = { ...this.state, ...newState };
        this.updateUI(prevState);
    }
    
    updateUI(prevState) {
        // Update welcome message visibility
        if (prevState.showWelcomeMessage !== this.state.showWelcomeMessage) {
            if (this.state.showWelcomeMessage && !this.state.isOpen) {
                this.showWelcomeMessage();
            } else {
                this.hideWelcomeMessage();
            }
        }
        
        // Update chat window visibility
        if (prevState.isOpen !== this.state.isOpen) {
            console.log('Chat window visibility changing:', prevState.isOpen, '->', this.state.isOpen);
            if (this.state.isOpen) {
                this.showChatWindow();
            } else {
                this.hideChatWindow();
            }
        }
        
        // Update minimized state
        if (prevState.isMinimized !== this.state.isMinimized) {
            this.updateMinimizedState();
        }
        
        // Update chat icon
        if (prevState.isOpen !== this.state.isOpen) {
            this.updateChatIcon();
        }
        
        // Update input state
        if (prevState.inputValue !== this.state.inputValue) {
            this.elements.messageInput.value = this.state.inputValue;
        }
        
        // Update send button state
        if (prevState.isLoading !== this.state.isLoading || prevState.inputValue !== this.state.inputValue) {
            this.updateSendButton();
        }
        
        // Update connection status
        if (prevState.isConnected !== this.state.isConnected) {
            this.updateConnectionStatus();
        }
    }
    
    showWelcomeMessage() {
        this.elements.welcomeMessage.style.display = 'block';
        this.elements.welcomeMessage.className = 'mr-4 mb-2 p-3 bg-white rounded-lg shadow-md border welcome-message-enter';
    }
    
    hideWelcomeMessage() {
        this.elements.welcomeMessage.className = 'mr-4 mb-2 p-3 bg-white rounded-lg shadow-md border welcome-message-exit';
        setTimeout(() => {
            this.elements.welcomeMessage.style.display = 'none';
        }, 300);
    }
    
    showChatWindow() {
        // Reset any previous styles
        this.elements.chatWindow.style.display = 'block';
        this.elements.chatWindow.style.visibility = 'visible';
        this.elements.chatWindow.style.opacity = '1';
        
        // Force reflow before adding animation class
        this.elements.chatWindow.offsetHeight;
        
        // Apply animation class
        this.elements.chatWindow.className = 'absolute glass-effect rounded-3xl shadow-2xl overflow-hidden chat-window-enter';
        
        // Update chat content height based on minimized state
        this.updateMinimizedState();
        
        // Connect WebSocket if not connected
        if (!this.state.isConnected) {
            this.connect();
        }
        
        // Add initial messages if empty
        if (this.state.messages.length === 0) {
            this.addInitialMessages();
        }
        
        // Load providers and models
        this.loadProviders();
        
        // Focus input
        setTimeout(() => {
            if (this.elements.messageInput) {
                this.elements.messageInput.focus();
            }
        }, 400);
    }
    
    hideChatWindow() {
        this.elements.chatWindow.className = 'absolute glass-effect rounded-3xl shadow-2xl overflow-hidden chat-window-exit';
        setTimeout(() => {
            this.elements.chatWindow.style.display = 'none';
            this.elements.chatWindow.style.visibility = 'hidden';
            this.elements.chatWindow.style.opacity = '0';
        }, 300);
    }
    
    updateMinimizedState() {
        const isMobile = window.innerWidth <= 640;
        const height = this.state.isMinimized ? '72px' : (isMobile ? 'calc(100vh - 8rem)' : '600px');
        this.elements.chatWindow.style.height = height;
        
        if (this.state.isMinimized) {
            this.elements.chatContent.style.display = 'none';
        } else {
            this.elements.chatContent.style.display = 'flex';
            // Recalculate chat content height
            const headerHeight = this.elements.chatWindow.querySelector('.glass-header').offsetHeight;
            const chatContentHeight = `calc(${height} - ${headerHeight}px)`;
            this.elements.chatContent.style.height = chatContentHeight;
        }
    }
    
    updateChatIcon() {
        const iconHtml = this.state.isOpen ? 
            '<svg class="w-7 h-7 text-white" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>' :
            '<img src="/static/icons/moji-character.svg" alt="MOJI" class="w-10 h-10">';
        
        this.elements.chatIcon.className = 'absolute flex items-center justify-center icon-rotate-enter';
        this.elements.chatIcon.innerHTML = iconHtml;
    }
    
    updateSendButton() {
        const disabled = this.state.isLoading || !this.state.inputValue.trim();
        this.elements.sendBtn.disabled = disabled;
        this.elements.messageInput.disabled = this.state.isLoading;
    }
    
    updateConnectionStatus() {
        if (this.state.isConnected) {
            this.elements.connectionStatus.style.display = 'none';
        } else {
            this.elements.connectionStatus.style.display = 'block';
            this.elements.connectionStatus.textContent = '연결 대기중...';
            this.elements.connectionStatus.className = 'fixed top-4 right-4 px-4 py-2 rounded-lg bg-yellow-100 text-yellow-800 border border-yellow-300 text-sm font-medium';
        }
    }
    
    addInitialMessages() {
        const welcomeMessages = [
            { id: 'welcome-1', text: '안녕하세요! SMHACCP 도우미 모지입니다.', sender: 'bot' },
            { id: 'welcome-2', text: '회사에 대해 궁금한 점을 무엇이든 물어보세요.', sender: 'bot' }
        ];
        
        this.setState({ messages: welcomeMessages });
        this.renderMessages();
    }
    
    toggleChat() {
        console.log('Toggle chat clicked, current state:', this.state.isOpen);
        
        this.setState({ 
            isOpen: !this.state.isOpen,
            showWelcomeMessage: false
        });
        
        if (this.welcomeTimer) {
            clearTimeout(this.welcomeTimer);
        }
        
        console.log('New state after toggle:', this.state.isOpen);
    }
    
    closeChat() {
        this.setState({ isOpen: false });
    }
    
    toggleMinimize() {
        this.setState({ isMinimized: !this.state.isMinimized });
    }
    
    handleSendMessage(e) {
        e.preventDefault();
        
        if (!this.state.inputValue.trim() || this.state.isLoading) {
            return;
        }
        
        const userMessage = {
            id: `user-${Date.now()}`,
            text: this.state.inputValue,
            sender: 'user',
            timestamp: new Date().toISOString()
        };
        
        // Add user message
        this.setState({ 
            messages: [...this.state.messages, userMessage],
            inputValue: '',
            isLoading: true
        });
        
        this.renderMessages();
        this.showTypingIndicator(true);
        
        // Send message via WebSocket
        this.sendMessage(userMessage.text);
    }
    
    handleQuickReply(text) {
        this.setState({ inputValue: text });
        // Trigger form submission
        this.elements.messageForm.dispatchEvent(new Event('submit'));
    }
    
    renderMessages() {
        if (!this.elements.messagesContainer) return;
        
        this.elements.messagesContainer.innerHTML = '';
        
        this.state.messages.forEach(msg => {
            const messageDiv = this.createMessageElement(msg);
            this.elements.messagesContainer.appendChild(messageDiv);
        });
        
        // Force scroll to bottom after rendering
        setTimeout(() => {
            this.scrollToBottom();
        }, 50);
    }
    
    createMessageElement(message) {
        const messageDiv = document.createElement('div');
        
        if (message.sender === 'system') {
            messageDiv.className = 'flex justify-center';
            const bubbleDiv = document.createElement('div');
            bubbleDiv.className = 'max-w-[80%] rounded-full px-3 py-1.5 bg-yellow-100 text-yellow-800 border border-yellow-300';
            
            const textP = document.createElement('p');
            textP.className = 'text-xs font-medium text-center';
            textP.textContent = message.text;
            
            bubbleDiv.appendChild(textP);
            messageDiv.appendChild(bubbleDiv);
        } else {
            messageDiv.className = `flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`;
            
            const bubbleDiv = document.createElement('div');
            bubbleDiv.className = `max-w-[80%] rounded-2xl px-4 py-2.5 ${
                message.sender === 'user'
                    ? 'bg-blue-500 text-white rounded-br-lg'
                    : 'bg-white text-gray-800 rounded-bl-lg border border-gray-200/80'
            }`;
            
            const textP = document.createElement('p');
            textP.className = 'text-sm leading-relaxed';
            textP.textContent = message.text;
            
            bubbleDiv.appendChild(textP);
            messageDiv.appendChild(bubbleDiv);
        }
        
        return messageDiv;
    }
    
    showTypingIndicator(show = true) {
        this.elements.typingIndicator.style.display = show ? 'block' : 'none';
        if (show) {
            this.scrollToBottom();
        }
    }
    
    scrollToBottom() {
        setTimeout(() => {
            if (this.elements.messagesContainer) {
                this.elements.messagesContainer.scrollTop = this.elements.messagesContainer.scrollHeight;
            }
        }, 100);
    }
    
    // WebSocket methods (integrated from original)
    connect() {
        try {
            this.ws = new WebSocket(this.config.wsUrl);
            
            this.ws.onopen = () => this.onOpen();
            this.ws.onmessage = (event) => this.onMessage(event);
            this.ws.onclose = () => this.onClose();
            this.ws.onerror = (error) => this.onError(error);
        } catch (error) {
            console.error('WebSocket connection error:', error);
            this.updateConnectionError('연결 실패');
        }
    }
    
    onOpen() {
        console.log('WebSocket connected');
        this.setState({ isConnected: true });
        this.reconnectAttempts = 0;
        
        // Send authentication
        this.ws.send(JSON.stringify({
            user_id: this.config.userId,
            user_name: this.config.userName
        }));
        
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
                    this.addBotMessage(data.text, data.timestamp);
                    break;
                case 'system':
                    this.addSystemMessage(data.text, data.timestamp);
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
        this.setState({ isConnected: false });
        
        // Attempt to reconnect
        if (this.reconnectAttempts < this.config.maxReconnectAttempts) {
            this.reconnectAttempts++;
            this.updateConnectionError(`재연결 시도 중... (${this.reconnectAttempts}/${this.config.maxReconnectAttempts})`);
            setTimeout(() => this.connect(), this.config.reconnectInterval);
        } else {
            this.updateConnectionError('연결 실패 - 페이지를 새로고침하세요');
        }
    }
    
    onError(error) {
        console.error('WebSocket error:', error);
        this.addSystemMessage('연결 오류가 발생했습니다.');
    }
    
    sendMessage(text) {
        if (!text || !text.trim()) return;
        
        if (!this.state.isConnected) {
            this.messageQueue.push(text);
            this.addSystemMessage('연결이 끊겼습니다. 재연결 중...');
            return;
        }
        
        const message = {
            type: 'text',
            text: text,
            timestamp: new Date().toISOString(),
            provider: this.state.currentProvider,
            model: this.state.currentModel,
            useRag: this.state.useRag,
            temperature: this.state.temperature
        };
        
        this.ws.send(JSON.stringify(message));
    }
    
    addBotMessage(text, timestamp = null) {
        const botMessage = {
            id: `bot-${Date.now()}`,
            text: text,
            sender: 'bot',
            timestamp: timestamp || new Date().toISOString()
        };
        
        this.setState({
            messages: [...this.state.messages, botMessage],
            isLoading: false
        });
        
        this.renderMessages();
        this.showTypingIndicator(false);
    }
    
    addSystemMessage(text, timestamp = null) {
        const systemMessage = {
            id: `system-${Date.now()}`,
            text: text,
            sender: 'system',
            timestamp: timestamp || new Date().toISOString()
        };
        
        this.setState({
            messages: [...this.state.messages, systemMessage]
        });
        
        this.renderMessages();
    }
    
    updateConnectionError(message) {
        this.elements.connectionStatus.style.display = 'block';
        this.elements.connectionStatus.textContent = message;
        this.elements.connectionStatus.className = 'fixed top-4 right-4 px-4 py-2 rounded-lg bg-red-100 text-red-800 border border-red-300 text-sm font-medium';
    }
    
    // Model and provider management methods
    async loadProviders() {
        try {
            // Load models for each provider
            for (const provider of Object.keys(this.providers)) {
                await this.loadModelsForProvider(provider);
            }
            
            this.updateProviderUI();
        } catch (error) {
            console.error('Failed to load providers:', error);
            // Still try to update UI with default values
            this.updateProviderUI();
        }
    }
    
    async loadModelsForProvider(provider) {
        try {
            const response = await fetch(`/api/v1/llm/models/${provider}/public`);
            
            if (response.ok) {
                const data = await response.json();
                this.providers[provider].models = data.models;
            } else {
                // Set default models if API fails
                this.setDefaultModels(provider);
            }
        } catch (error) {
            console.error(`Failed to load models for ${provider}:`, error);
            this.setDefaultModels(provider);
        }
    }
    
    setDefaultModels(provider) {
        const defaultModels = {
            'openai': [
                { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo' },
                { id: 'gpt-4', name: 'GPT-4' },
                { id: 'gpt-4-turbo', name: 'GPT-4 Turbo' }
            ],
            'anthropic': [
                { id: 'claude-3-sonnet-20240229', name: 'Claude 3 Sonnet' },
                { id: 'claude-3-haiku-20240307', name: 'Claude 3 Haiku' }
            ],
            'deepseek': [
                { id: 'deepseek-r1', name: 'DeepSeek R1' },
                { id: 'deepseek-chat', name: 'DeepSeek Chat' }
            ],
            'custom': [
                { id: 'llama3.2:latest', name: 'Llama 3.2' },
                { id: 'your-model-name', name: 'Custom Model' }
            ],
            'deepseek-local': [
                { id: 'deepseek-r1', name: 'DeepSeek R1 (Local)' }
            ],
            'exaone-local': [
                { id: 'exaone-3.0', name: 'EXAONE 3.0 (Local)' }
            ]
        };
        
        this.providers[provider].models = defaultModels[provider] || [
            { id: 'default-model', name: 'Default Model' }
        ];
    }
    
    async switchProvider(provider) {
        this.setState({ currentProvider: provider });
        const models = this.providers[provider].models;
        
        if (models.length > 0) {
            this.setState({ currentModel: models[0].id });
            this.updateModelSelector();
            
            const providerName = this.providers[provider].name;
            const modelName = models[0].name;
            this.addSystemMessage(`모델 변경: ${providerName} - ${modelName}`);
        }
    }
    
    updateProviderUI() {
        if (!this.elements.providerSelector) return;
        
        // Update provider selector
        this.elements.providerSelector.innerHTML = Object.entries(this.providers)
            .map(([key, provider]) => `
                <option value="${key}" ${key === this.state.currentProvider ? 'selected' : ''}>
                    ${provider.icon} ${provider.name}
                </option>
            `).join('');
        
        this.updateModelSelector();
    }
    
    updateModelSelector() {
        if (!this.elements.modelSelector) return;
        
        const models = this.providers[this.state.currentProvider].models;
        this.elements.modelSelector.innerHTML = models
            .map(model => `
                <option value="${model.id}" ${model.id === this.state.currentModel ? 'selected' : ''}>
                    ${model.name}
                </option>
            `).join('');
        
        this.updateModelInfo();
    }
    
    updateModelInfo() {
        // This could be used to show additional model information if needed
        console.log(`Current model: ${this.state.currentProvider} - ${this.state.currentModel}`);
    }
    
    // Static initialization method for widget embedding
    static init(config) {
        return new MojiWebChatV2(config);
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MojiWebChatV2;
}

// Global access
window.MojiWebChatV2 = MojiWebChatV2;