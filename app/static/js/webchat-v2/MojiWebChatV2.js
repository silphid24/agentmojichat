/**
 * MOJI WebChat V2 - Modularized version
 * Main entry point that orchestrates all modules
 */
import { EventEmitter } from './core/EventEmitter.js';
import { StateManager, createInitialState } from './core/StateManager.js';
import { WebSocketManager } from './modules/WebSocketManager.js';
import { UIRenderer } from './modules/UIRenderer.js';
import { MessageHandler } from './modules/MessageHandler.js';
import { ConfigManager } from './modules/ConfigManager.js';

export class MojiWebChatV2 {
    constructor(userConfig = {}) {
        // Core systems
        this.events = new EventEmitter();
        this.config = new ConfigManager(this.events);
        
        // Validate and merge config
        const config = this.config.mergeWithDefaults(userConfig);
        const errors = this.config.validateConfig(config);
        
        if (errors.length > 0) {
            throw new Error(`Invalid configuration: ${errors.join(', ')}`);
        }
        
        this.settings = config;
        
        // Initialize state
        const initialState = createInitialState();
        initialState.userId = config.userId;
        initialState.userName = config.userName;
        
        // Load saved settings if available
        const savedSettings = this.config.loadSettings();
        if (savedSettings) {
            Object.assign(initialState, savedSettings);
        }
        
        this.state = new StateManager(initialState);
        
        // Initialize modules
        this.ws = new WebSocketManager(config, this.events);
        this.elements = {};
        this.ui = null;
        this.messages = null;
        
        // Timers
        this.welcomeTimer = null;
        
        // Initialize when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.init());
        } else {
            this.init();
        }
    }

    init() {
        try {
            this.cacheElements();
            this.ui = new UIRenderer(this.elements, this.events);
            this.messages = new MessageHandler(this.state, this.ws, this.events);
            
            this.setupEventListeners();
            this.setupStateSubscribers();
            this.startWelcomeTimer();
            
            if (this.settings.debug) {
                console.log('MojiWebChatV2 initialized successfully');
            }
        } catch (error) {
            console.error('Failed to initialize MojiWebChatV2:', error);
            this.showError('초기화 실패: ' + error.message);
        }
    }

    cacheElements() {
        const requiredElements = [
            'chat-widget', 'chat-toggle', 'chat-window', 'chat-content',
            'messages-container', 'message-form', 'message-input', 'send-btn'
        ];
        
        const missingElements = [];
        
        requiredElements.forEach(id => {
            const element = document.getElementById(id);
            if (!element) {
                missingElements.push(id);
            }
            // Convert kebab-case to camelCase properly
            const camelKey = id.replace(/-([a-z])/g, (match, letter) => letter.toUpperCase());
            this.elements[camelKey] = element;
        });
        
        // Cache optional elements
        const optionalElements = [
            'welcome-message', 'typing-indicator', 'chat-icon',
            'minimize-btn', 'close-btn', 'connection-status',
            'provider-selector', 'model-selector', 'rag-toggle',
            'temperature-slider', 'temp-value'
        ];
        
        optionalElements.forEach(id => {
            // Convert kebab-case to camelCase properly
            const camelKey = id.replace(/-([a-z])/g, (match, letter) => letter.toUpperCase());
            this.elements[camelKey] = document.getElementById(id);
        });
        
        // Quick reply buttons (NodeList)
        this.elements.quickReplyBtns = document.querySelectorAll('.quick-reply-btn');
        
        if (missingElements.length > 0) {
            throw new Error(`Required elements not found: ${missingElements.join(', ')}`);
        }
    }

    setupEventListeners() {
        // UI Events
        this.elements.chatToggle?.addEventListener('click', () => this.toggleChat());
        this.elements.closeBtn?.addEventListener('click', () => this.closeChat());
        this.elements.minimizeBtn?.addEventListener('click', () => this.toggleMinimize());
        
        // Message form
        this.elements.messageForm?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleSendMessage();
        });
        
        // Input field
        this.elements.messageInput?.addEventListener('input', (e) => {
            this.state.setState({ inputValue: e.target.value });
        });
        
        // Enter key handling
        this.elements.messageInput?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleSendMessage();
            }
        });
        
        // Quick replies
        this.elements.quickReplyBtns?.forEach(btn => {
            btn.addEventListener('click', () => {
                const text = btn.getAttribute('data-text');
                this.handleQuickReply(text);
            });
        });
        
        // Settings controls
        this.elements.providerSelector?.addEventListener('change', (e) => {
            this.switchProvider(e.target.value);
        });
        
        this.elements.modelSelector?.addEventListener('change', (e) => {
            this.state.setState({ currentModel: e.target.value });
            this.saveSettings();
        });
        
        this.elements.ragToggle?.addEventListener('change', (e) => {
            this.state.setState({ useRag: e.target.checked });
            this.messages.addMessage({
                id: `system-${Date.now()}`,
                text: `RAG 모드: ${e.target.checked ? '활성' : '비활성'}`,
                sender: 'system',
                timestamp: new Date().toISOString()
            });
            this.saveSettings();
        });
        
        this.elements.temperatureSlider?.addEventListener('input', (e) => {
            const temperature = parseFloat(e.target.value);
            this.state.setState({ temperature });
            if (this.elements.tempValue) {
                this.elements.tempValue.textContent = temperature.toFixed(1);
            }
        });
        
        this.elements.temperatureSlider?.addEventListener('change', (e) => {
            const temperature = parseFloat(e.target.value);
            const info = this.config.getTemperatureInfo(temperature);
            this.messages.addMessage({
                id: `system-${Date.now()}`,
                text: `창의성 설정: ${temperature.toFixed(1)} (${info.name})`,
                sender: 'system',
                timestamp: new Date().toISOString()
            });
            this.saveSettings();
        });
        
        // Global shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.state.getState().isOpen) {
                this.closeChat();
            }
        });
        
        // Window resize
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                if (this.state.getState().isOpen) {
                    this.ui.updateMinimizedState(this.state.getState().isMinimized);
                }
            }, 200);
        });
    }

    setupStateSubscribers() {
        // Subscribe to state changes
        this.state.subscribe((newState, prevState) => {
            // Chat visibility
            if (prevState.isOpen !== newState.isOpen) {
                if (newState.isOpen) {
                    this.ui.showChatWindow();
                    if (!newState.isConnected) {
                        this.ws.connect();
                    }
                    if (newState.messages.length === 0) {
                        this.addWelcomeMessages();
                    }
                    this.loadProviders();
                } else {
                    this.ui.hideChatWindow();
                }
                this.ui.updateChatIcon(newState.isOpen);
            }
            
            // Welcome message
            if (prevState.showWelcomeMessage !== newState.showWelcomeMessage) {
                if (newState.showWelcomeMessage && !newState.isOpen) {
                    this.ui.showWelcomeMessage();
                } else {
                    this.ui.hideWelcomeMessage();
                }
            }
            
            // Minimized state
            if (prevState.isMinimized !== newState.isMinimized) {
                this.ui.updateMinimizedState(newState.isMinimized);
            }
            
            // Messages
            if (prevState.messages !== newState.messages) {
                this.ui.renderMessages(newState.messages);
            }
            
            // Input state
            if (prevState.inputValue !== newState.inputValue && this.elements.messageInput) {
                this.elements.messageInput.value = newState.inputValue;
            }
            
            // Loading state
            if (prevState.isLoading !== newState.isLoading || 
                prevState.inputValue !== newState.inputValue) {
                this.ui.updateSendButton(newState.isLoading, newState.inputValue.trim().length > 0);
            }
            
            // Connection state
            if (prevState.isConnected !== newState.isConnected) {
                this.ui.updateConnectionStatus(newState.isConnected ? 'connected' : 'connecting');
            }
        });
        
        // WebSocket events
        this.events.on('connection:open', () => {
            console.log('WebSocket connected successfully');
            this.state.setState({ 
                isConnected: true,
                reconnectAttempts: 0,
                connectionError: null
            });
        });
        
        this.events.on('connection:close', () => {
            console.log('WebSocket connection closed');
            this.state.setState({ isConnected: false });
        });
        
        this.events.on('connection:error', (data) => {
            console.error('WebSocket connection error:', data.error);
            this.state.setState({ 
                connectionError: data.error,
                isConnected: false
            });
            this.ui.updateConnectionStatus('error');
        });
        
        this.events.on('connection:reconnecting', (data) => {
            console.log(`Reconnecting... attempt ${data.attempt}/${data.maxAttempts}`);
            this.state.setState({ reconnectAttempts: data.attempt });
            this.ui.updateConnectionStatus('reconnecting');
        });
        
        this.events.on('connection:failed', () => {
            console.error('WebSocket connection failed after max attempts');
            this.state.setState({ 
                isConnected: false,
                connectionError: 'Connection failed'
            });
            this.ui.updateConnectionStatus('error');
            
            // 사용자에게 연결 실패 메시지 표시
            if (this.messages) {
                this.messages.addMessage({
                    id: `error-${Date.now()}`,
                    text: '서버 연결에 실패했습니다. 페이지를 새로고침해 주세요.',
                    sender: 'system',
                    timestamp: new Date().toISOString()
                });
            }
        });
        
        this.events.on('typing:update', (show) => {
            this.ui.showTypingIndicator(show);
        });
    }

    // Chat control methods
    toggleChat() {
        const isOpen = !this.state.getState().isOpen;
        this.state.setState({ 
            isOpen,
            showWelcomeMessage: false
        });
        
        if (this.welcomeTimer) {
            clearTimeout(this.welcomeTimer);
            this.welcomeTimer = null;
        }
    }

    closeChat() {
        this.state.setState({ isOpen: false });
    }

    toggleMinimize() {
        this.state.setState({ 
            isMinimized: !this.state.getState().isMinimized 
        });
    }

    // Message handling
    handleSendMessage() {
        const text = this.state.getState().inputValue;
        
        // Check for commands first
        if (this.messages.handleCommand(text)) {
            this.state.setState({ inputValue: '' });
            return;
        }
        
        // Validate and send regular message
        if (this.messages.validateMessage(text)) {
            this.messages.sendMessage(text);
        }
    }

    handleQuickReply(text) {
        this.state.setState({ inputValue: text });
        this.handleSendMessage();
    }

    // Welcome messages
    startWelcomeTimer() {
        if (this.settings.animation !== false) {
            this.welcomeTimer = setTimeout(() => {
                this.state.setState({ showWelcomeMessage: true });
            }, 2000);
        }
    }

    addWelcomeMessages() {
        this.messages.addMessage({
            id: 'welcome-1',
            text: '안녕하세요! SMHACCP 도우미 모지입니다.',
            sender: 'bot',
            timestamp: new Date().toISOString()
        });
        
        this.messages.addMessage({
            id: 'welcome-2',
            text: '무엇을 도와드릴까요?',
            sender: 'bot',
            timestamp: new Date().toISOString()
        });
    }

    // Provider management
    async loadProviders() {
        const providers = this.config.getAllProviders();
        
        // Update provider selector
        if (this.elements.providerSelector) {
            this.elements.providerSelector.innerHTML = providers
                .map(provider => `
                    <option value="${provider.id}" ${provider.id === this.state.getState().currentProvider ? 'selected' : ''}>
                        ${provider.icon} ${provider.name}
                    </option>
                `).join('');
        }
        
        // Load models for current provider
        await this.updateModelSelector();
    }

    async switchProvider(providerId) {
        const provider = this.config.getProvider(providerId);
        if (!provider) return;
        
        this.state.setState({ 
            currentProvider: providerId,
            currentModel: provider.defaultModel
        });
        
        await this.updateModelSelector();
        
        this.messages.addMessage({
            id: `system-${Date.now()}`,
            text: `모델 변경: ${provider.name}`,
            sender: 'system',
            timestamp: new Date().toISOString()
        });
        
        this.saveSettings();
    }

    async updateModelSelector() {
        const providerId = this.state.getState().currentProvider;
        const models = await this.config.loadProviderModels(providerId);
        
        if (this.elements.modelSelector) {
            this.elements.modelSelector.innerHTML = models
                .map(model => `
                    <option value="${model.id}" ${model.id === this.state.getState().currentModel ? 'selected' : ''}>
                        ${model.name}
                    </option>
                `).join('');
        }
    }

    // Settings persistence
    saveSettings() {
        const state = this.state.getState();
        this.config.saveSettings({
            currentProvider: state.currentProvider,
            currentModel: state.currentModel,
            temperature: state.temperature,
            useRag: state.useRag
        });
    }

    // Error handling
    showError(message) {
        console.error('MojiWebChatV2 Error:', message);
        
        // Show error in UI if possible
        if (this.messages) {
            this.messages.addMessage({
                id: `error-${Date.now()}`,
                text: `오류: ${message}`,
                sender: 'system',
                timestamp: new Date().toISOString()
            });
        }
    }

    // Public API
    open() {
        this.state.setState({ isOpen: true });
    }

    close() {
        this.state.setState({ isOpen: false });
    }

    sendMessage(text) {
        if (this.messages?.validateMessage(text)) {
            this.messages.sendMessage(text);
        }
    }

    destroy() {
        // Clean up
        if (this.welcomeTimer) {
            clearTimeout(this.welcomeTimer);
        }
        
        this.ws.disconnect();
        this.events.off();
        
        // Remove global reference
        if (window.MojiWebChat === this) {
            delete window.MojiWebChat;
        }
    }

    // Static factory method
    static create(config) {
        return new MojiWebChatV2(config);
    }
}

// Export for global use
window.MojiWebChatV2 = MojiWebChatV2;