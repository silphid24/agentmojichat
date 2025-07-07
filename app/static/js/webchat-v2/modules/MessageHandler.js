/**
 * Message handling logic
 */
export class MessageHandler {
    constructor(stateManager, wsManager, eventEmitter) {
        this.state = stateManager;
        this.ws = wsManager;
        this.events = eventEmitter;
        
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Handle incoming messages
        this.events.on('message:received', (data) => {
            this.addMessage({
                id: `bot-${Date.now()}`,
                text: data.text,
                sender: 'bot',
                timestamp: data.timestamp || new Date().toISOString()
            });
            
            this.state.setState({ isLoading: false });
            this.events.emit('typing:update', false);
        });

        this.events.on('message:system', (data) => {
            this.addMessage({
                id: `system-${Date.now()}`,
                text: data.text,
                sender: 'system',
                timestamp: data.timestamp || new Date().toISOString()
            });
        });

        this.events.on('message:error', (data) => {
            this.addMessage({
                id: `error-${Date.now()}`,
                text: data.message || '오류가 발생했습니다.',
                sender: 'system',
                timestamp: new Date().toISOString()
            });
            
            this.state.setState({ isLoading: false });
            this.events.emit('typing:update', false);
        });
    }

    sendMessage(text) {
        if (!text?.trim()) return false;

        const currentState = this.state.getState();
        
        // Add user message to state
        const userMessage = {
            id: `user-${Date.now()}`,
            text: text.trim(),
            sender: 'user',
            timestamp: new Date().toISOString()
        };

        this.addMessage(userMessage);
        this.state.setState({
            inputValue: '',
            isLoading: true
        });

        // Show typing indicator
        this.events.emit('typing:update', true);

        // Send via WebSocket
        const sent = this.ws.sendMessage(text, {
            provider: currentState.currentProvider,
            model: currentState.currentModel,
            useRag: currentState.useRag,
            temperature: currentState.temperature
        });

        if (!sent) {
            this.addMessage({
                id: `system-${Date.now()}`,
                text: '연결이 끊겼습니다. 재연결 중...',
                sender: 'system',
                timestamp: new Date().toISOString()
            });
        }

        return true;
    }

    addMessage(message) {
        const currentMessages = this.state.getState().messages;
        this.state.setState({
            messages: [...currentMessages, message]
        });
    }

    clearMessages() {
        this.state.setState({ messages: [] });
    }

    // Special commands handling
    handleCommand(text) {
        if (!text.startsWith('/')) return false;

        const [command, ...args] = text.slice(1).split(' ');
        const arg = args.join(' ');

        switch (command.toLowerCase()) {
            case 'clear':
                this.clearMessages();
                this.addMessage({
                    id: `system-${Date.now()}`,
                    text: '대화 내용이 초기화되었습니다.',
                    sender: 'system',
                    timestamp: new Date().toISOString()
                });
                return true;

            case 'help':
                this.addMessage({
                    id: `system-${Date.now()}`,
                    text: '사용 가능한 명령어:\n/clear - 대화 초기화\n/rag <질문> - RAG 검색\n/model - 현재 모델 정보',
                    sender: 'system',
                    timestamp: new Date().toISOString()
                });
                return true;

            case 'rag':
                if (arg) {
                    this.sendRagQuery(arg);
                } else {
                    this.addMessage({
                        id: `system-${Date.now()}`,
                        text: '사용법: /rag <질문>',
                        sender: 'system',
                        timestamp: new Date().toISOString()
                    });
                }
                return true;

            case 'model':
                const state = this.state.getState();
                this.addMessage({
                    id: `system-${Date.now()}`,
                    text: `현재 모델: ${state.currentProvider} - ${state.currentModel}\n온도: ${state.temperature}\nRAG: ${state.useRag ? '활성' : '비활성'}`,
                    sender: 'system',
                    timestamp: new Date().toISOString()
                });
                return true;

            default:
                return false;
        }
    }

    sendRagQuery(query) {
        this.state.setState({ isLoading: true });
        this.events.emit('typing:update', true);

        this.ws.send({
            type: 'rag_query',
            query: query,
            timestamp: new Date().toISOString()
        });
    }

    // Message validation
    validateMessage(text) {
        if (!text || typeof text !== 'string') return false;
        
        const trimmed = text.trim();
        if (trimmed.length === 0) return false;
        if (trimmed.length > 4000) return false; // Max length
        
        return true;
    }
}