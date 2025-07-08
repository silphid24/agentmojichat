/**
 * WebSocket connection management module
 */
export class WebSocketManager {
    constructor(config, eventEmitter) {
        this.config = config;
        this.events = eventEmitter;
        this.ws = null;
        this.messageQueue = [];
        this.reconnectTimer = null;
        this.pingInterval = null;
        this.reconnectAttempts = 0;
    }

    connect() {
        if (this.ws?.readyState === WebSocket.OPEN) {
            console.log('WebSocket already connected');
            return Promise.resolve();
        }

        console.log(`Attempting to connect to: ${this.config.wsUrl}`);
        
        return new Promise((resolve, reject) => {
            try {
                this.ws = new WebSocket(this.config.wsUrl);
                
                // Connection timeout
                const connectTimeout = setTimeout(() => {
                    console.error('WebSocket connection timeout');
                    this.ws.close();
                    reject(new Error('Connection timeout'));
                }, 10000); // 10 seconds timeout
                
                this.ws.onopen = () => {
                    clearTimeout(connectTimeout);
                    console.log('WebSocket connection opened successfully');
                    this.onOpen();
                    resolve();
                };
                
                this.ws.onmessage = (event) => {
                    if (this.config.debug) {
                        console.log('WebSocket message received:', event.data);
                    }
                    this.onMessage(event);
                };
                
                this.ws.onclose = (event) => {
                    clearTimeout(connectTimeout);
                    console.log(`WebSocket connection closed: code=${event.code}, reason=${event.reason}`);
                    this.onClose();
                };
                
                this.ws.onerror = (error) => {
                    clearTimeout(connectTimeout);
                    console.error('WebSocket error:', error);
                    this.onError(error);
                    reject(error);
                };
            } catch (error) {
                console.error('Failed to create WebSocket:', error);
                this.events.emit('connection:error', { error: error.message });
                reject(error);
            }
        });
    }

    disconnect() {
        this.clearTimers();
        
        if (this.ws) {
            this.ws.close(1000, 'User disconnect');
            this.ws = null;
        }
    }

    onOpen() {
        console.log('WebSocket connection established');
        this.events.emit('connection:open');
        this.reconnectAttempts = 0;  // 성공적으로 연결되면 재연결 시도 횟수 리셋
        this.startPingInterval();
        
        // Send authentication
        console.log('Sending authentication message');
        this.send({
            type: 'auth',
            user_id: this.config.userId,
            user_name: this.config.userName
        });
        
        // Send queued messages
        this.flushMessageQueue();
    }

    onMessage(event) {
        try {
            const data = JSON.parse(event.data);
            
            // Handle different message types
            switch (data.type) {
                case 'pong':
                    // Keep-alive response
                    break;
                    
                case 'text':
                    this.events.emit('message:received', {
                        text: data.text,
                        timestamp: data.timestamp,
                        sender: 'bot'
                    });
                    break;
                    
                case 'system':
                    this.events.emit('message:system', {
                        text: data.text,
                        timestamp: data.timestamp
                    });
                    break;
                    
                case 'typing':
                    this.events.emit('typing:update', data.show);
                    break;
                    
                case 'error':
                    this.events.emit('message:error', data);
                    break;
                    
                default:
                    console.warn('Unknown message type:', data.type);
            }
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    }

    onClose() {
        this.clearTimers();
        this.events.emit('connection:close');
        
        // Attempt reconnection
        if (this.config.autoReconnect !== false) {
            this.scheduleReconnect();
        }
    }

    onError(error) {
        console.error('WebSocket error:', error);
        this.events.emit('connection:error', { error: 'Connection failed' });
    }

    send(data) {
        if (this.ws?.readyState === WebSocket.OPEN) {
            const jsonData = JSON.stringify(data);
            if (this.config.debug && data.type !== 'ping' && data.type !== 'pong') {
                console.log('Sending WebSocket message:', data);
            }
            this.ws.send(jsonData);
            return true;
        } else {
            // Queue message if not connected
            console.log('WebSocket not connected, queuing message:', data.type);
            this.messageQueue.push(data);
            return false;
        }
    }

    sendMessage(text, options = {}) {
        const message = {
            type: 'text',
            text: text,
            timestamp: new Date().toISOString(),
            ...options
        };
        
        console.log('Sending user message:', text);
        return this.send(message);
    }

    flushMessageQueue() {
        while (this.messageQueue.length > 0 && this.ws?.readyState === WebSocket.OPEN) {
            const message = this.messageQueue.shift();
            this.ws.send(JSON.stringify(message));
        }
    }

    scheduleReconnect() {
        // 최대 재연결 시도 횟수 체크
        if (this.reconnectAttempts >= (this.config.maxReconnectAttempts || 5)) {
            this.events.emit('connection:failed');
            console.log('Max reconnection attempts reached');
            return;
        }

        const delay = Math.min(
            (this.config.reconnectInterval || 5000) * Math.pow(2, this.reconnectAttempts),
            30000 // Max 30 seconds
        );
        
        console.log(`Scheduling reconnect attempt ${this.reconnectAttempts + 1} in ${delay}ms`);
        
        this.reconnectTimer = setTimeout(() => {
            this.reconnectAttempts++;
            this.events.emit('connection:reconnecting', { 
                attempt: this.reconnectAttempts,
                maxAttempts: this.config.maxReconnectAttempts || 5
            });
            
            this.connect().catch((error) => {
                console.error('Reconnection failed:', error);
                // scheduleReconnect will be called again from onClose if needed
            });
        }, delay);
    }

    startPingInterval() {
        // Send ping every 30 seconds to keep connection alive
        this.pingInterval = setInterval(() => {
            if (this.ws?.readyState === WebSocket.OPEN) {
                this.send({ type: 'ping' });
            }
        }, 30000);
    }

    clearTimers() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }

    getReadyState() {
        return this.ws?.readyState || WebSocket.CLOSED;
    }

    isConnected() {
        return this.ws?.readyState === WebSocket.OPEN;
    }
}