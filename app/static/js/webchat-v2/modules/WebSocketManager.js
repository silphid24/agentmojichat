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
    }

    connect() {
        if (this.ws?.readyState === WebSocket.OPEN) {
            return Promise.resolve();
        }

        return new Promise((resolve, reject) => {
            try {
                this.ws = new WebSocket(this.config.wsUrl);
                
                this.ws.onopen = () => {
                    this.onOpen();
                    resolve();
                };
                
                this.ws.onmessage = (event) => this.onMessage(event);
                this.ws.onclose = () => this.onClose();
                this.ws.onerror = (error) => {
                    this.onError(error);
                    reject(error);
                };
            } catch (error) {
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
        this.events.emit('connection:open');
        this.startPingInterval();
        
        // Send authentication
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
            this.ws.send(JSON.stringify(data));
            return true;
        } else {
            // Queue message if not connected
            this.messageQueue.push(data);
            return false;
        }
    }

    sendMessage(text, options = {}) {
        return this.send({
            type: 'text',
            text: text,
            timestamp: new Date().toISOString(),
            ...options
        });
    }

    flushMessageQueue() {
        while (this.messageQueue.length > 0 && this.ws?.readyState === WebSocket.OPEN) {
            const message = this.messageQueue.shift();
            this.ws.send(JSON.stringify(message));
        }
    }

    scheduleReconnect() {
        const delay = Math.min(
            this.config.reconnectInterval * Math.pow(2, this.reconnectAttempts),
            30000 // Max 30 seconds
        );
        
        this.reconnectTimer = setTimeout(() => {
            this.reconnectAttempts++;
            this.events.emit('connection:reconnecting', { 
                attempt: this.reconnectAttempts,
                maxAttempts: this.config.maxReconnectAttempts 
            });
            
            this.connect().catch(() => {
                if (this.reconnectAttempts >= this.config.maxReconnectAttempts) {
                    this.events.emit('connection:failed');
                }
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