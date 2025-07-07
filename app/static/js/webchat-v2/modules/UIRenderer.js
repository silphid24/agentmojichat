/**
 * UI Rendering module - handles all DOM manipulation
 */
export class UIRenderer {
    constructor(elements, eventEmitter) {
        this.elements = elements;
        this.events = eventEmitter;
        this.messageElements = new Map(); // Cache message elements
    }

    // Chat window visibility
    showChatWindow() {
        const { chatWindow, chatToggle } = this.elements;
        if (!chatWindow) return;

        chatWindow.style.display = 'block';
        chatWindow.style.visibility = 'visible';
        chatWindow.style.opacity = '1';
        chatWindow.className = 'absolute glass-effect rounded-3xl shadow-2xl overflow-hidden chat-window-enter';
        
        if (chatToggle) {
            chatToggle.style.display = 'none';
        }
    }

    hideChatWindow() {
        const { chatWindow, chatToggle } = this.elements;
        if (!chatWindow) return;

        chatWindow.className = 'absolute glass-effect rounded-3xl shadow-2xl overflow-hidden chat-window-exit';
        
        setTimeout(() => {
            if (chatWindow) {
                chatWindow.style.display = 'none';
                chatWindow.style.visibility = 'hidden';
                chatWindow.style.opacity = '0';
            }
            if (chatToggle) {
                chatToggle.style.display = 'flex';
            }
        }, 300);
    }

    // Welcome message
    showWelcomeMessage() {
        const { welcomeMessage } = this.elements;
        if (!welcomeMessage) return;

        welcomeMessage.style.display = 'block';
        welcomeMessage.className = 'mr-4 mb-2 p-3 bg-white rounded-lg shadow-md border welcome-message-enter';
    }

    hideWelcomeMessage() {
        const { welcomeMessage } = this.elements;
        if (!welcomeMessage) return;

        welcomeMessage.className = 'mr-4 mb-2 p-3 bg-white rounded-lg shadow-md border welcome-message-exit';
        setTimeout(() => {
            if (welcomeMessage) {
                welcomeMessage.style.display = 'none';
            }
        }, 300);
    }

    // Messages rendering with virtual scrolling for performance
    renderMessages(messages) {
        const { messagesContainer } = this.elements;
        if (!messagesContainer) return;

        // Use DocumentFragment for better performance
        const fragment = document.createDocumentFragment();
        
        // Clear existing messages
        messagesContainer.innerHTML = '';
        this.messageElements.clear();

        messages.forEach(message => {
            const element = this.createMessageElement(message);
            this.messageElements.set(message.id, element);
            fragment.appendChild(element);
        });

        messagesContainer.appendChild(fragment);
        this.scrollToBottom();
    }

    // Add single message (more efficient for real-time updates)
    addMessage(message) {
        const { messagesContainer } = this.elements;
        if (!messagesContainer) return;

        const element = this.createMessageElement(message);
        this.messageElements.set(message.id, element);
        messagesContainer.appendChild(element);
        this.scrollToBottom();
    }

    createMessageElement(message) {
        const messageDiv = document.createElement('div');
        
        if (message.sender === 'system') {
            messageDiv.className = 'flex justify-center message-fade-in';
            messageDiv.innerHTML = `
                <div class="max-w-[80%] rounded-full px-3 py-1.5 bg-yellow-100 text-yellow-800 border border-yellow-300">
                    <p class="text-xs font-medium text-center">${this.escapeHtml(message.text)}</p>
                </div>
            `;
        } else {
            const isUser = message.sender === 'user';
            messageDiv.className = `flex ${isUser ? 'justify-end' : 'justify-start'} message-fade-in`;
            messageDiv.innerHTML = `
                <div class="max-w-[80%] rounded-2xl px-4 py-2.5 ${
                    isUser
                        ? 'bg-blue-500 text-white rounded-br-lg'
                        : 'bg-white text-gray-800 rounded-bl-lg border border-gray-200/80'
                }">
                    <p class="text-sm leading-relaxed">${this.escapeHtml(message.text)}</p>
                    ${message.timestamp ? `
                        <p class="text-xs opacity-70 mt-1">${this.formatTime(message.timestamp)}</p>
                    ` : ''}
                </div>
            `;
        }

        return messageDiv;
    }

    // Typing indicator
    showTypingIndicator(show = true) {
        const { typingIndicator } = this.elements;
        if (!typingIndicator) return;

        typingIndicator.style.display = show ? 'block' : 'none';
        if (show) {
            this.scrollToBottom();
        }
    }

    // Update UI elements based on state
    updateChatIcon(isOpen) {
        const { chatIcon } = this.elements;
        if (!chatIcon) return;

        const iconHtml = isOpen ? 
            '<svg class="w-7 h-7 text-white" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>' :
            '<svg class="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 01-2.555-.337A5.972 5.972 0 015.41 20.97a5.969 5.969 0 01-.474-.065 4.48 4.48 0 00.978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25z" /></svg>';
        
        chatIcon.className = 'absolute flex items-center justify-center icon-rotate-enter';
        chatIcon.innerHTML = iconHtml;
    }

    updateSendButton(isLoading, hasInput) {
        const { sendBtn, messageInput } = this.elements;
        
        if (sendBtn) {
            sendBtn.disabled = isLoading || !hasInput;
        }
        
        if (messageInput) {
            messageInput.disabled = isLoading;
        }
    }

    updateConnectionStatus(status) {
        const { connectionStatus } = this.elements;
        if (!connectionStatus) return;

        const statusConfig = {
            connected: {
                display: 'none'
            },
            connecting: {
                display: 'block',
                text: '연결 중...',
                className: 'fixed top-4 right-4 px-4 py-2 rounded-lg bg-yellow-100 text-yellow-800 border border-yellow-300 text-sm font-medium'
            },
            reconnecting: {
                display: 'block',
                text: '재연결 중...',
                className: 'fixed top-4 right-4 px-4 py-2 rounded-lg bg-orange-100 text-orange-800 border border-orange-300 text-sm font-medium'
            },
            error: {
                display: 'block',
                text: '연결 실패',
                className: 'fixed top-4 right-4 px-4 py-2 rounded-lg bg-red-100 text-red-800 border border-red-300 text-sm font-medium'
            }
        };

        const config = statusConfig[status] || statusConfig.error;
        
        connectionStatus.style.display = config.display;
        if (config.text) {
            connectionStatus.textContent = config.text;
            connectionStatus.className = config.className;
        }
    }

    updateMinimizedState(isMinimized) {
        const { chatWindow, chatContent } = this.elements;
        if (!chatWindow || !chatContent) return;

        const isMobile = window.innerWidth <= 640;
        const height = isMinimized ? '72px' : (isMobile ? 'calc(100vh - 8rem)' : '600px');
        
        chatWindow.style.height = height;
        chatContent.style.display = isMinimized ? 'none' : 'flex';
        
        if (!isMinimized) {
            const header = chatWindow.querySelector('.glass-header');
            if (header) {
                const headerHeight = header.offsetHeight;
                chatContent.style.height = `calc(${height} - ${headerHeight}px)`;
            }
        }
    }

    // Utility methods
    scrollToBottom() {
        const { messagesContainer } = this.elements;
        if (!messagesContainer) return;

        requestAnimationFrame(() => {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        });
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    formatTime(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleTimeString('ko-KR', {
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    // Input handling
    clearInput() {
        const { messageInput } = this.elements;
        if (messageInput) {
            messageInput.value = '';
            messageInput.focus();
        }
    }

    setInputValue(value) {
        const { messageInput } = this.elements;
        if (messageInput) {
            messageInput.value = value;
        }
    }

    focusInput() {
        const { messageInput } = this.elements;
        if (messageInput) {
            messageInput.focus();
        }
    }
}