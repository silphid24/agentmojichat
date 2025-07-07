/**
 * Configuration and model management
 */
export class ConfigManager {
    constructor(eventEmitter) {
        this.events = eventEmitter;
        
        // Provider configurations
        this.providers = {
            'openai': { 
                name: 'OpenAI', 
                icon: '🤖',
                models: [
                    { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo' },
                    { id: 'gpt-4', name: 'GPT-4' },
                    { id: 'gpt-4-turbo', name: 'GPT-4 Turbo' }
                ],
                defaultModel: 'gpt-3.5-turbo'
            },
            'anthropic': { 
                name: 'Anthropic', 
                icon: '🧠',
                models: [
                    { id: 'claude-3-sonnet-20240229', name: 'Claude 3 Sonnet' },
                    { id: 'claude-3-haiku-20240307', name: 'Claude 3 Haiku' }
                ],
                defaultModel: 'claude-3-sonnet-20240229'
            },
            'deepseek': { 
                name: 'DeepSeek', 
                icon: '🚀',
                models: [
                    { id: 'deepseek-r1', name: 'DeepSeek R1' },
                    { id: 'deepseek-chat', name: 'DeepSeek Chat' }
                ],
                defaultModel: 'deepseek-r1'
            },
            'custom': { 
                name: 'Workstation LLM', 
                icon: '🖥️',
                models: [
                    { id: 'llama3.2:latest', name: 'Llama 3.2' },
                    { id: 'custom-model', name: 'Custom Model' }
                ],
                defaultModel: 'llama3.2:latest'
            },
            'deepseek-local': { 
                name: 'DeepSeek (Local)', 
                icon: '💻',
                models: [
                    { id: 'deepseek-r1', name: 'DeepSeek R1 (Local)' }
                ],
                defaultModel: 'deepseek-r1'
            },
            'exaone-local': { 
                name: 'EXAONE (Local)', 
                icon: '🔮',
                models: [
                    { id: 'exaone-3.0', name: 'EXAONE 3.0 (Local)' }
                ],
                defaultModel: 'exaone-3.0'
            }
        };
        
        // Temperature presets
        this.temperaturePresets = {
            0.0: { name: '매우 정확', description: '일관되고 예측 가능한 응답' },
            0.3: { name: '정확', description: '사실 기반의 정확한 응답' },
            0.7: { name: '균형적', description: '정확성과 창의성의 균형' },
            1.0: { name: '창의적', description: '다양하고 창의적인 응답' },
            1.5: { name: '매우 창의적', description: '예측 불가능하고 독특한 응답' },
            2.0: { name: '실험적', description: '매우 창의적이지만 불안정할 수 있음' }
        };
    }

    async loadProviderModels(provider) {
        try {
            const response = await fetch(`/api/v1/llm/models/${provider}/public`);
            
            if (response.ok) {
                const data = await response.json();
                this.providers[provider].models = data.models;
                this.events.emit('provider:modelsLoaded', { provider, models: data.models });
                return data.models;
            }
        } catch (error) {
            console.error(`Failed to load models for ${provider}:`, error);
        }
        
        // Return default models on error
        return this.providers[provider].models;
    }

    getProvider(providerId) {
        return this.providers[providerId];
    }

    getAllProviders() {
        return Object.entries(this.providers).map(([id, provider]) => ({
            id,
            ...provider
        }));
    }

    getModelsForProvider(providerId) {
        return this.providers[providerId]?.models || [];
    }

    getDefaultModelForProvider(providerId) {
        return this.providers[providerId]?.defaultModel;
    }

    getTemperatureInfo(temperature) {
        // Find closest preset
        const presets = Object.keys(this.temperaturePresets)
            .map(Number)
            .sort((a, b) => Math.abs(a - temperature) - Math.abs(b - temperature));
        
        const closest = presets[0];
        return this.temperaturePresets[closest];
    }

    validateConfig(config) {
        const errors = [];
        
        // Validate WebSocket URL
        if (!config.wsUrl) {
            errors.push('WebSocket URL is required');
        } else {
            try {
                new URL(config.wsUrl);
            } catch {
                errors.push('Invalid WebSocket URL');
            }
        }
        
        // Validate reconnect settings
        if (config.maxReconnectAttempts && config.maxReconnectAttempts < 1) {
            errors.push('maxReconnectAttempts must be at least 1');
        }
        
        if (config.reconnectInterval && config.reconnectInterval < 1000) {
            errors.push('reconnectInterval must be at least 1000ms');
        }
        
        return errors;
    }

    mergeWithDefaults(userConfig) {
        const defaults = {
            wsUrl: `ws://${window.location.host}/api/v1/adapters/webchat/ws`,
            userId: this.generateUserId(),
            userName: 'Guest User',
            reconnectInterval: 5000,
            maxReconnectAttempts: 5,
            autoReconnect: true,
            debug: false,
            locale: 'ko-KR',
            theme: 'light',
            position: 'bottom-right',
            animation: true,
            sound: false
        };
        
        return { ...defaults, ...userConfig };
    }

    generateUserId() {
        return `user_${Math.random().toString(36).substr(2, 9)}`;
    }

    // Settings persistence
    saveSettings(settings) {
        try {
            localStorage.setItem('moji_chat_settings', JSON.stringify(settings));
            this.events.emit('settings:saved', settings);
        } catch (error) {
            console.error('Failed to save settings:', error);
        }
    }

    loadSettings() {
        try {
            const saved = localStorage.getItem('moji_chat_settings');
            if (saved) {
                return JSON.parse(saved);
            }
        } catch (error) {
            console.error('Failed to load settings:', error);
        }
        return null;
    }

    clearSettings() {
        try {
            localStorage.removeItem('moji_chat_settings');
            this.events.emit('settings:cleared');
        } catch (error) {
            console.error('Failed to clear settings:', error);
        }
    }
}