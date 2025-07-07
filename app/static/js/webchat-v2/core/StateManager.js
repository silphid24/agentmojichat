/**
 * State management module - React-like state management
 */
export class StateManager {
    constructor(initialState = {}) {
        this.state = { ...initialState };
        this.subscribers = new Set();
        this.middlewares = [];
    }

    getState() {
        return { ...this.state };
    }

    setState(updates) {
        const prevState = this.getState();
        const newState = typeof updates === 'function' 
            ? updates(prevState) 
            : { ...this.state, ...updates };

        // Apply middlewares
        let finalState = newState;
        for (const middleware of this.middlewares) {
            finalState = middleware(finalState, prevState) || finalState;
        }

        this.state = finalState;
        this.notifySubscribers(prevState);
    }

    subscribe(callback) {
        this.subscribers.add(callback);
        return () => this.subscribers.delete(callback);
    }

    notifySubscribers(prevState) {
        this.subscribers.forEach(callback => {
            try {
                callback(this.state, prevState);
            } catch (error) {
                console.error('Error in state subscriber:', error);
            }
        });
    }

    use(middleware) {
        this.middlewares.push(middleware);
    }

    // Computed values with memoization
    createSelector(dependencies, compute) {
        let lastDeps = [];
        let lastResult;

        return () => {
            const currentDeps = dependencies.map(dep => 
                typeof dep === 'function' ? dep(this.state) : this.state[dep]
            );

            const depsChanged = currentDeps.some((dep, i) => dep !== lastDeps[i]);

            if (depsChanged || lastResult === undefined) {
                lastDeps = currentDeps;
                lastResult = compute(...currentDeps);
            }

            return lastResult;
        };
    }
}

// Default state shape
export const createInitialState = () => ({
    // UI State
    isOpen: false,
    isMinimized: false,
    showWelcomeMessage: true,
    isLoading: false,
    
    // Chat State
    messages: [],
    inputValue: '',
    
    // Connection State
    isConnected: false,
    connectionError: null,
    reconnectAttempts: 0,
    
    // Settings
    currentProvider: 'openai',
    currentModel: 'gpt-3.5-turbo',
    temperature: 0.7,
    useRag: true,
    
    // User Info
    userId: null,
    userName: 'Guest User'
});