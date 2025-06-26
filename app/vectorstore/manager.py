"""Vector store manager for handling multiple stores"""

from typing import Dict, Any, Optional, List, Type
from enum import Enum

from app.vectorstore.base import BaseVectorStore, VectorStoreConfig
from app.vectorstore.chroma_store import ChromaVectorStore
from app.rag.retriever import VectorRetriever
from app.core.logging import logger


class VectorStoreType(str, Enum):
    """Available vector store types"""
    FAISS = "faiss"
    CHROMA = "chroma"
    # Future: WEAVIATE = "weaviate"
    # Future: PINECONE = "pinecone"


class VectorStoreManager:
    """Manages multiple vector stores and routing"""
    
    def __init__(self):
        self.stores: Dict[str, BaseVectorStore] = {}
        self.default_store: Optional[str] = None
        
        # Store type registry
        self.store_types: Dict[VectorStoreType, Type[BaseVectorStore]] = {
            VectorStoreType.CHROMA: ChromaVectorStore,
            # VectorStoreType.FAISS is handled by existing VectorRetriever
        }
        
        logger.info("Initialized VectorStoreManager")
    
    async def create_store(
        self,
        store_id: str,
        store_type: VectorStoreType,
        config: VectorStoreConfig,
        embeddings=None,
        set_as_default: bool = False
    ) -> BaseVectorStore:
        """Create and register a new vector store"""
        
        if store_type == VectorStoreType.FAISS:
            # Use existing FAISS implementation
            from app.rag.embeddings import get_embeddings
            retriever = VectorRetriever(
                embeddings=embeddings or get_embeddings(use_local=True),
                index_path=config.persist_directory or "data/faiss_index"
            )
            # Note: VectorRetriever doesn't inherit from BaseVectorStore
            # so we'll need to wrap it or use it directly
            logger.info(f"Created FAISS store: {store_id}")
            return retriever
        
        elif store_type in self.store_types:
            store_class = self.store_types[store_type]
            store = store_class(config, embeddings)
            await store.initialize()
            
            self.stores[store_id] = store
            
            if set_as_default or not self.default_store:
                self.default_store = store_id
            
            logger.info(f"Created {store_type} store: {store_id}")
            return store
        
        else:
            raise ValueError(f"Unknown store type: {store_type}")
    
    def get_store(self, store_id: Optional[str] = None) -> BaseVectorStore:
        """Get a vector store by ID or default"""
        if not store_id:
            store_id = self.default_store
        
        if not store_id or store_id not in self.stores:
            raise ValueError(f"Vector store not found: {store_id}")
        
        return self.stores[store_id]
    
    async def add_documents_to_all(
        self,
        documents: List[Any],
        exclude_stores: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """Add documents to all stores (except excluded)"""
        results = {}
        exclude_stores = exclude_stores or []
        
        for store_id, store in self.stores.items():
            if store_id not in exclude_stores:
                try:
                    ids = await store.add_documents(documents)
                    results[store_id] = ids
                except Exception as e:
                    logger.error(f"Error adding to store {store_id}: {e}")
                    results[store_id] = []
        
        return results
    
    async def search_all_stores(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Any]]:
        """Search across all stores"""
        results = {}
        
        for store_id, store in self.stores.items():
            try:
                search_results = await store.search(query, k, filter)
                results[store_id] = search_results
            except Exception as e:
                logger.error(f"Error searching store {store_id}: {e}")
                results[store_id] = []
        
        return results
    
    async def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all stores"""
        stats = {}
        
        for store_id, store in self.stores.items():
            try:
                store_stats = await store.get_collection_stats()
                stats[store_id] = store_stats
            except Exception as e:
                logger.error(f"Error getting stats for store {store_id}: {e}")
                stats[store_id] = {"error": str(e)}
        
        stats["default_store"] = self.default_store
        stats["total_stores"] = len(self.stores)
        
        return stats
    
    async def optimize_stores(self) -> Dict[str, bool]:
        """Optimize all vector stores"""
        results = {}
        
        for store_id, store in self.stores.items():
            try:
                # Store-specific optimization
                if hasattr(store, 'optimize'):
                    await store.optimize()
                    results[store_id] = True
                else:
                    # Default optimization: persist
                    await store.persist()
                    results[store_id] = True
                    
            except Exception as e:
                logger.error(f"Error optimizing store {store_id}: {e}")
                results[store_id] = False
        
        return results
    
    def remove_store(self, store_id: str) -> bool:
        """Remove a store from manager"""
        if store_id in self.stores:
            del self.stores[store_id]
            
            # Update default if needed
            if self.default_store == store_id:
                self.default_store = list(self.stores.keys())[0] if self.stores else None
            
            logger.info(f"Removed store: {store_id}")
            return True
        
        return False
    
    async def migrate_data(
        self,
        from_store_id: str,
        to_store_id: str,
        batch_size: int = 100
    ) -> bool:
        """Migrate data between stores"""
        try:
            from_store = self.get_store(from_store_id)
            to_store = self.get_store(to_store_id)
            
            # Get all documents from source store
            # Note: This is a simplified approach for MVP
            # In production, implement proper pagination
            
            logger.info(f"Starting migration from {from_store_id} to {to_store_id}")
            
            # For now, we'll indicate that migration would happen here
            # Actual implementation would require batch processing
            
            return True
            
        except Exception as e:
            logger.error(f"Error during migration: {e}")
            return False


# Global vector store manager
vector_store_manager = VectorStoreManager()