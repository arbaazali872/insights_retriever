import logging
from typing import List, Optional
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from config.settings import CHROMA_DIR, EMBEDDING_MODEL, EMBEDDING_DEVICE, TOP_K_RESULTS

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.embeddings = self._load_embeddings()
        self.vectorstore = self._init_vectorstore()
    
    def _load_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialize embedding model"""
        logger.info(f"Loading embeddings: {EMBEDDING_MODEL}")
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': EMBEDDING_DEVICE},
            encode_kwargs={'normalize_embeddings': False}
        )
    
    def _init_vectorstore(self) -> Chroma:
        """Initialize or load ChromaDB"""
        try:
            vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(CHROMA_DIR)
            )
            logger.info(f"Vector store initialized: {self.collection_name}")
            return vectorstore
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to vector store"""
        try:
            self.vectorstore.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = TOP_K_RESULTS) -> List[Document]:
        """Search for similar documents"""
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents")
            return results
        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise
    
    def get_retriever(self, k: int = TOP_K_RESULTS):
        """Get retriever for RAG chain"""
        return self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )
    
    def get_stats(self) -> dict:
        """Get vector store statistics"""
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}
    
    def clear(self) -> None:
        """Clear all documents from vector store"""
        try:
            self.vectorstore.delete_collection()
            self.vectorstore = self._init_vectorstore()
            logger.info("Vector store cleared")
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            raise