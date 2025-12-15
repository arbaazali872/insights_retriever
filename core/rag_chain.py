import logging
from typing import Dict, List
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from core.vectorstore import VectorStoreManager
from core.llm_manager import LLMManager
from config.prompts import CONVERSATION_PROMPT
from config.settings import MAX_CONVERSATION_HISTORY

logger = logging.getLogger(__name__)

class RAGChain:
    def __init__(self, vectorstore_manager: VectorStoreManager):
        self.vectorstore = vectorstore_manager
        self.llm_manager = LLMManager(streaming=True)
        self.memory = self._init_memory()
        self.chain = self._create_chain()
    
    def _init_memory(self) -> ConversationBufferMemory:
        """Initialize conversation memory"""
        return ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"  # Required for return_source_documents=True
        )
    
    def _create_chain(self) -> ConversationalRetrievalChain:
        """Create RAG chain with memory"""
        try:
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm_manager.get_llm(),
                retriever=self.vectorstore.get_retriever(),
                memory=self.memory,
                return_source_documents=True,
                verbose=False
            )
            
            logger.info("RAG chain created successfully")
            return chain
            
        except Exception as e:
            logger.error(f"Error creating RAG chain: {e}")
            raise
    
    async def query(self, question: str, callbacks=None) -> Dict:
        """Query the RAG system - ASYNC"""
        try:
            logger.info(f"Processing query: {question}")
            
            # Use acall instead of invoke for async operation
            response = await self.chain.acall(
                {"question": question},
                callbacks=callbacks
            )
            
            logger.info(f"Got response with {len(response.get('source_documents', []))} sources")
            
            # Format sources
            sources = []
            for doc in response.get("source_documents", []):
                sources.append({
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                })
            
            result = {
                "answer": response["answer"],
                "sources": sources
            }
            
            logger.info(f"Answer length: {len(result['answer'])} chars")
            return result
            
        except Exception as e:
            logger.error(f"Error querying RAG chain: {e}", exc_info=True)
            raise
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        logger.info("Conversation memory cleared")
    
    def get_memory_size(self) -> int:
        """Get number of messages in memory"""
        return len(self.memory.chat_memory.messages)