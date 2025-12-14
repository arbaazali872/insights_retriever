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
        self.llm_manager = LLMManager(streaming=False)
        self.memory = self._init_memory()
        self.chain = self._create_chain()
    
    def _init_memory(self) -> ConversationBufferMemory:
        """Initialize conversation memory"""
        return ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    def _create_chain(self) -> ConversationalRetrievalChain:
        """Create RAG chain with memory"""
        try:
            prompt = PromptTemplate(
                template=CONVERSATION_PROMPT,
                input_variables=["chat_history", "context", "question"]
            )
            
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm_manager.get_llm(),
                retriever=self.vectorstore.get_retriever(),
                memory=self.memory,
                return_source_documents=True,
                verbose=True
            )
            
            logger.info("RAG chain created successfully")
            return chain
            
        except Exception as e:
            logger.error(f"Error creating RAG chain: {e}")
            raise
    
    def query(self, question: str) -> Dict:
        """Query the RAG system"""
        try:
            response = self.chain({"question": question})
            
            # Format sources
            sources = []
            for doc in response.get("source_documents", []):
                sources.append({
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                })
            
            return {
                "answer": response["answer"],
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error querying RAG chain: {e}")
            raise
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        logger.info("Conversation memory cleared")
    
    def get_memory_size(self) -> int:
        """Get number of messages in memory"""
        return len(self.memory.chat_memory.messages)