import logging
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from config.settings import OLLAMA_MODEL, OLLAMA_BASE_URL, TEMPERATURE

logger = logging.getLogger(__name__)

class LLMManager:
    def __init__(self, streaming: bool = True):
        self.streaming = streaming
        self.llm = self._init_llm()
    
    def _init_llm(self) -> Ollama:
        """Initialize Ollama LLM"""
        try:
            # Don't use callback manager here - let Chainlit handle it
            llm = Ollama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=TEMPERATURE
            )
            
            logger.info(f"LLM initialized: {OLLAMA_MODEL}")
            return llm
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise
    
    def get_llm(self) -> Ollama:
        """Get LLM instance"""
        return self.llm
    
    def test_connection(self) -> bool:
        """Test if Ollama is running"""
        try:
            response = self.llm("Hello")
            logger.info("Ollama connection successful")
            return True
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            return False