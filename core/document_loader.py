import validators
import logging
from typing import List, Dict
from pathlib import Path
from newspaper import Article
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, DOCS_DIR

logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def load_url(self, url: str) -> List[Document]:
        """Load and process article from URL"""
        try:
            if not validators.url(url):
                raise ValueError(f"Invalid URL: {url}")
            
            article = Article(url)
            article.download()
            article.parse()
            
            metadata = {
                "source": url,
                "title": article.title,
                "authors": ", ".join(article.authors),
                "publish_date": str(article.publish_date) if article.publish_date else "Unknown",
                "type": "article"
            }
            
            # Create document and split
            doc = Document(page_content=article.text, metadata=metadata)
            chunks = self.text_splitter.split_documents([doc])
            
            logger.info(f"Loaded URL: {url} ({len(chunks)} chunks)")
            return chunks
            
        except Exception as e:
            logger.error(f"Error loading URL {url}: {e}")
            raise
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load and process PDF file"""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            metadata = {
                "source": Path(file_path).name,
                "type": "pdf",
                "pages": len(reader.pages)
            }
            
            doc = Document(page_content=text, metadata=metadata)
            chunks = self.text_splitter.split_documents([doc])
            
            logger.info(f"Loaded PDF: {file_path} ({len(chunks)} chunks)")
            return chunks
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise
    
    def load_text(self, text: str, source_name: str = "text_input") -> List[Document]:
        """Load and process raw text"""
        try:
            metadata = {
                "source": source_name,
                "type": "text"
            }
            
            doc = Document(page_content=text, metadata=metadata)
            chunks = self.text_splitter.split_documents([doc])
            
            logger.info(f"Loaded text: {source_name} ({len(chunks)} chunks)")
            return chunks
            
        except Exception as e:
            logger.error(f"Error loading text: {e}")
            raise
    
    def load_multiple_urls(self, urls: List[str]) -> List[Document]:
        """Load multiple URLs"""
        all_chunks = []
        for url in urls:
            try:
                chunks = self.load_url(url)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"Skipping URL {url}: {e}")
                continue
        
        return all_chunks