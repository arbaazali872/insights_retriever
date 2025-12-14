import chainlit as cl
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.document_loader import DocumentLoader
from core.vectorstore import VectorStoreManager
from core.rag_chain import RAGChain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
doc_loader = DocumentLoader()
vectorstore = VectorStoreManager()
rag_chain = None

@cl.on_chat_start
async def start():
    """Initialize chat session"""
    global rag_chain
    
    await cl.Message(
        content="👋 **Welcome to InsightVault!**\n\n"
                "I'm your local AI research assistant. I can help you analyze documents, articles, and extract insights.\n\n"
                "**How to use:**\n"
                "1. Upload documents or provide URLs using `/add`\n"
                "2. Ask me questions about your documents\n"
                "3. I'll search and provide answers with sources\n\n"
                "**Commands:**\n"
                "- `/add <url>` - Add article from URL\n"
                "- `/stats` - View knowledge base stats\n"
                "- `/clear` - Clear conversation history\n"
                "- `/reset` - Clear all documents\n\n"
                "Try: `/add https://example.com/article` or upload a PDF!"
    ).send()
    
    # Initialize RAG chain
    rag_chain = RAGChain(vectorstore)
    cl.user_session.set("rag_chain", rag_chain)

@cl.on_message
async def main(message: cl.Message):
    """Handle user messages"""
    global rag_chain
    
    if rag_chain is None:
        rag_chain = cl.user_session.get("rag_chain")
    
    user_input = message.content.strip()
    
    # Handle commands
    if user_input.startswith("/add"):
        await handle_add_url(user_input)
        return
    
    elif user_input == "/stats":
        await handle_stats()
        return
    
    elif user_input == "/clear":
        await handle_clear_memory()
        return
    
    elif user_input == "/reset":
        await handle_reset()
        return
    
    # Handle file uploads
    if message.elements:
        await handle_file_upload(message.elements)
        return
    
    # Handle normal query with AsyncLangchainCallbackHandler
    try:
        stats = vectorstore.get_stats()
        if stats.get("total_documents", 0) == 0:
            await cl.Message(
                content="⚠️ No documents loaded yet. Please add documents first using `/add <url>` or upload files."
            ).send()
            return
        
        # CRITICAL FIX: Add callback handler
        cb = cl.AsyncLangchainCallbackHandler()
        
        # Query RAG chain with callback
        response = await rag_chain.query(user_input, callbacks=[cb])
        
        if not response or not response.get('answer'):
            await cl.Message(content="❌ No answer generated").send()
            return
        
        # Format response
        answer_text = f"{response['answer']}\n\n"
        
        # Create text elements for sources
        text_elements = []
        if response['sources']:
            answer_text += "**Sources:**\n"
            for i, source in enumerate(response['sources'][:3], 1):
                src_info = source['metadata']
                source_name = f"source_{i}"
                
                # Create text element
                text_elements.append(
                    cl.Text(
                        content=source['content'],
                        name=source_name,
                        display="side"
                    )
                )
                
                answer_text += f"{i}. **{src_info.get('title', src_info.get('source', 'Unknown'))}**\n"
            
            source_names = [el.name for el in text_elements]
            if source_names:
                answer_text += f"\nSources: {', '.join(source_names)}"
        
        await cl.Message(content=answer_text, elements=text_elements).send()
        logger.info("Answer sent to UI")
        
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        await cl.Message(content=f"❌ Error: {str(e)}").send()

async def handle_add_url(command: str):
    """Add document from URL"""
    global rag_chain
    try:
        url = command.replace("/add", "").strip()
        if not url:
            await cl.Message(content="❌ Please provide a URL: `/add <url>`").send()
            return
        
        msg = cl.Message(content=f"📥 Loading article from: {url}")
        await msg.send()
        
        # Load and process
        chunks = doc_loader.load_url(url)
        vectorstore.add_documents(chunks)
        
        # Reinitialize RAG chain with new data
        rag_chain = RAGChain(vectorstore)
        cl.user_session.set("rag_chain", rag_chain)
        
        msg.content = f"✅ Successfully added article!\n" \
                      f"- URL: {url}\n" \
                      f"- Chunks: {len(chunks)}"
        await msg.update()
        
    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()

async def handle_file_upload(elements):
    """Handle file uploads"""
    global rag_chain
    try:
        for element in elements:
            msg = cl.Message(content=f"📥 Processing: {element.name}")
            await msg.send()
            
            if element.name.endswith('.pdf'):
                chunks = doc_loader.load_pdf(element.path)
            elif element.name.endswith('.txt'):
                with open(element.path, 'r') as f:
                    text = f.read()
                chunks = doc_loader.load_text(text, element.name)
            else:
                await cl.Message(content=f"❌ Unsupported file type: {element.name}").send()
                continue
            
            vectorstore.add_documents(chunks)
            
            # Reinitialize RAG chain
            rag_chain = RAGChain(vectorstore)
            cl.user_session.set("rag_chain", rag_chain)
            
            msg.content = f"✅ Successfully added: {element.name}\n" \
                          f"- Chunks: {len(chunks)}"
            await msg.update()
            
    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()

async def handle_stats():
    """Show knowledge base statistics"""
    stats = vectorstore.get_stats()
    await cl.Message(
        content=f"📊 **Knowledge Base Stats:**\n\n"
                f"- Total documents: {stats.get('total_documents', 0)}\n"
                f"- Collection: {stats.get('collection_name', 'N/A')}\n"
                f"- Memory messages: {rag_chain.get_memory_size()}"
    ).send()

async def handle_clear_memory():
    """Clear conversation memory"""
    rag_chain.clear_memory()
    await cl.Message(content="🗑️ Conversation history cleared!").send()

async def handle_reset():
    """Reset entire knowledge base"""
    global rag_chain
    vectorstore.clear()
    rag_chain = RAGChain(vectorstore)
    cl.user_session.set("rag_chain", rag_chain)
    await cl.Message(content="🗑️ All documents and history cleared!").send()