from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import chromadb
from pathlib import Path
from uuid import uuid4
import logging
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import re
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(filename)s: %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    # Base directory for document storage
    DATA_PATH = Path("source/documents")

    def __init__(self):
        # Configure text splitting with overlap for better context preservation
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,  # Size of each text chunk
            chunk_overlap=200,  # Overlap between chunks to maintain context
            separators=["\n\n", "\n", ". ", " ", ""]  # Priority order for splitting
        )
        
        # Initialize embedding model for semantic search
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"  # Efficient, general-purpose embeddings
        )
        
        # Setup ChromaDB for vector storage
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection("documents")
        logger.info("DocumentProcessor initialized")

    def load_documents(self):
        """Load PDFs and HTML files from the specified data directory"""
        documents = []
        pdf_count = html_count = 0

        if not self.DATA_PATH.exists():
            logger.error(f"Directory does not exist: {self.DATA_PATH}")
            return documents

        # Process each file type
        for file_path in self.DATA_PATH.glob("*"):
            try:
                if file_path.suffix == '.pdf':
                    docs = self.process_pdf(str(file_path))
                elif file_path.suffix == '.html':
                    docs = self.process_html(str(file_path))
                else:
                    continue

                if docs:
                    documents.extend(docs)
                    if file_path.suffix == '.pdf':
                        pdf_count += 1
                    elif file_path.suffix == '.html':
                        html_count += 1
                    logger.info(f"✓ Loaded {file_path.suffix[1:].upper()}: {file_path.name}")
                    
            except Exception as e:
                logger.error(f"✗ Failed to load {file_path.name}: {e}")

        logger.info(f"Summary: Loaded {pdf_count} PDFs and {html_count} HTMLs")
        return documents

    def process_documents(self):
        """Process all documents and store in ChromaDB"""
        # Start document loading phase
        logger.info("=== Starting Document Loading ===")
        documents = self.load_documents()
        
        if documents:
            # Begin processing phase if documents were loaded
            logger.info("=== Starting Document Processing ===")
            processed_count = 0  # Track number of processed documents
            total_chunks = 0    # Track total number of chunks created
            
            # Process each document
            for doc in documents:
                try:
                    # Extract text content, handling different document formats
                    text_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    
                    # Split document into smaller chunks for better processing
                    chunks = self.text_splitter.split_text(text_content)
                    total_chunks += len(chunks)
                    
                    # Create embeddings for semantic search
                    embeddings = self.embedding_model.embed_documents(chunks)
                    
                    # Generate unique IDs for each chunk
                    ids = [str(uuid4()) for _ in chunks]
                    
                    # Store chunks and embeddings in ChromaDB
                    self.collection.add(
                        embeddings=embeddings,  # Vector embeddings for semantic search
                        documents=chunks,       # Text chunks for retrieval
                        ids=ids                 # Unique identifiers for each chunk
                    )
                    processed_count += 1
                    logger.info(f"✓ Processed document {processed_count}/{len(documents)}")
                    
                except Exception as e:
                    logger.error(f"✗ Failed to process document {processed_count + 1}: {e}")

            # Log processing summary
            logger.info("=== Document Processing Complete ===")
            logger.info(f"Summary: Processed {processed_count}/{len(documents)} documents into {total_chunks} chunks")
        else:
            logger.warning("No documents were loaded, skipping processing")

    def process_pdf(self, pdf_path):
        """Process a PDF file and return its content"""
        try:
            reader = PdfReader(pdf_path)
            text_content = ""
            
            for page in reader.pages:
                text_content += page.extract_text() + "\n\n"
            
            return [Document(page_content=text_content, metadata={"source": pdf_path})]
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return None

    def process_html(self, file_path: str) -> List[Document]:
        """Process HTML file and extract meaningful content"""
        try:
            # Read and parse HTML
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove non-content elements to improve quality
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract and clean text
            text = soup.get_text(separator='\n', strip=True)
            
            # Normalize whitespace while preserving structure
            text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
            text = re.sub(r'[ \t]+', ' ', text)      # Clean up horizontal whitespace
            
            return [Document(
                page_content=text,
                metadata={"source": file_path}
            )]
        except Exception as e:
            logger.error(f"Error processing HTML {file_path}: {e}")
            return None

    def embed_documents(self, texts):
        """Embed documents using the HuggingFace model"""
        return self.embedding_model.embed_documents(texts)