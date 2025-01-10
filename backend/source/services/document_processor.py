# from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader, TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# # #from langchain_community.embeddings import HuggingFaceEmbeddings
# # import chromadb
# # import torch
# from transformers import AutoTokenizer, AutoModel
# import torch
# import chromadb

# class DocumentProcessor:
#     def __init__(self):
#         self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#         # If you want to use HuggingFaceEmbeddings, uncomment and initialize it properly
#         # self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
#         self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2") 
#         self.chroma_client = chromadb.Client()
#         self.collection = self.chroma_client.create_collection("documents")

#     def process_pdf(self, pdf_path):
#         try:
#             loader = PyPDFLoader(pdf_path)
#             documents = loader.load()
#             self._process_documents(documents)

#         except Exception as e:
#             print(f"Error processing PDF {pdf_path}: {e}")

#     def process_html(self, html_path):
#         try:
#             loader = UnstructuredHTMLLoader(html_path)
#             documents = loader.load()
#             self._process_documents(documents)
#         except Exception as e:
#             print(f"Error processing HTML {html_path}: {e}")

#     def process_text(self, text_path):
#         try:
#             loader = TextLoader(text_path)
#             documents = loader.load()
#             self._process_documents(documents)
#         except Exception as e:
#             print(f"Error processing text file {text_path}: {e}")

#     def process_document(self, file_path):
#         """Process a document based on its file type."""
#         try:
#             if file_path.endswith(".pdf"):
#                 self.process_pdf(file_path)
#             elif file_path.endswith(".html"):
#                 self.process_html(file_path)
#             elif file_path.endswith(".txt"):
#                 self.process_text(file_path)
#             else:
#                 raise ValueError(f"Unsupported file type: {file_path}")
#         except Exception as e:
#             print(f"Error processing file {file_path}: {e}")

#     def _process_documents(self, documents):
        
#         for doc in documents:
#             chunks = self.text_splitter.split_text(doc)
#             # embeddings = self.embedding_model.embed_documents(chunks)
#             embeddings = self.embed_documents(chunks)
#             self.collection.add(embeddings=embeddings, documents=chunks)

#     def embed_documents(self, texts):
#         inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#         return outputs.last_hidden_state.mean(dim=1).numpy()

    

from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
import torch
import chromadb
from pathlib import Path
from uuid import uuid4
from langchain_huggingface import HuggingFaceEmbeddings


class DocumentProcessor:
    DATA_PATH = Path("source/documents")  # Define the data path here
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)

        # Initialize tokenizer and model
        # try:
        #     self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        #     self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        # except Exception as e:
        #     print(f"Error initializing tokenizer/model: {e}")
        #     raise

        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection("documents")
        print("DocumentProcessor initialized.")


    def load_documents(self):
        """Load all PDFs and HTML files from the specified data directory"""
        documents = []

         # Check if the directory exists and list files
        if not self.DATA_PATH.exists():
            print(f"Directory does not exist: {self.DATA_PATH}")
            return documents

        files = list(self.DATA_PATH.glob("*"))
        print(f"Files found in directory: {files}")

        # Load PDFs
        for pdf_path in self.DATA_PATH.glob("*.pdf"):
            try:
                print(f"Loading PDF: {pdf_path}")
                loader = PyPDFLoader(str(pdf_path))
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading PDF {pdf_path}: {e}")
        # pdf_loader = PyPDFLoader("source/documents/alumni_scholarship.pdf")
        # documents.extend(pdf_loader.load())


        # Load HTML files
        for html_path in self.DATA_PATH.glob("*.html"):
            try:
                print(f"Loading HTML: {html_path}")
                loader = UnstructuredHTMLLoader(str(html_path))
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading HTML {html_path}: {e}")
        
        # Ensure documents are being loaded
        print(f"DOCUMENT Loaded ")
        return documents

    def process_documents(self):
        documents = self.load_documents()
        for doc in documents:
            if hasattr(doc, 'page_content'):
                text_content = doc.page_content
            else:
                text_content = str(doc)

            chunks = self.text_splitter.split_text(text_content)
            embeddings = self.embedding_model.embed_documents(chunks)
            ids = [str(uuid4()) for _ in chunks]
            self.collection.add(embeddings=embeddings, documents=chunks, ids=ids)


    def process_pdf(self, pdf_path):
        print(f"Processing PDF: {pdf_path}")
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            self._process_documents(documents)
            print(f"Process documents called")
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")

    def process_html(self, html_path):
        try:
            loader = UnstructuredHTMLLoader(html_path)
            documents = loader.load()
            self._process_documents(documents)
        except Exception as e:
            print(f"Error processing HTML {html_path}: {e}")

    def process_text(self, text_path):
        try:
            loader = TextLoader(text_path)
            documents = loader.load()
            self._process_documents(documents)
        except Exception as e:
            print(f"Error processing text file {text_path}: {e}")

    def process_document(self, file_path):
        """Process a document based on its file type."""
        try:
            if file_path.endswith(".pdf"):
                self.process_pdf(file_path)
            elif file_path.endswith(".html"):
                self.process_html(file_path)
            elif file_path.endswith(".txt"):
                self.process_text(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    def _process_documents(self, documents):
        print("Starting to process documents...")
        for doc in documents:
            # Print the type of the document
            print(f"Document type: {type(doc)}")

            # Extract text content from the document
            if hasattr(doc, 'page_content') and isinstance(doc.page_content, str):
                text_content = doc.page_content
            elif isinstance(doc, str):
                text_content = doc
            else:
                raise TypeError(f"Expected a string or an object with a 'page_content' attribute, got {type(doc)}")
            print(f"Text content: {text_content[:100]}...")  # Print the first 100 characters for debugging
            chunks = self.document_processor.text_splitter.split_text(text_content)
            embeddings = self.embed_documents(chunks)
            # Ensure embeddings and chunks are being added
            #print(f"Adding to collection: {chunks}")
            self.collection.add(embeddings=embeddings, documents=chunks)

    # def embed_documents(self, texts):
    #     inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    #     with torch.no_grad():
    #         outputs = self.model(**inputs)
    #     return outputs.last_hidden_state.mean(dim=1).numpy()

    def embed_documents(self, texts):
        """Embed documents using the HuggingFaceEmbeddings model."""
        return self.embedding_model.embed_documents(texts)