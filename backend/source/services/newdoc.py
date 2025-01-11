def process_documents(self):
    """Main method to process all documents"""
    print("Starting document processing...")
    documents = self.load_documents()
    
    for doc in documents:
        try:
            # Get the source file path if available
            source_path = doc.metadata.get('source', '') if hasattr(doc, 'metadata') else ''
            
            if source_path.endswith('.pdf'):
                print(f"Processing PDF: {source_path}")
                self.process_pdf(source_path)
            elif source_path.endswith('.html'):
                print(f"Processing HTML: {source_path}")
                self.process_html(source_path)
            else:
                # Process generic document content
                if hasattr(doc, 'page_content'):
                    text_content = doc.page_content
                else:
                    text_content = str(doc)
                
                chunks = self.text_splitter.split_text(text_content)
                embeddings = self.embedding_model.embed_documents(chunks)
                ids = [str(uuid4()) for _ in chunks]
                self.collection.add(embeddings=embeddings, documents=chunks, ids=ids)
                
        except Exception as e:
            print(f"Error processing document: {e}")
            continue

    print("Document processing completed")

def process_pdf(self, pdf_path):
    """Process a single PDF file"""
    print(f"Processing PDF: {pdf_path}")
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        for doc in documents:
            chunks = self.text_splitter.split_text(doc.page_content)
            embeddings = self.embedding_model.embed_documents(chunks)
            ids = [str(uuid4()) for _ in chunks]
            self.collection.add(embeddings=embeddings, documents=chunks, ids=ids)
            
        print(f"Successfully processed PDF: {pdf_path}")
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")

def process_html(self, html_path):
    """Process a single HTML file"""
    print(f"Processing HTML: {html_path}")
    try:
        loader = UnstructuredHTMLLoader(html_path)
        documents = loader.load()
        
        for doc in documents:
            chunks = self.text_splitter.split_text(doc.page_content)
            embeddings = self.embedding_model.embed_documents(chunks)
            ids = [str(uuid4()) for _ in chunks]
            self.collection.add(embeddings=embeddings, documents=chunks, ids=ids)
            
        print(f"Successfully processed HTML: {html_path}")
    except Exception as e:
        print(f"Error processing HTML {html_path}: {e}")