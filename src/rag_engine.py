import os
import requests
from bs4 import BeautifulSoup
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from typing import List, Dict, Tuple
import re
from urllib.parse import urljoin, urlparse
from config import Config

class RAGEngine:
    def __init__(self):
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.documents = []
        self.embeddings = None
        self.index = None
        self.chunk_size = Config.CHUNK_SIZE_RAG
        self.chunk_overlap = Config.CHUNK_OVERLAP
        
        # Ensure directories exist
        os.makedirs(Config.EMBEDDINGS_DIR, exist_ok=True)
        os.makedirs(Config.DOCUMENTS_DIR, exist_ok=True)
    
    def scrape_website(self, url: str) -> str:
        """Scrape website content"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            print(f"Error scraping website {url}: {e}")
            return ""
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            return ""
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If this isn't the last chunk, try to end at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                last_period = text.rfind('.', end - 100, end)
                last_exclamation = text.rfind('!', end - 100, end)
                last_question = text.rfind('?', end - 100, end)
                
                best_end = max(last_period, last_exclamation, last_question)
                if best_end > start:
                    end = best_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def process_document(self, content: str, source: str) -> List[Dict]:
        """Process document content into chunks with metadata"""
        chunks = self.chunk_text(content)
        documents = []
        
        for i, chunk in enumerate(chunks):
            doc = {
                'content': chunk,
                'source': source,
                'chunk_id': i,
                'word_count': len(chunk.split())
            }
            documents.append(doc)
        
        return documents
    
    def add_url(self, url: str):
        """Add website content to the knowledge base"""
        print(f"Processing URL: {url}")
        content = self.scrape_website(url)
        if content:
            documents = self.process_document(content, url)
            self.documents.extend(documents)
            print(f"Added {len(documents)} chunks from {url}")
        else:
            print(f"No content extracted from {url}")
    
    def add_pdf(self, pdf_path: str):
        """Add PDF content to the knowledge base"""
        print(f"Processing PDF: {pdf_path}")
        content = self.extract_pdf_text(pdf_path)
        if content:
            documents = self.process_document(content, pdf_path)
            self.documents.extend(documents)
            print(f"Added {len(documents)} chunks from {pdf_path}")
        else:
            print(f"No content extracted from {pdf_path}")
    
    def build_index(self):
        """Build FAISS index from documents"""
        if not self.documents:
            print("No documents to index")
            return
        
        print("Building embeddings...")
        texts = [doc['content'] for doc in self.documents]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.index.add(embeddings.astype('float32'))
        
        self.embeddings = embeddings
        print(f"Built index with {len(self.documents)} documents")
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """Search for relevant documents"""
        if top_k is None:
            top_k = Config.TOP_K_RESULTS
            
        if not self.index:
            print("Index not built. Call build_index() first.")
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                result = self.documents[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results
    
    def get_context(self, query: str, max_length: int = None) -> str:
        """Get relevant context for a query"""
        if max_length is None:
            max_length = Config.MAX_CONTEXT_LENGTH
        
        results = self.search(query)
        if not results:
            return ""
        
        context_parts = []
        current_length = 0
        
        for result in results:
            content = result['content']
            if current_length + len(content) <= max_length:
                context_parts.append(f"Source: {result['source']}\n{content}")
                current_length += len(content)
            else:
                # Add partial content if it fits
                remaining_length = max_length - current_length
                if remaining_length > 100:  # Only add if meaningful amount of text
                    partial_content = content[:remaining_length-3] + "..."
                    context_parts.append(f"Source: {result['source']}\n{partial_content}")
                break
        
        return "\n\n".join(context_parts)
    
    def save_index(self, path: str):
        """Save the FAISS index and documents"""
        if self.index is None:
            print("No index to save")
            return
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.faiss")
        
        # Save documents and embeddings
        with open(f"{path}.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings
            }, f)
        
        print(f"Index saved to {path}")
    
    def load_index(self, path: str):
        """Load the FAISS index and documents"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{path}.faiss")
            
            # Load documents and embeddings
            with open(f"{path}.pkl", 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.embeddings = data['embeddings']
            
            print(f"Index loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get statistics about the knowledge base"""
        if not self.documents:
            return {}
        
        total_words = sum(doc['word_count'] for doc in self.documents)
        sources = list(set(doc['source'] for doc in self.documents))
        
        return {
            'total_documents': len(self.documents),
            'total_words': total_words,
            'unique_sources': len(sources),
            'sources': sources,
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0
        }