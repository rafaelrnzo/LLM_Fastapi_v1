import os
import shutil
from fastapi import HTTPException
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from services.vector_store import VectorStoreService
from config import settings

class DocumentProcessor:
    def __init__(self, vector_service: VectorStoreService):
        self.vector_service = vector_service
    
    def process_pdf(self, file_path: str, collection_name: str = None):
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        if not file_path.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

        if collection_name is None:
            collection_name = settings.CHROMA_COLLECTION_NAME
        
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200).split_documents(documents)
        
        Chroma.from_documents(
            documents=chunks, 
            embedding=self.vector_service.embedding_function, 
            collection_name=collection_name, 
            client=self.vector_service.chroma_client
        )
        
        os.remove(file_path)  # Hapus file setelah diproses
        return {"message": f"Added {len(chunks)} chunks to ChromaDB in collection '{collection_name}'"}
