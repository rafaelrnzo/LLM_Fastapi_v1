import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from config import settings
from fastapi import HTTPException

class VectorStoreService:
    def __init__(self):
        self.chroma_client = chromadb.HttpClient(
            host=settings.CHROMA_HOST, 
            port=settings.CHROMA_PORT, 
            settings=ChromaSettings()
        )
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma(
            embedding_function=self.embedding_function,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            client=self.chroma_client,
        )
        self.retriever = self.vectorstore.as_retriever()
    
    def retrieve_docs(self, question):
        return self.retriever.invoke(question)
    
    def combine_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def delete_collection(self):
        try:
            self.chroma_client.delete_collection(name=settings.CHROMA_COLLECTION_NAME)
            return {"message": f"Collection '{settings.CHROMA_COLLECTION_NAME}' deleted successfully!"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting ChromaDB collection: {e}")

