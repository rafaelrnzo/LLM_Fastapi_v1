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
        
    def get_vectorstore(self, collection_name: str = None):
        return Chroma(
            embedding_function=self.embedding_function,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            client=self.chroma_client,
        )

    def retrieve_docs(self, question, collection_name: str = None):
        vectorstore = self.get_vectorstore(collection_name)
        retriever = vectorstore.as_retriever()
        return retriever.invoke(question)

    def combine_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def delete_collection(self, collection_name: str):
        try:
            self.chroma_client.delete_collection(name=collection_name)
            return {"message": f"Collection '{collection_name}' deleted successfully!"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting ChromaDB collection: {e}")

    def get_collection_list(self):
        try:
            collection_names = self.chroma_client.list_collections()
            return {"collections": collection_names}  
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error when getting ChromaDB collection list: {e}")

def get_vector_store():
    return VectorStoreService()
