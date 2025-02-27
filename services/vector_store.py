import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from config import settings
from fastapi import HTTPException

class VectorStoreService:
    def __init__(self):
        try:
            self.chroma_client = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT,
                settings=ChromaSettings()
            )
            self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to ChromaDB: {e}")

    def get_vectorstore(self, collection_name: str):
        if not collection_name:
            raise ValueError("Collection name must be provided")

        return Chroma(
            embedding_function=self.embedding_function,
            collection_name=collection_name,
            client=self.chroma_client,
        )

    def retrieve_docs(self, question: str, collection_name: str):
        try:
            vectorstore = self.get_vectorstore(collection_name)
            retriever = vectorstore.as_retriever()
            return retriever.invoke(question)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error retrieving documents: {e}")

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
            collections = self.chroma_client.list_collections()
            return {"collections": collections}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error retrieving ChromaDB collections: {e}")

vector_store_service = VectorStoreService()

def get_vector_store():
    return vector_store_service