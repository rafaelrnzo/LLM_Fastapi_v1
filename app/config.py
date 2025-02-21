import os

class Settings:
    CHROMA_HOST = "192.168.100.3"
    CHROMA_PORT = 8000
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "default_collection")
    OLLAMA_HOST = "http://192.168.100.3:11434"
    OLLAMA_MODEL = "llama3.2:latest"

settings = Settings()
