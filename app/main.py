# --- main.py ---
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from api.routes import router

app = FastAPI(
    title="RAG Question Generator API",
    description="API for generating questions using RAG with ChromaDB and Ollama",
    version="1.0.0"
)

app.include_router(router, tags=["RAG"])

@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse("/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)