from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from api.models import QueryRequest, PDFUploadRequest
from services.vector_store import VectorStoreService
from services.llm_service import LLMService
from services.document_processor import DocumentProcessor

router = APIRouter()
vector_service = VectorStoreService()
llm_service = LLMService()
doc_processor = DocumentProcessor(vector_service)

@router.post("/prompt")
def rag_chain(request: QueryRequest):
    docs = vector_service.retrieve_docs(request.question)
    response = llm_service.generate_mcq(request.question, vector_service.combine_docs(docs))
    return {"query": request.question, "response": response}

@router.post("/prompt-json")
def rag_chain_json(request: QueryRequest):
    try:
        docs = vector_service.retrieve_docs(request.question)
        formatted_context = vector_service.combine_docs(docs)
        
        is_mcq = any(keyword in request.question.lower() 
                    for keyword in ['soal', 'pilihan ganda', 'mcq', 'multiple choice'])
        
        if is_mcq:
            json_response = llm_service.generate_mcq(request.question, formatted_context)
        else:
            json_response = llm_service.generate_json_response(request.question, formatted_context)
        
        return JSONResponse(content={
            "status": "success",
            "query": request.question,
            "response": json_response,
            "metadata": {
                "model": llm_service.model,
                "document_chunks": len(docs),
                "type": "mcq" if is_mcq else "general"
            }
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "query": request.question
            }
        )

@router.post("/prompt-essay")
def rag_chain_essay(request: QueryRequest):
    try:
        docs = vector_service.retrieve_docs(request.question)
        formatted_context = vector_service.combine_docs(docs)
        
        json_response = llm_service.generate_essay(request.question, formatted_context)
        
        return JSONResponse(content={
            "status": "success",
            "query": request.question,
            "response": json_response,
            "metadata": {
                "model": llm_service.model,
                "document_chunks": len(docs),
                "type": "essay"
            }
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "query": request.question
            }
        )

@router.post("/upload-pdf")
def upload_pdf(request: PDFUploadRequest):
    return doc_processor.process_pdf(request.file_path)

@router.delete("/delete-collection")
def delete_collection():
    return vector_service.delete_collection()

