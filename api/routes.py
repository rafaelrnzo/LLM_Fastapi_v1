from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from api.models import QueryRequest, PDFUploadRequest
from services.vector_store import VectorStoreService
from services.llm_service import LLMService
from services.document_processor import DocumentProcessor
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Query
from services.vector_store import get_vector_store
from services.document_processor import DocumentProcessor
import os
import shutil


router = APIRouter()
vector_service = VectorStoreService()
llm_service = LLMService()
doc_processor = DocumentProcessor(vector_service)

# @router.post("/prompt")
# def rag_chain(request: QueryRequest):
#     docs = vector_service.retrieve_docs(request.question)
#     response = llm_service.generate_mcq(request.question, vector_service.combine_docs(docs))
#     return {"query": request.question, "response": response}

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
def upload_pdf(
    file: UploadFile = File(...),
    collection_name: str = Form(None),
    vector_service=Depends(get_vector_store)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    temp_dir = "./temp_files"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    doc_processor = DocumentProcessor(vector_service)
    return doc_processor.process_pdf(file_path, collection_name)
    
    
@router.get("/collection-list")
def get_collection_list():
    return vector_service.get_collection_list()


@router.delete("/delete-collection")
def delete_collection(collection_name: str = Query(..., description="Name of the collection to delete")):
    return vector_service.delete_collection(collection_name)