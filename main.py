import os
import re
import json
import chromadb
import ollama
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = FastAPI()

CHROMA_HOST = "192.168.100.3"
CHROMA_PORT = 8000
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "default_collection")
OLLAMA_HOST = "http://192.168.100.3:11434"
OLLAMA_MODEL = "llama3.2:latest"
ollama.base_url = OLLAMA_HOST

chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, settings=Settings())
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    embedding_function=embedding_function,
    collection_name=CHROMA_COLLECTION_NAME,
    client=chroma_client,
)
retriever = vectorstore.as_retriever()

class QueryRequest(BaseModel):
    question: str

class PDFUploadRequest(BaseModel):
    file_path: str

class EssayQuestion(BaseModel):
    number: int
    question: str
    answer: str
    explanation: str = None

class EssayResponse(BaseModel):
    total_questions: int
    questions: list[EssayQuestion]

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_mcq_prompt(question: str, context: str) -> str:
    return f"""Berdasarkan konteks berikut, buatkan soal pilihan ganda sesuai permintaan.
    Setiap soal HARUS memiliki pertanyaan yang lengkap (bukan hanya nomor).
    
    Format yang WAJIB diikuti untuk setiap soal:
    
    Soal [nomor]:
    [Tulis pertanyaan lengkap disini (WAJIB)]
    A) [pilihan A]
    B) [pilihan B]
    C) [pilihan C]
    D) [pilihan D]
    Jawaban: [A/B/C/D]
    
    Contoh format yang benar:
    Soal 1:
    Apakah fungsi utama dari katup bahan bakar pada sistem MPK?
    A) Mengatur aliran
    B) Menyaring kotoran
    C) Mengukur tekanan
    D) Menghentikan aliran
    Jawaban: A
    
    Perhatikan bahwa setiap soal HARUS memiliki:
    1. Pertanyaan lengkap (bukan hanya nomor)
    2. Empat pilihan jawaban (A,B,C,D)
    3. Satu jawaban yang benar
    
    Konteks: {context}
    
    Permintaan: {question}"""

def format_essay_prompt(question: str, context: str) -> str:
    return f"""Berdasarkan konteks berikut, buatkan soal essay sesuai permintaan.
    Format setiap soal dengan struktur yang konsisten seperti berikut:
    
    Soal [nomor]:
    [pertanyaan lengkap]
    
    Jawaban:
    [jawaban lengkap]
    
    Penjelasan:
    [penjelasan detail tentang jawaban]
    
    Pastikan setiap soal memiliki:
    1. Pertanyaan yang jelas dan mendetail
    2. Jawaban yang komprehensif
    3. Penjelasan tambahan yang membantu pemahaman
    
    Konteks: {context}
    
    Permintaan: {question}"""

def format_prompt_for_json(question: str, context: str) -> str:
    return f"""Based on the following context, answer the question and format your response as valid JSON.
    The JSON should include 'answer' and 'confidence' fields.
    
    Context: {context}
    
    Question: {question}
    
    Respond with valid JSON only, following this structure:
    {{
        "answer": "your detailed answer here",
        "confidence": 0.95,
        "references": ["relevant reference 1", "relevant reference 2"],
        "tags": ["relevant_tag1", "relevant_tag2"]
    }}"""

def parse_mcq_text(content: str):
    try:
        questions = []
        raw_questions = content.split("Soal")[1:]
        
        for i, raw_question in enumerate(raw_questions, 1):
            try:
                lines = [line.strip() for line in raw_question.strip().split('\n') if line.strip()]
                
                question_text = ""
                options_started = False
                question_lines = []
                
                for line in lines:
                    if line.startswith('A)') or line.startswith('A.'):
                        options_started = True
                        break
                    question_lines.append(line)
                
                question_text = ' '.join(question_lines)
                question_text = re.sub(r'^\d+:\s*', '', question_text)
                
                options = {}
                current_option = None
                for line in lines:
                    if line.startswith(('A)', 'B)', 'C)', 'D)', 'A.', 'B.', 'C.', 'D.')):
                        current_option = line[0]
                        options[current_option] = line[2:].strip()
                
                answer = None
                for line in lines:
                    if "Jawaban:" in line:
                        answer_patterns = [
                            r'Jawaban:\s*([A-D])\)',
                            r'Jawaban:\s*([A-D])\.',
                            r'Jawaban:\s*([A-D])',
                        ]
                        for pattern in answer_patterns:
                            match = re.search(pattern, line)
                            if match:
                                answer = match.group(1)
                                break
                        if not answer:
                            answer = line.split(":")[-1].strip()
                            if answer in ['A', 'B', 'C', 'D']:
                                break
                
                if question_text and options and answer:
                    questions.append({
                        "number": i,
                        "question": question_text.strip(),
                        "options": {
                            "A": options.get("A", ""),
                            "B": options.get("B", ""),
                            "C": options.get("C", ""),
                            "D": options.get("D", "")
                        },
                        "answer": answer
                    })
            except Exception as e:
                print(f"Error parsing question {i}: {str(e)}")
                continue
        
        return {
            "total_questions": len(questions),
            "questions": questions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing MCQ response: {str(e)}")

def parse_essay_text(content: str):
    try:
        questions = []
        raw_questions = content.split("Soal")[1:]
        
        for i, raw_question in enumerate(raw_questions, 1):
            try:
                sections = raw_question.strip().split("\n\n")
                
                question_text = sections[0].strip()
                if question_text.startswith(f"{i}:"):
                    question_text = question_text[len(f"{i}:"):].strip()
                
                answer_text = ""
                explanation_text = ""
                
                for section in sections:
                    if section.lower().startswith("jawaban:"):
                        answer_text = section[8:].strip()
                    elif section.lower().startswith("penjelasan:"):
                        explanation_text = section[10:].strip()
                
                if question_text and answer_text:
                    questions.append({
                        "number": i,
                        "question": question_text,
                        "answer": answer_text,
                        "explanation": explanation_text if explanation_text else None
                    })
            except Exception as e:
                print(f"Error parsing essay question {i}: {str(e)}")
                continue
        
        return {
            "total_questions": len(questions),
            "questions": questions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing essay response: {str(e)}")

def ollama_llm_json(question: str, context: str):
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{
                'role': 'user', 
                'content': format_prompt_for_json(question, context)
            }]
        )
        
        content = response['message']['content']
        try:
            parsed_json = json.loads(content)
        except json.JSONDecodeError:
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                parsed_json = json.loads(json_match.group(1))
            else:
                raise HTTPException(status_code=500, detail="Failed to parse LLM response as JSON")
        
        return parsed_json
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing LLM response: {str(e)}")

def ollama_llm_mcq(question: str, context: str):
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{
                'role': 'user', 
                'content': format_mcq_prompt(question, context)
            }]
        )
        
        content = response['message']['content']
        parsed_json = parse_mcq_text(content)
        return parsed_json
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing LLM response: {str(e)}")

def ollama_llm_essay(question: str, context: str):
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{
                'role': 'user', 
                'content': format_essay_prompt(question, context)
            }]
        )
        
        content = response['message']['content']
        parsed_json = parse_essay_text(content)
        return parsed_json
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing LLM response: {str(e)}")

def delete_all_chroma_data():
    try:
        chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
        return {"message": f"Collection '{CHROMA_COLLECTION_NAME}' deleted successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting ChromaDB collection: {e}")

@app.post("/prompt")
def rag_chain(request: QueryRequest):
    docs = retriever.invoke(request.question)
    response = ollama_llm_mcq(request.question, combine_docs(docs))
    return {"query": request.question, "response": response}

@app.post("/prompt-json")
def rag_chain_json(request: QueryRequest):
    try:
        docs = retriever.invoke(request.question)
        formatted_context = combine_docs(docs)
        
        is_mcq = any(keyword in request.question.lower() 
                    for keyword in ['soal', 'pilihan ganda', 'mcq', 'multiple choice'])
        
        if is_mcq:
            json_response = ollama_llm_mcq(request.question, formatted_context)
        else:
            json_response = ollama_llm_json(request.question, formatted_context)
        
        return JSONResponse(content={
            "status": "success",
            "query": request.question,
            "response": json_response,
            "metadata": {
                "model": OLLAMA_MODEL,
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

@app.post("/prompt-essay")
def rag_chain_essay(request: QueryRequest):
    try:
        docs = retriever.invoke(request.question)
        formatted_context = combine_docs(docs)
        
        json_response = ollama_llm_essay(request.question, formatted_context)
        
        return JSONResponse(content={
            "status": "success",
            "query": request.question,
            "response": json_response,
            "metadata": {
                "model": OLLAMA_MODEL,
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

@app.post("/upload-pdf")
def upload_pdf(request: PDFUploadRequest):
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
    
    loader = PyPDFLoader(request.file_path)
    documents = loader.load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200).split_documents(documents)
    
    Chroma.from_documents(documents=chunks, embedding=embedding_function, collection_name=CHROMA_COLLECTION_NAME, client=chroma_client)
    return {"message": f"Added {len(chunks)} chunks to ChromaDB in collection '{CHROMA_COLLECTION_NAME}'"}

@app.delete("/delete-collection")
def delete_collection():
    return delete_all_chroma_data()

@app.get("/")
async def redirect_to_docs():
    return RedirectResponse("/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)