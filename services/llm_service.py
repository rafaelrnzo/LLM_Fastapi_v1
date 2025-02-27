import json
import re
import ollama
from fastapi import HTTPException
from utils.text_parsers import parse_mcq_text, parse_essay_text
from config import settings

class LLMService:
    def __init__(self):
        ollama.base_url = settings.OLLAMA_HOST
        self.model = settings.OLLAMA_MODEL
    
    @staticmethod
    def format_mcq_prompt(question: str, context: str) -> str:
        return f"""Berdasarkan konteks berikut, buatkan soal pilihan ganda sesuai permintaan.
        Setiap soal HARUS memiliki pertanyaan yang lengkap (bukan hanya nomor).
        
        Format yang WAJIB diikuti untuk setiap soal:
        
        Soal [nomor]:
        [Tulis pertanyaan lengkap di sini (WAJIB)]
        A) [pilihan A]
        B) [pilihan B]
        C) [pilihan C]
        D) [pilihan D]
        Jawaban: [A/B/C/D] (Jawaban benar harus tersebar secara acak, tidak boleh hanya di A atau B)
        
        Contoh format yang benar:
        Soal 1:
        Apakah fungsi utama dari katup bahan bakar pada sistem MPK?
        A) Mengatur aliran
        B) Menyaring kotoran
        C) Mengukur tekanan
        D) Menghentikan aliran
        Jawaban: A
        
        Soal 2:
        Apa yang dimaksud dengan titik didih dalam ilmu fisika?
        A) Titik di mana cairan membeku
        B) Titik di mana cairan menguap menjadi gas
        C) Titik di mana zat padat mencair
        D) Titik di mana gas berubah menjadi cair
        Jawaban: B
        
        Perhatikan bahwa setiap soal HARUS memiliki:
        1. Pertanyaan lengkap (bukan hanya nomor).
        2. Empat pilihan jawaban (A, B, C, D).
        3. **Jawaban benar harus tersebar di antara A, B, C, dan D, bukan hanya di A atau B.**
        4. Pastikan tidak ada soal yang terduplifikasi.
        5. Pastikan total soal yang dibuat sesuai dengan soal yang saya minta di prompt saya.
        
        Konteks: {context}
        
        Permintaan: {question}"""


    def format_essay_prompt(self, question: str, context: str) -> str:
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

    def format_prompt_for_json(self, question: str, context: str) -> str:
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
    
    def generate_mcq(self, question: str, context: str):
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user', 
                    'content': self.format_mcq_prompt(question, context)
                }]
            )
            
            content = response['message']['content']
            parsed_json = parse_mcq_text(content)
            return parsed_json
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing LLM response: {str(e)}")
    
    def generate_essay(self, question: str, context: str):
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user', 
                    'content': self.format_essay_prompt(question, context)
                }]
            )
            
            content = response['message']['content']
            parsed_json = parse_essay_text(content)
            return parsed_json
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing LLM response: {str(e)}")
    
    def generate_json_response(self, question: str, context: str):
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user', 
                    'content': self.format_prompt_for_json(question, context)
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
