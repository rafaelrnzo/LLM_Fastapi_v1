import re
from fastapi import HTTPException

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
