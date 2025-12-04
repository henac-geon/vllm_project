# ============================================
# 고급 PDF → 데이터셋 생성기
# LLM을 활용한 고품질 Q&A 생성
# ============================================

import pdfplumber
import json
import re
from tqdm import tqdm
from typing import List, Dict

class PaperDatasetGenerator:
    """논문 PDF에서 학습용 데이터셋을 생성하는 클래스"""
    
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF에서 텍스트 추출"""
        full_text = ""
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n\n"
        
        return self._clean_text(full_text)
    
    def _clean_text(self, text: str) -> str:
        """텍스트 정제"""
        # 하이픈으로 끊긴 단어 연결
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        # 여러 줄바꿈 정리
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 불필요한 공백 제거
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()
    
    def split_into_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
        """텍스트를 청크로 분할"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [c for c in chunks if len(c) > 100]
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """논문 섹션 추출"""
        section_keywords = {
            'abstract': ['abstract'],
            'introduction': ['introduction', '1. introduction', '1 introduction'],
            'related_work': ['related work', 'background', '2. related', '2 related'],
            'method': ['method', 'methodology', 'approach', 'model', '3. method', '3 method'],
            'experiment': ['experiment', 'evaluation', 'result', '4. experiment', '4 experiment'],
            'discussion': ['discussion', '5. discussion'],
            'conclusion': ['conclusion', '6. conclusion', 'summary'],
        }
        
        sections = {}
        lines = text.lower().split('\n')
        
        current_section = 'header'
        current_content = []
        
        for line in text.split('\n'):
            line_lower = line.lower().strip()
            
            found = False
            for section_name, keywords in section_keywords.items():
                if any(kw in line_lower for kw in keywords) and len(line_lower) < 50:
                    if current_content:
                        sections[current_section] = '\n'.join(current_content)
                    current_section = section_name
                    current_content = []
                    found = True
                    break
            
            if not found:
                current_content.append(line)
        
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def generate_qa_pairs_rule_based(self, chunks: List[str]) -> List[Dict]:
        """규칙 기반 Q&A 쌍 생성"""
        dataset = []
        
        templates = [
            {
                "type": "summary",
                "instruction": "다음 내용을 요약해주세요.",
                "output_template": "요약: {content}"
            },
            {
                "type": "explain",
                "instruction": "다음 학술 내용을 쉽게 설명해주세요.",
                "output_template": "설명: {content}"
            },
            {
                "type": "keypoint",
                "instruction": "다음 내용의 핵심 포인트를 추출해주세요.",
                "output_template": "핵심 포인트:\n{content}"
            },
            {
                "type": "qa",
                "instruction": "다음 텍스트를 바탕으로 질문에 답해주세요.",
                "output_template": "{content}"
            },
        ]
        
        for chunk in chunks:
            # 요약 태스크
            dataset.append({
                "instruction": "다음 내용을 요약해주세요.",
                "input": chunk,
                "output": f"요약: {self._create_summary(chunk)}"
            })
            
            # 설명 태스크
            dataset.append({
                "instruction": "다음 학술 내용을 쉽게 설명해주세요.",
                "input": chunk,
                "output": f"설명: {self._create_explanation(chunk)}"
            })
        
        return dataset
    
    def _create_summary(self, text: str, max_len: int = 200) -> str:
        """간단한 요약 생성 (첫 문장들 추출)"""
        sentences = re.split(r'[.!?]\s+', text)
        summary = ""
        for s in sentences:
            if len(summary) + len(s) < max_len:
                summary += s + ". "
            else:
                break
        return summary.strip() or text[:max_len]
    
    def _create_explanation(self, text: str, max_len: int = 300) -> str:
        """설명 생성"""
        return text[:max_len].strip() + "..."
    
    def generate_qa_pairs_with_llm(self, chunks: List[str], num_questions: int = 2) -> List[Dict]:
        """LLM을 사용한 고품질 Q&A 생성"""
        if not self.model or not self.tokenizer:
            raise ValueError("모델과 토크나이저가 필요합니다.")
        
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(self.model)
        
        dataset = []
        
        for chunk in tqdm(chunks, desc="Q&A 생성 중"):
            # 1. 질문 생성
            questions = self._generate_questions(chunk, num_questions)
            
            # 2. 각 질문에 대한 답변 생성
            for question in questions:
                answer = self._generate_answer(chunk, question)
                
                dataset.append({
                    "instruction": "다음 텍스트를 바탕으로 질문에 답해주세요.",
                    "input": f"텍스트: {chunk}\n\n질문: {question}",
                    "output": answer
                })
        
        return dataset
    
    def _generate_questions(self, context: str, num_questions: int = 2) -> List[str]:
        """컨텍스트에서 질문 생성"""
        prompt = self.alpaca_prompt.format(
            f"다음 텍스트를 읽고, 이 내용에 대해 물어볼 수 있는 핵심 질문 {num_questions}개를 만들어주세요. 각 질문은 새 줄에 번호와 함께 작성하세요.",
            context[:1500],
            ""
        )
        
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.8,
            do_sample=True,
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "### Response:" in response:
            response = response.split("### Response:")[-1]
        
        # 질문 파싱
        questions = []
        for line in response.split('\n'):
            line = re.sub(r'^[\d]+[.)\s]+', '', line.strip())
            if line and '?' in line:
                questions.append(line)
        
        return questions[:num_questions]
    
    def _generate_answer(self, context: str, question: str) -> str:
        """질문에 대한 답변 생성"""
        prompt = self.alpaca_prompt.format(
            "다음 텍스트를 바탕으로 질문에 정확하고 자세하게 답해주세요.",
            f"텍스트: {context[:1500]}\n\n질문: {question}",
            ""
        )
        
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        return response
    
    def generate_section_based_dataset(self, sections: Dict[str, str]) -> List[Dict]:
        """섹션별 instruction 데이터셋 생성"""
        dataset = []
        
        section_prompts = {
            'abstract': [
                ("이 논문의 초록을 설명해주세요.", "초록 내용"),
                ("이 연구의 목적과 결과를 요약해주세요.", "연구 요약"),
            ],
            'introduction': [
                ("이 논문의 연구 배경을 설명해주세요.", "연구 배경"),
                ("이 연구가 해결하고자 하는 문제는 무엇인가요?", "연구 문제"),
            ],
            'method': [
                ("제안하는 방법론을 설명해주세요.", "방법론 설명"),
                ("핵심 알고리즘이나 접근 방식은 무엇인가요?", "핵심 접근"),
            ],
            'experiment': [
                ("실험 설정과 결과를 요약해주세요.", "실험 요약"),
                ("주요 실험 결과는 무엇인가요?", "실험 결과"),
            ],
            'conclusion': [
                ("이 논문의 결론을 요약해주세요.", "결론"),
                ("향후 연구 방향은 무엇인가요?", "향후 연구"),
            ],
        }
        
        for section_name, content in sections.items():
            if len(content) < 100:
                continue
            
            prompts = section_prompts.get(section_name, [
                (f"{section_name} 섹션의 내용을 요약해주세요.", "요약")
            ])
            
            for instruction, prefix in prompts:
                dataset.append({
                    "instruction": instruction,
                    "input": content[:2000],
                    "output": f"{prefix}: {content[:1000]}"
                })
        
        return dataset
    
    def create_full_dataset(self, pdf_path: str, use_llm: bool = False) -> List[Dict]:
        """전체 데이터셋 생성 파이프라인"""
        print(f"📄 PDF 처리 중: {pdf_path}")
        
        # 1. 텍스트 추출
        text = self.extract_text_from_pdf(pdf_path)
        print(f"   텍스트 추출: {len(text)} 문자")
        
        # 2. 청크 분할
        chunks = self.split_into_chunks(text)
        print(f"   청크 분할: {len(chunks)}개")
        
        # 3. 섹션 추출
        sections = self.extract_sections(text)
        print(f"   섹션 발견: {list(sections.keys())}")
        
        # 4. 데이터셋 생성
        dataset = []
        
        # 규칙 기반
        dataset.extend(self.generate_qa_pairs_rule_based(chunks))
        print(f"   규칙 기반 데이터: {len(dataset)}개")
        
        # 섹션 기반
        section_data = self.generate_section_based_dataset(sections)
        dataset.extend(section_data)
        print(f"   섹션 기반 데이터: {len(section_data)}개")
        
        # LLM 기반 (선택)
        if use_llm and self.model:
            llm_data = self.generate_qa_pairs_with_llm(chunks[:10])
            dataset.extend(llm_data)
            print(f"   LLM 기반 데이터: {len(llm_data)}개")
        
        print(f"✅ 총 데이터셋: {len(dataset)}개")
        return dataset
    
    def save_dataset(self, dataset: List[Dict], output_path: str):
        """데이터셋 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"✅ 저장 완료: {output_path}")


# ============================================
# 사용 예시
# ============================================
if __name__ == "__main__":
    # 1. 기본 사용 (규칙 기반만)
    generator = PaperDatasetGenerator()
    dataset = generator.create_full_dataset("paper.pdf", use_llm=False)
    generator.save_dataset(dataset, "paper_dataset.json")
    
    # 2. LLM과 함께 사용 (더 높은 품질)
    # from unsloth import FastLanguageModel
    # model, tokenizer = FastLanguageModel.from_pretrained(...)
    # generator = PaperDatasetGenerator(model, tokenizer)
    # dataset = generator.create_full_dataset("paper.pdf", use_llm=True)
