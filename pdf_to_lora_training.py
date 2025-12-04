# ============================================
# PDF 논문 → 데이터셋 → LoRA 학습 파이프라인
# ============================================
# 
# 📌 사용법:
# 1. Google Colab에서 실행
# 2. PDF 파일 업로드
# 3. 셀 순서대로 실행
#
# ============================================

# %% [Cell 1] 패키지 설치
# ============================================
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes
!pip install pdfplumber pypdf sentence-transformers

# %% [Cell 2] 라이브러리 임포트
# ============================================
import pdfplumber
import json
import re
import os
from tqdm import tqdm
from datasets import Dataset
from google.colab import files

print("✅ 라이브러리 로드 완료!")

# %% [Cell 3] PDF 파일 업로드
# ============================================
print("📄 PDF 파일을 업로드하세요...")
uploaded = files.upload()

# 업로드된 PDF 파일 이름 가져오기
pdf_files = [f for f in uploaded.keys() if f.endswith('.pdf')]
print(f"✅ 업로드된 PDF: {pdf_files}")

# %% [Cell 4] PDF에서 텍스트 추출
# ============================================

def extract_text_from_pdf(pdf_path):
    """PDF에서 텍스트 추출"""
    full_text = ""
    
    with pdfplumber.open(pdf_path) as pdf:
        print(f"📖 총 {len(pdf.pages)} 페이지")
        
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                full_text += f"\n\n--- Page {i+1} ---\n\n"
                full_text += text
    
    return full_text


def clean_text(text):
    """텍스트 정제"""
    # 여러 줄바꿈을 하나로
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 불필요한 공백 제거
    text = re.sub(r' {2,}', ' ', text)
    # 하이픈으로 끊긴 단어 연결
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    return text.strip()


# PDF 텍스트 추출
all_texts = {}
for pdf_file in pdf_files:
    print(f"\n📄 처리 중: {pdf_file}")
    raw_text = extract_text_from_pdf(pdf_file)
    cleaned_text = clean_text(raw_text)
    all_texts[pdf_file] = cleaned_text
    print(f"   추출된 텍스트 길이: {len(cleaned_text)} 문자")

# 전체 텍스트 합치기
combined_text = "\n\n".join(all_texts.values())
print(f"\n✅ 전체 텍스트 추출 완료! 총 {len(combined_text)} 문자")

# %% [Cell 5] 텍스트를 청크로 분할
# ============================================

def split_into_chunks(text, chunk_size=1000, overlap=100):
    """텍스트를 청크로 분할"""
    chunks = []
    
    # 문단 단위로 먼저 분리
    paragraphs = text.split('\n\n')
    
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
    
    return chunks


def split_into_sections(text):
    """논문 섹션별로 분리 (Abstract, Introduction 등)"""
    section_patterns = [
        r'(?i)(abstract)',
        r'(?i)(introduction)',
        r'(?i)(related work|background)',
        r'(?i)(method|methodology|approach)',
        r'(?i)(experiment|evaluation|result)',
        r'(?i)(discussion)',
        r'(?i)(conclusion)',
        r'(?i)(reference|bibliography)',
    ]
    
    sections = {}
    current_section = "header"
    current_text = ""
    
    lines = text.split('\n')
    
    for line in lines:
        found_section = False
        for pattern in section_patterns:
            if re.match(pattern, line.strip()):
                if current_text:
                    sections[current_section] = current_text.strip()
                current_section = line.strip().lower()
                current_text = ""
                found_section = True
                break
        
        if not found_section:
            current_text += line + "\n"
    
    if current_text:
        sections[current_section] = current_text.strip()
    
    return sections


# 청크 생성
chunks = split_into_chunks(combined_text, chunk_size=800)
print(f"✅ {len(chunks)}개의 청크로 분할됨")

# 섹션 분리 시도
sections = split_into_sections(combined_text)
print(f"✅ 발견된 섹션: {list(sections.keys())}")

# %% [Cell 6] 데이터셋 생성 - 방법 선택
# ============================================

# 📌 방법 1: 규칙 기반 Q&A 생성
def create_qa_dataset_rule_based(chunks, paper_title="논문"):
    """규칙 기반으로 Q&A 데이터셋 생성"""
    dataset = []
    
    qa_templates = [
        {
            "instruction": "다음 내용을 요약해주세요.",
            "input_prefix": "",
            "output_prefix": "요약: "
        },
        {
            "instruction": "다음 내용의 핵심 포인트를 설명해주세요.",
            "input_prefix": "",
            "output_prefix": "핵심 포인트: "
        },
        {
            "instruction": "다음 텍스트를 바탕으로 질문에 답해주세요.",
            "input_prefix": "텍스트: ",
            "output_prefix": ""
        },
    ]
    
    for i, chunk in enumerate(chunks):
        if len(chunk) < 100:  # 너무 짧은 청크 스킵
            continue
        
        # 요약 태스크
        dataset.append({
            "instruction": "다음 내용을 요약해주세요.",
            "input": chunk,
            "output": f"이 내용은 {paper_title}의 일부로, " + chunk[:200] + "..."
        })
        
        # 설명 태스크
        dataset.append({
            "instruction": "다음 학술 내용을 쉽게 설명해주세요.",
            "input": chunk,
            "output": f"쉽게 설명하면, {chunk[:300]}..."
        })
    
    return dataset


# 📌 방법 2: LLM을 사용한 고품질 Q&A 생성 (권장)
def create_qa_dataset_with_llm(chunks, model, tokenizer, alpaca_prompt):
    """LLM을 사용하여 Q&A 데이터셋 생성"""
    from unsloth import FastLanguageModel
    
    dataset = []
    FastLanguageModel.for_inference(model)
    
    for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="Q&A 생성 중"):
        if len(chunk) < 100:
            continue
        
        # 질문 생성 프롬프트
        question_prompt = alpaca_prompt.format(
            "다음 텍스트를 읽고, 이 내용에 대해 물어볼 수 있는 질문 3개를 만들어주세요. 각 질문은 새 줄에 작성하세요.",
            chunk[:1500],
            ""
        )
        
        inputs = tokenizer([question_prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
        questions_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 질문 파싱
        if "### Response:" in questions_text:
            questions_text = questions_text.split("### Response:")[-1].strip()
        
        questions = [q.strip() for q in questions_text.split('\n') if q.strip() and '?' in q]
        
        # 각 질문에 대한 답변 생성
        for question in questions[:2]:  # 질문 2개만 사용
            answer_prompt = alpaca_prompt.format(
                "다음 텍스트를 바탕으로 질문에 답해주세요.",
                f"텍스트: {chunk[:1500]}\n\n질문: {question}",
                ""
            )
            
            inputs = tokenizer([answer_prompt], return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.3)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "### Response:" in answer:
                answer = answer.split("### Response:")[-1].strip()
            
            dataset.append({
                "instruction": "다음 텍스트를 바탕으로 질문에 답해주세요.",
                "input": f"텍스트: {chunk}\n\n질문: {question}",
                "output": answer
            })
    
    return dataset


# 📌 방법 3: 직접 instruction-output 쌍 생성
def create_instruction_dataset(sections):
    """논문 섹션별로 instruction 데이터셋 생성"""
    dataset = []
    
    section_instructions = {
        "abstract": [
            ("이 논문의 초록을 설명해주세요.", "이 논문의 초록입니다: "),
            ("이 연구의 주요 기여점은 무엇인가요?", "주요 기여점: "),
        ],
        "introduction": [
            ("이 논문의 연구 배경과 동기를 설명해주세요.", "연구 배경: "),
            ("이 연구가 해결하고자 하는 문제는?", "해결하고자 하는 문제: "),
        ],
        "method": [
            ("이 논문의 방법론을 설명해주세요.", "방법론: "),
            ("제안하는 접근 방식의 핵심 아이디어는?", "핵심 아이디어: "),
        ],
        "experiment": [
            ("실험 결과를 요약해주세요.", "실험 결과: "),
            ("어떤 데이터셋과 메트릭을 사용했나요?", "사용된 데이터셋과 메트릭: "),
        ],
        "conclusion": [
            ("이 논문의 결론은 무엇인가요?", "결론: "),
            ("향후 연구 방향은?", "향후 연구: "),
        ],
    }
    
    for section_name, content in sections.items():
        if len(content) < 50:
            continue
            
        # 해당 섹션의 instruction 찾기
        for key, instructions in section_instructions.items():
            if key in section_name.lower():
                for instruction, prefix in instructions:
                    dataset.append({
                        "instruction": instruction,
                        "input": content[:2000],
                        "output": prefix + content[:1000]
                    })
                break
        
        # 기본 요약 instruction
        dataset.append({
            "instruction": f"다음 {section_name} 섹션의 내용을 요약해주세요.",
            "input": content[:2000],
            "output": f"이 섹션의 요약: {content[:800]}"
        })
    
    return dataset

# 규칙 기반 데이터셋 생성 (빠름)
dataset_rules = create_qa_dataset_rule_based(chunks)
print(f"✅ 규칙 기반 데이터셋: {len(dataset_rules)}개 샘플")

# 섹션 기반 데이터셋 생성
dataset_sections = create_instruction_dataset(sections)
print(f"✅ 섹션 기반 데이터셋: {len(dataset_sections)}개 샘플")

# 합치기
training_data = dataset_rules + dataset_sections
print(f"✅ 총 학습 데이터: {len(training_data)}개 샘플")

# %% [Cell 7] 데이터셋 저장 및 확인
# ============================================

# JSON으로 저장
with open("paper_dataset.json", "w", encoding="utf-8") as f:
    json.dump(training_data, f, ensure_ascii=False, indent=2)

print("✅ paper_dataset.json 저장 완료!")

# 샘플 확인
print("\n📋 데이터셋 샘플:")
print("=" * 60)
for i, sample in enumerate(training_data[:3]):
    print(f"\n[샘플 {i+1}]")
    print(f"Instruction: {sample['instruction'][:100]}...")
    print(f"Input: {sample['input'][:150]}...")
    print(f"Output: {sample['output'][:150]}...")
    print("-" * 60)

# %% [Cell 8] 모델 로드 및 LoRA 설정
# ============================================
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
import torch

# 설정
max_seq_length = 2048
dtype = None
load_in_4bit = True

# 🔥 GPT 오픈소스 모델 선택 (원하는 모델 주석 해제)
model_name = "unsloth/Meta-Llama-3.1-8B"        # Llama 3.1 8B
# model_name = "unsloth/mistral-7b-v0.3"        # Mistral 7B
# model_name = "unsloth/gemma-2-9b"              # Gemma 2 9B
# model_name = "unsloth/Qwen2.5-7B"              # Qwen 2.5 7B
# model_name = "unsloth/Phi-3.5-mini-instruct"  # Phi 3.5 Mini

print(f"🚀 모델 로드 중: {model_name}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

print("✅ 모델 로드 완료!")

# LoRA 어댑터 설정
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

print("✅ LoRA 어댑터 설정 완료!")

# %% [Cell 9] 데이터 포맷팅
# ============================================

# Alpaca 프롬프트 템플릿
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    texts = []
    for instruction, input_text, output in zip(
        examples["instruction"],
        examples["input"],
        examples["output"]
    ):
        text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# Dataset 객체로 변환
dataset = Dataset.from_list(training_data)
dataset = dataset.map(formatting_prompts_func, batched=True)

print(f"✅ 데이터셋 준비 완료! ({len(dataset)}개 샘플)")

# %% [Cell 10] 학습 실행
# ============================================

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=3,          # 전체 데이터 3번 반복
        # max_steps=100,             # 또는 최대 스텝 수 지정
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",  # cosine 스케줄러
        seed=3407,
        output_dir="outputs",
        report_to="none",
        save_strategy="epoch",
    ),
)

print("🚀 학습 시작!")
print("=" * 60)

trainer_stats = trainer.train()

print("\n" + "=" * 60)
print("✅ 학습 완료!")
print(f"⏱️ 총 학습 시간: {trainer_stats.metrics['train_runtime']:.2f}초")
print(f"📉 최종 Loss: {trainer_stats.metrics['train_loss']:.4f}")

# %% [Cell 11] 모델 저장
# ============================================

# LoRA 어댑터 저장
model.save_pretrained("paper_lora_model")
tokenizer.save_pretrained("paper_lora_model")

print("✅ LoRA 모델 저장 완료! (paper_lora_model)")

# Google Drive에 저장
from google.colab import drive
drive.mount('/content/drive')

import shutil
shutil.copytree("paper_lora_model", "/content/drive/MyDrive/paper_lora_model")
print("✅ Google Drive에 저장 완료!")

# %% [Cell 12] 추론 테스트
# ============================================

# 추론 모드 전환
FastLanguageModel.for_inference(model)

def ask_paper(question, context=""):
    """논문 관련 질문하기"""
    if context:
        input_text = f"텍스트: {context}\n\n질문: {question}"
    else:
        input_text = question
    
    prompt = alpaca_prompt.format(
        "다음 질문에 답해주세요.",
        input_text,
        ""
    )
    
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        use_cache=True,
    )
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    if "### Response:" in response:
        return response.split("### Response:")[-1].strip()
    return response


# 테스트 질문
print("=" * 60)
print("🤖 모델 테스트")
print("=" * 60)

test_questions = [
    "이 논문의 주요 기여점은 무엇인가요?",
    "제안하는 방법론의 핵심 아이디어를 설명해주세요.",
    "실험 결과를 요약해주세요.",
]

# 첫 번째 청크를 컨텍스트로 사용
context = chunks[0] if chunks else ""

for q in test_questions:
    print(f"\n❓ 질문: {q}")
    answer = ask_paper(q, context)
    print(f"💬 답변: {answer[:500]}...")
    print("-" * 40)

# %% [Cell 13] (선택) LLM으로 고품질 데이터셋 생성
# ============================================
# 모델 로드 후 실행하면 더 좋은 품질의 데이터셋 생성

# dataset_llm = create_qa_dataset_with_llm(chunks[:20], model, tokenizer, alpaca_prompt)
# print(f"✅ LLM 생성 데이터셋: {len(dataset_llm)}개")

# # 기존 데이터와 합치기
# training_data_enhanced = training_data + dataset_llm

# %% [Cell 14] (선택) GGUF 형식으로 내보내기
# ============================================
# llama.cpp, Ollama 등에서 사용 가능

# q4_k_m: 좋은 품질 + 작은 용량 (권장)
# model.save_pretrained_gguf("paper_model_gguf", tokenizer, quantization_method="q4_k_m")

# q8_0: 더 높은 품질
# model.save_pretrained_gguf("paper_model_gguf_q8", tokenizer, quantization_method="q8_0")

# %% [Cell 15] (선택) Hugging Face Hub 업로드
# ============================================

# from huggingface_hub import login
# login(token="YOUR_HF_TOKEN")  # https://huggingface.co/settings/tokens

# model.push_to_hub("your-username/paper-finetuned-llama")
# tokenizer.push_to_hub("your-username/paper-finetuned-llama")

print("\n🎉 모든 작업 완료!")
print("=" * 60)
print("📁 저장된 파일:")
print("   - paper_dataset.json (학습 데이터)")
print("   - paper_lora_model/ (LoRA 어댑터)")
print("   - Google Drive에 백업됨")
