# ============================================
# Unsloth LLM Fine-tuning Script
# ============================================

# 1. 필수 패키지 설치 (Colab/Jupyter에서 실행)
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps trl peft accelerate bitsandbytes

# ============================================
# 라이브러리 임포트
# ============================================
from unsloth import FastLanguageModel
from datasets import Dataset, load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
import torch
import json

# ============================================
# 설정
# ============================================
max_seq_length = 2048
dtype = None  # Auto detection (Float16 for Tesla T4, V100, Bfloat16 for Ampere+)
load_in_4bit = True  # 4bit 양자화 사용 (메모리 절약)

# ============================================
# 모델 로드
# ============================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",  # 또는 다른 모델 선택
    # 다른 모델 옵션:
    # "unsloth/mistral-7b-v0.3"
    # "unsloth/gemma-2-9b"
    # "unsloth/Phi-3.5-mini-instruct"
    # "unsloth/Qwen2.5-7B"
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# ============================================
# LoRA 어댑터 설정
# ============================================
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank (8, 16, 32, 64 중 선택)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,  # 0으로 설정하면 최적화됨
    bias="none",
    use_gradient_checkpointing="unsloth",  # 메모리 절약
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# ============================================
# 커스텀 데이터셋 생성 방법
# ============================================

# 방법 1: 직접 데이터 생성
def create_custom_dataset():
    """직접 데이터를 정의하여 데이터셋 생성"""
    
    data = [
        {
            "instruction": "다음 문장을 영어로 번역해주세요.",
            "input": "오늘 날씨가 좋습니다.",
            "output": "The weather is nice today."
        },
        {
            "instruction": "다음 질문에 답해주세요.",
            "input": "파이썬에서 리스트를 정렬하는 방법은?",
            "output": "파이썬에서 리스트를 정렬하는 방법은 두 가지가 있습니다:\n1. sort() 메서드: 원본 리스트를 직접 정렬합니다. 예: my_list.sort()\n2. sorted() 함수: 새로운 정렬된 리스트를 반환합니다. 예: sorted_list = sorted(my_list)"
        },
        {
            "instruction": "주어진 텍스트를 요약해주세요.",
            "input": "인공지능(AI)은 컴퓨터 시스템이 인간의 지능을 모방하여 학습, 추론, 문제 해결 등의 작업을 수행하는 기술입니다. 최근 딥러닝의 발전으로 AI는 이미지 인식, 자연어 처리, 게임 등 다양한 분야에서 혁신적인 성과를 이루고 있습니다.",
            "output": "AI는 인간 지능을 모방하는 컴퓨터 기술로, 딥러닝 발전으로 다양한 분야에서 혁신을 이루고 있습니다."
        },
        {
            "instruction": "코드를 작성해주세요.",
            "input": "피보나치 수열의 n번째 항을 구하는 파이썬 함수",
            "output": "def fibonacci(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\n# 또는 효율적인 반복문 버전\ndef fibonacci_iterative(n):\n    if n <= 0:\n        return 0\n    a, b = 0, 1\n    for _ in range(n-1):\n        a, b = b, a + b\n    return b"
        },
        # 더 많은 데이터 추가...
    ]
    
    return Dataset.from_list(data)


# 방법 2: JSON 파일에서 로드
def load_dataset_from_json(file_path):
    """JSON 파일에서 데이터셋 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)


# 방법 3: Hugging Face Hub에서 로드
def load_dataset_from_hub():
    """Hugging Face Hub에서 데이터셋 로드"""
    dataset = load_dataset("maywell/ko_wikidata_QA", split="train[:1000]")
    return dataset


# ============================================
# 프롬프트 템플릿 정의
# ============================================

# Alpaca 스타일 프롬프트
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# ChatML 스타일 프롬프트 (Qwen, Mistral 등에 적합)
chatml_prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
{}<|im_end|}"""

# Llama 3 스타일 프롬프트
llama3_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{}"""


# ============================================
# 데이터 포맷팅 함수
# ============================================
EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    """데이터를 프롬프트 형식으로 변환"""
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
        texts.append(text)
    
    return {"text": texts}


# ============================================
# 데이터셋 준비
# ============================================
print("데이터셋 준비 중...")

# 커스텀 데이터셋 생성
dataset = create_custom_dataset()

# 또는 JSON에서 로드
# dataset = load_dataset_from_json("your_data.json")

# 또는 Hub에서 로드
# dataset = load_dataset_from_hub()

# 프롬프트 형식으로 변환
dataset = dataset.map(formatting_prompts_func, batched=True)

print(f"데이터셋 크기: {len(dataset)}")
print(f"샘플 데이터:\n{dataset[0]['text'][:500]}...")

# ============================================
# 트레이너 설정
# ============================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=2,
    packing=False,  # 짧은 시퀀스 패킹 (메모리 효율)
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        # num_train_epochs=3,  # 전체 에폭 수
        max_steps=100,  # 또는 최대 스텝 수 지정
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",  # wandb 사용시 "wandb"로 변경
        save_strategy="steps",
        save_steps=50,
    ),
)

# ============================================
# 학습 시작
# ============================================
print("학습 시작...")
trainer_stats = trainer.train()

print(f"\n학습 완료!")
print(f"총 학습 시간: {trainer_stats.metrics['train_runtime']:.2f}초")
print(f"최종 Loss: {trainer_stats.metrics['train_loss']:.4f}")

# ============================================
# 모델 저장
# ============================================

# LoRA 어댑터만 저장 (용량 작음)
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# 전체 모델 병합 후 저장 (16bit)
# model.save_pretrained_merged("model_merged", tokenizer, save_method="merged_16bit")

# GGUF 형식으로 저장 (llama.cpp용)
# model.save_pretrained_gguf("model_gguf", tokenizer, quantization_method="q4_k_m")

print("모델 저장 완료!")

# ============================================
# 추론 테스트
# ============================================
print("\n추론 테스트...")

# 추론 모드로 전환
FastLanguageModel.for_inference(model)

# 테스트 입력
test_input = alpaca_prompt.format(
    "다음 질문에 답해주세요.",
    "머신러닝과 딥러닝의 차이점은 무엇인가요?",
    ""  # 출력은 비워둠
)

inputs = tokenizer([test_input], return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    use_cache=True,
)

response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(f"\n생성된 응답:\n{response}")

# ============================================
# Hugging Face Hub에 업로드 (선택사항)
# ============================================
# model.push_to_hub("your-username/your-model-name", token="your-hf-token")
# tokenizer.push_to_hub("your-username/your-model-name", token="your-hf-token")