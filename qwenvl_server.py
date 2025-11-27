"""
Qwen3-VL Image & Text Chat Server
이미지와 텍스트를 받아 분석하는 챗봇 시스템
"""
import os
import io
import base64
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from PIL import Image
import uvicorn
from transformers import AutoModelForImageTextToText, AutoProcessor
import time
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="Qwen3-VL Chat Server",
    description="이미지와 텍스트를 분석하는 멀티모달 챗봇 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files 설정
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Pydantic 모델 정의
class ChatRequest(BaseModel):
    text: str
    image_base64: Optional[str] = None
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7

class ChatResponse(BaseModel):
    response: str
    inference_time: float
    status: str

# 전역 모델 변수
model = None
processor = None

def load_model():
    """모델과 프로세서를 로드합니다."""
    global model, processor

    logger.info(" 모델 로딩 중...")

    # Disable XET and enable offline mode
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_ENABLE_XET"] = "0"

    try:
        # 모델 로딩
        model = AutoModelForImageTextToText.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct",
            dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True
        )
        model.eval()

        # 프로세서 로딩
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct",
            local_files_only=True
        )

        logger.info("✅ 모델 로딩 완료!")
        return True

    except Exception as e:
        logger.error(f"❌ 모델 로딩 실패: {str(e)}")
        return False

def process_image(image_data):
    """이미지 데이터를 처리합니다."""
    try:
        if isinstance(image_data, str):
            # Base64 디코딩
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        elif isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        else:
            image = image_data

        # RGB 변환
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return image
    except Exception as e:
        logger.error(f"이미지 처리 오류: {str(e)}")
        raise HTTPException(status_code=400, detail=f"이미지 처리 실패: {str(e)}")

def generate_response(text: str, image=None, max_tokens: int = 256, temperature: float = 0.7):
    """텍스트와 이미지를 기반으로 응답을 생성합니다."""
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")

    try:
        # 메시지 구성
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": text})

        messages = [{"role": "user", "content": content}]

        # 입력 준비
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        # 추론 시작
        start_time = time.time()

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.9 if temperature > 0 else None,
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        inference_time = time.time() - start_time

        # 결과 디코딩
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output_text, inference_time

    except Exception as e:
        logger.error(f"추론 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"추론 실패: {str(e)}")

# API 엔드포인트
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델을 로드합니다."""
    success = load_model()
    if not success:
        logger.warning("⚠️  서버가 시작되었지만 모델 로딩에 실패했습니다.")

@app.get("/", response_class=HTMLResponse)
async def root():
    """웹 인터페이스를 제공합니다."""
    html_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_file):
        with open(html_file, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return """
        <html>
            <body>
                <h1>Qwen3-VL Chat Server</h1>
                <p>서버가 실행 중입니다.</p>
                <p>API 문서: <a href="/docs">/docs</a></p>
            </body>
        </html>
        """

@app.get("/api")
async def api_root():
    """API 서버 상태를 확인합니다."""
    return {
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "chat": "/chat",
            "chat_with_file": "/chat/upload",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    텍스트와 이미지(base64)를 받아 응답을 생성합니다.

    - **text**: 사용자 질문 (필수)
    - **image_base64**: Base64 인코딩된 이미지 (선택)
    - **max_tokens**: 최대 생성 토큰 수 (기본: 256)
    - **temperature**: 생성 온도 (기본: 0.7, 0이면 greedy decoding)
    """
    try:
        image = None
        if request.image_base64:
            image = process_image(request.image_base64)

        response_text, inference_time = generate_response(
            text=request.text,
            image=image,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        return ChatResponse(
            response=response_text,
            inference_time=round(inference_time, 2),
            status="success"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/upload", response_model=ChatResponse)
async def chat_with_upload(
    text: str = Form(...),
    image: Optional[UploadFile] = File(None),
    max_tokens: int = Form(256),
    temperature: float = Form(0.7)
):
    """
    파일 업로드 방식으로 이미지와 텍스트를 받아 응답을 생성합니다.

    - **text**: 사용자 질문 (필수)
    - **image**: 이미지 파일 (선택)
    - **max_tokens**: 최대 생성 토큰 수 (기본: 256)
    - **temperature**: 생성 온도 (기본: 0.7)
    """
    try:
        image_obj = None
        if image:
            # 이미지 파일 읽기
            contents = await image.read()
            image_obj = process_image(contents)

        response_text, inference_time = generate_response(
            text=text,
            image=image_obj,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return ChatResponse(
            response=response_text,
            inference_time=round(inference_time, 2),
            status="success"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat with upload 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 서버 실행
    uvicorn.run(
        "chat_server:app",
        host="0.0.0.0",
        port=24112,
        reload=False,
        log_level="info"
    )
