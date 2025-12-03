"""
VARCO-VISION Image & Video & Text Chat Server
이미지, 동영상, 텍스트를 받아 분석하는 챗봇 시스템
FastAPI 기반의 REST API 서버로 VARCO-VISION 비전 언어 모델을 서빙합니다.
"""

# ===== 표준 라이브러리 임포트 =====
import os          # 운영체제 관련 기능 (환경 변수, 파일 경로 등)
import io          # 입출력 스트림 처리 (메모리 상의 파일 객체)
import base64      # Base64 인코딩/디코딩 (이미지를 텍스트로 변환)
import time        # 시간 측정 (추론 시간 계산용)
import logging     # 로깅 기능 (서버 로그 출력)

# ===== PyTorch 임포트 =====
import torch       # PyTorch 딥러닝 프레임워크 (GPU 연산, 텐서 처리)

# ===== FastAPI 관련 임포트 =====
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
# FastAPI: FastAPI 애플리케이션 메인 클래스
# File: 파일 업로드 파라미터 정의
# UploadFile: 업로드된 파일 객체
# Form: 폼 데이터 파라미터 정의
# HTTPException: HTTP 예외 처리 (400, 500 등)

from fastapi.responses import JSONResponse, HTMLResponse
# JSONResponse: JSON 형식 응답
# HTMLResponse: HTML 형식 응답

from fastapi.staticfiles import StaticFiles
# StaticFiles: 정적 파일 서빙 (HTML, CSS, JS 등)

from fastapi.middleware.cors import CORSMiddleware
# CORSMiddleware: CORS(Cross-Origin Resource Sharing) 설정 미들웨어

# ===== Pydantic 임포트 =====
from pydantic import BaseModel
# BaseModel: 데이터 검증 및 직렬화를 위한 기본 클래스

# ===== 타입 힌팅 임포트 =====
from typing import Optional, List
# Optional: 선택적 파라미터 타입 (None 허용)
# List: 리스트 타입 힌팅

# ===== 이미지 처리 라이브러리 =====
from PIL import Image
# Pillow 라이브러리의 Image 모듈 (이미지 읽기, 변환 등)

# ===== 동영상 처리 라이브러리 =====
import cv2
# OpenCV 라이브러리 (동영상에서 프레임 추출)
import numpy as np
# NumPy 라이브러리 (배열 처리)

# ===== ASGI 서버 =====
import uvicorn
# Uvicorn: ASGI 웹 서버 (FastAPI 앱 실행용)

# ===== Transformers 라이브러리 =====
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
# AutoProcessor: 모델에 맞는 전처리기 자동 로딩
# LlavaOnevisionForConditionalGeneration: VARCO-VISION 모델


# ===== 로깅 설정 =====
logging.basicConfig(level=logging.INFO)  # 로깅 레벨을 INFO로 설정
logger = logging.getLogger(__name__)     # 현재 모듈의 로거 객체 생성


# ===== FastAPI 애플리케이션 초기화 =====
app = FastAPI(
    title="VARCO-VISION Chat Server",
    description="이미지, 동영상, 텍스트를 분석하는 멀티모달 챗봇 API",
    version="1.0.0"
)


# ===== CORS 미들웨어 설정 =====
# 다른 도메인에서 이 API를 호출할 수 있도록 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                # 모든 출처(도메인) 허용
    allow_credentials=True,             # 쿠키 및 인증 정보 허용
    allow_methods=["*"],                # 모든 HTTP 메서드 허용
    allow_headers=["*"],                # 모든 HTTP 헤더 허용
)


# ===== 정적 파일 서빙 설정 =====
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ===== Pydantic 데이터 모델 정의 =====
# 클라이언트가 보내는 요청 데이터 구조
class ChatRequest(BaseModel):
    text: str                           # 사용자 질문 텍스트 (필수)
    image_base64: Optional[str] = None  # Base64 인코딩된 이미지 (선택)
    images_base64: Optional[List[str]] = None  # Base64 인코딩된 이미지 배열 (선택)
    max_tokens: Optional[int] = 1024    # 생성할 최대 토큰 수 (VARCO-VISION 기본값)
    temperature: Optional[float] = 0.7  # 생성 온도

# 서버가 반환하는 응답 데이터 구조
class ChatResponse(BaseModel):
    response: str                       # AI가 생성한 응답 텍스트
    inference_time: float               # 추론에 걸린 시간 (초)
    status: str                         # 요청 처리 상태


# ===== 전역 모델 변수 =====
model = None      # VARCO-VISION 모델 객체
processor = None  # 모델 전처리기 객체


# ===== 모델 로딩 함수 =====
def load_model():
    """
    VARCO-VISION 모델과 프로세서를 로드하는 함수

    Returns:
        bool: 로딩 성공 시 True, 실패 시 False
    """
    global model, processor

    logger.info(" VARCO-VISION 모델 로딩 중...")

    try:
        # ===== 모델 로딩 =====
        model_name = "/home/owner/llm/varcollmmodel"

        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.float16,
            attn_implementation="sdpa",
            device_map="auto",
        )
        model.eval()  # 평가 모드로 설정

        # ===== 프로세서 로딩 =====
        processor = AutoProcessor.from_pretrained(model_name)

        logger.info("✅ VARCO-VISION 모델 로딩 완료!")
        return True

    except Exception as e:
        logger.error(f"❌ 모델 로딩 실패: {str(e)}")
        return False


# ===== 이미지 처리 함수 =====
def process_image(image_data):
    """
    다양한 형식의 이미지 데이터를 PIL Image 객체로 변환하는 함수

    Args:
        image_data: str (Base64) 또는 bytes 형식의 이미지 데이터

    Returns:
        PIL.Image: RGB 형식으로 변환된 이미지 객체

    Raises:
        HTTPException: 이미지 처리 실패 시 400 에러
    """
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


# ===== 동영상 프레임 추출 함수 =====
def extract_frames_from_video(video_data, max_frames: int = 10, fps: float = 1.0):
    """
    동영상에서 프레임을 추출하여 PIL Image 리스트로 반환하는 함수

    Args:
        video_data: bytes 형식의 동영상 데이터
        max_frames (int): 추출할 최대 프레임 수 (기본값: 10)
        fps (float): 초당 추출할 프레임 수 (기본값: 1.0)

    Returns:
        List[PIL.Image]: 추출된 프레임들의 PIL Image 리스트

    Raises:
        HTTPException: 동영상 처리 실패 시 400 에러
    """
    try:
        import tempfile

        # 임시 파일로 동영상 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_data)
            tmp_path = tmp_file.name

        # OpenCV로 동영상 열기
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise Exception("동영상 파일을 열 수 없습니다.")

        # 동영상 정보 가져오기
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / video_fps if video_fps > 0 else 0

        logger.info(f"동영상 정보 - FPS: {video_fps}, 전체 프레임: {total_frames}, 길이: {duration:.2f}초")

        # 프레임 추출 간격 계산
        frame_interval = int(video_fps / fps) if fps > 0 else 1
        frame_interval = max(1, frame_interval)

        frames = []
        frame_count = 0

        # 프레임 추출 루프
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)

                if len(frames) >= max_frames:
                    break

            frame_count += 1

        # 리소스 정리
        cap.release()
        os.remove(tmp_path)

        logger.info(f"동영상에서 {len(frames)}개의 프레임 추출 완료")

        if len(frames) == 0:
            raise Exception("동영상에서 프레임을 추출할 수 없습니다.")

        return frames

    except Exception as e:
        logger.error(f"동영상 처리 오류: {str(e)}")
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=400, detail=f"동영상 처리 실패: {str(e)}")


# ===== 응답 생성 함수 =====
def generate_response(text: str, image=None, images=None, max_tokens: int = 1024, temperature: float = 0.7):
    """
    텍스트와 이미지(들)을 기반으로 AI 응답을 생성하는 함수

    Args:
        text (str): 사용자 질문 텍스트
        image (PIL.Image, optional): 단일 이미지 객체
        images (List[PIL.Image], optional): 여러 이미지 객체 리스트
        max_tokens (int): 생성할 최대 토큰 수
        temperature (float): 생성 온도

    Returns:
        tuple: (응답 텍스트, 추론 시간)

    Raises:
        HTTPException: 모델 미로드 시 503, 추론 실패 시 500 에러
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")

    try:
        # 메시지 구성 (VARCO-VISION 형식)
        content = []

        # 이미지 추가
        if images is not None and len(images) > 0:
            for img in images:
                # VARCO-VISION은 URL 또는 이미지 객체를 받습니다
                # 여기서는 이미지 객체를 직접 사용
                content.append({"type": "image", "image": img})
        elif image is not None:
            content.append({"type": "image", "image": image})

        content.append({"type": "text", "text": text})

        conversation = [{"role": "user", "content": content}]

        # 입력 데이터 준비
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, torch.float16)

        # 추론 시작
        start_time = time.time()

        with torch.inference_mode():
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.9 if temperature > 0 else None,
                use_cache=True,
            )

        inference_time = time.time() - start_time

        # 결과 디코딩
        generate_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
        ]
        output_text = processor.decode(generate_ids_trimmed[0], skip_special_tokens=True)

        return output_text, inference_time

    except Exception as e:
        logger.error(f"추론 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"추론 실패: {str(e)}")


# ===== FastAPI 이벤트 핸들러 =====
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델을 로드하는 함수"""
    success = load_model()
    if not success:
        logger.warning("⚠️  서버가 시작되었지만 모델 로딩에 실패했습니다.")


# ===== 루트 엔드포인트 (웹 인터페이스) =====
@app.get("/", response_class=HTMLResponse)
async def root():
    """루트 경로에서 웹 인터페이스를 제공하는 엔드포인트"""
    html_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_file):
        with open(html_file, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return """
        <html>
            <body>
                <h1>VARCO-VISION Chat Server</h1>
                <p>서버가 실행 중입니다.</p>
                <p>API 문서: <a href="/docs">/docs</a></p>
            </body>
        </html>
        """


# ===== API 상태 엔드포인트 =====
@app.get("/api")
async def api_root():
    """API 서버 상태를 확인하는 엔드포인트"""
    return {
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "chat": "/chat",
            "chat_with_file": "/chat/upload",
            "health": "/health"
        }
    }


# ===== 헬스 체크 엔드포인트 =====
@app.get("/health")
async def health_check():
    """서버 및 모델 상태를 확인하는 헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }


# ===== 채팅 엔드포인트 (JSON 방식) =====
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    텍스트와 이미지(base64)를 받아 AI 응답을 생성하는 엔드포인트
    """
    try:
        image = None
        images = None

        # 멀티 이미지 처리
        if request.images_base64 and len(request.images_base64) > 0:
            images = []
            for img_base64 in request.images_base64:
                images.append(process_image(img_base64))
        # 단일 이미지 처리
        elif request.image_base64:
            image = process_image(request.image_base64)

        # AI 응답 생성
        response_text, inference_time = generate_response(
            text=request.text,
            image=image,
            images=images,
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


# ===== 채팅 엔드포인트 (파일 업로드 방식) =====
@app.post("/chat/upload", response_model=ChatResponse)
async def chat_with_upload(
    text: str = Form(...),
    image: Optional[UploadFile] = File(None),
    images: Optional[List[UploadFile]] = File(None),
    video: Optional[UploadFile] = File(None),
    max_tokens: int = Form(1024),
    temperature: float = Form(0.7),
    max_frames: int = Form(10),
    frame_fps: float = Form(1.0)
):
    """
    파일 업로드 방식으로 이미지(들), 동영상, 텍스트를 받아 AI 응답을 생성하는 엔드포인트
    """
    try:
        image_obj = None
        images_obj = None

        # 동영상 처리
        if video:
            logger.info(f"동영상 파일 수신: {video.filename}")
            video_contents = await video.read()
            images_obj = extract_frames_from_video(video_contents, max_frames=max_frames, fps=frame_fps)
            logger.info(f"동영상에서 {len(images_obj)}개의 프레임 추출 완료")

        # 멀티 이미지 처리
        elif images and len(images) > 0:
            images_obj = []
            for img_file in images:
                contents = await img_file.read()
                images_obj.append(process_image(contents))

        # 단일 이미지 처리
        elif image:
            contents = await image.read()
            image_obj = process_image(contents)

        # AI 응답 생성
        response_text, inference_time = generate_response(
            text=text,
            image=image_obj,
            images=images_obj,
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


# ===== 메인 실행 블록 =====
if __name__ == "__main__":
    uvicorn.run(
        "varcovision_server:app",
        host="0.0.0.0",
        port=24114,         # VARCO-VISION 전용 포트
        reload=False,
        log_level="info"
    )
