"""
Qwen3-VL Image & Text Chat Server
이미지와 텍스트를 받아 분석하는 챗봇 시스템
FastAPI 기반의 REST API 서버로 Qwen3-VL 비전 언어 모델을 서빙합니다.
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

from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
# JSONResponse: JSON 형식 응답
# HTMLResponse: HTML 형식 응답
# FileResponse: 파일 응답 (사용되지 않지만 임포트됨)

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
from transformers import AutoModelForImageTextToText, AutoProcessor
# AutoModelForImageTextToText: 이미지-텍스트 멀티모달 모델 자동 로딩
# AutoProcessor: 모델에 맞는 전처리기 자동 로딩


# ===== 로깅 설정 =====
logging.basicConfig(level=logging.INFO)  # 로깅 레벨을 INFO로 설정 (INFO, WARNING, ERROR 출력)
logger = logging.getLogger(__name__)     # 현재 모듈의 로거 객체 생성 (__name__은 'chat_server')


# ===== FastAPI 애플리케이션 초기화 =====
app = FastAPI(
    title="Qwen3-VL Chat Server",      # API 문서에 표시될 제목
    description="이미지와 텍스트를 분석하는 멀티모달 챗봇 API",  # API 설명
    version="1.0.0"                     # API 버전 정보
)


# ===== CORS 미들웨어 설정 =====
# 다른 도메인에서 이 API를 호출할 수 있도록 허용
app.add_middleware(
    CORSMiddleware,                     # CORS 미들웨어 추가
    allow_origins=["*"],                # 모든 출처(도메인) 허용 (보안상 프로덕션에서는 특정 도메인만 허용 권장)
    allow_credentials=True,             # 쿠키 및 인증 정보 허용
    allow_methods=["*"],                # 모든 HTTP 메서드 허용 (GET, POST, PUT, DELETE 등)
    allow_headers=["*"],                # 모든 HTTP 헤더 허용
)


# ===== 정적 파일 서빙 설정 =====
static_dir = os.path.join(os.path.dirname(__file__), "static")  # 현재 파일 위치에서 'static' 디렉토리 경로 생성
if os.path.exists(static_dir):                                   # static 디렉토리가 존재하는지 확인
    app.mount("/static", StaticFiles(directory=static_dir), name="static")  # /static 경로로 정적 파일 마운트


# ===== Pydantic 데이터 모델 정의 =====
# 클라이언트가 보내는 요청 데이터 구조
class ChatRequest(BaseModel):
    text: str                           # 사용자 질문 텍스트 (필수)
    image_base64: Optional[str] = None  # Base64 인코딩된 이미지 (선택, 기본값 None, 하위 호환성 유지)
    images_base64: Optional[List[str]] = None  # Base64 인코딩된 이미지 배열 (선택, 멀티 이미지용)
    max_tokens: Optional[int] = 256     # 생성할 최대 토큰 수 (선택, 기본값 256)
    temperature: Optional[float] = 0.7  # 생성 온도 (선택, 기본값 0.7, 높을수록 창의적)

# 서버가 반환하는 응답 데이터 구조
class ChatResponse(BaseModel):
    response: str                       # AI가 생성한 응답 텍스트
    inference_time: float               # 추론에 걸린 시간 (초)
    status: str                         # 요청 처리 상태 ("success" 등)


# ===== 전역 모델 변수 =====
# 서버 시작 시 한 번만 로드하여 메모리에 유지
model = None      # Qwen3-VL 모델 객체 (초기값 None)
processor = None  # 모델 전처리기 객체 (초기값 None)


# ===== 모델 로딩 함수 =====
def load_model():
    """
    Qwen3-VL 모델과 프로세서를 로드하는 함수

    Returns:
        bool: 로딩 성공 시 True, 실패 시 False
    """
    global model, processor  # 전역 변수 model, processor 사용 선언

    logger.info(" 모델 로딩 중...")  # 로그 출력: 모델 로딩 시작

    # ===== Hugging Face Hub 설정 =====
    os.environ["HF_HUB_OFFLINE"] = "1"      # 오프라인 모드 활성화 (로컬 파일만 사용)
    os.environ["HF_HUB_ENABLE_XET"] = "0"   # XET 기능 비활성화

    try:
        # ===== 모델 로딩 =====
        model = AutoModelForImageTextToText.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct",    # 모델 이름 (Hugging Face 모델 허브)
            dtype=torch.bfloat16,           # 데이터 타입: bfloat16 (메모리 절약, 속도 향상)
            device_map="auto",              # 자동으로 GPU/CPU 배치 결정
            local_files_only=True           # 로컬 캐시 파일만 사용 (다운로드 안 함)
        )
        model.eval()  # 평가 모드로 설정 (학습 모드 비활성화, 드롭아웃 등 끔)

        # ===== 프로세서 로딩 =====
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct",    # 모델과 같은 이름
            local_files_only=True           # 로컬 캐시 파일만 사용
        )

        logger.info("✅ 모델 로딩 완료!")  # 로그 출력: 로딩 성공
        return True  # 성공 반환

    except Exception as e:  # 예외 발생 시
        logger.error(f"❌ 모델 로딩 실패: {str(e)}")  # 에러 로그 출력
        return False  # 실패 반환


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
        # ===== 입력 데이터 타입별 처리 =====
        if isinstance(image_data, str):  # 문자열(Base64) 형식인 경우
            # Base64 디코딩
            if image_data.startswith('data:image'):  # Data URL 형식인 경우 (예: "data:image/png;base64,...")
                image_data = image_data.split(',')[1]  # 헤더 제거하고 실제 Base64 데이터만 추출
            image_bytes = base64.b64decode(image_data)  # Base64를 바이트로 디코딩
            image = Image.open(io.BytesIO(image_bytes))  # 바이트를 PIL Image로 변환

        elif isinstance(image_data, bytes):  # 바이트 형식인 경우
            image = Image.open(io.BytesIO(image_data))  # 바이트를 PIL Image로 변환

        else:  # 이미 PIL Image 객체인 경우
            image = image_data  # 그대로 사용

        # ===== RGB 변환 =====
        if image.mode != 'RGB':  # 이미지가 RGB가 아닌 경우 (RGBA, L 등)
            image = image.convert('RGB')  # RGB로 변환 (모델이 RGB만 받음)

        return image  # 변환된 이미지 반환

    except Exception as e:  # 예외 발생 시
        logger.error(f"이미지 처리 오류: {str(e)}")  # 에러 로그 출력
        raise HTTPException(status_code=400, detail=f"이미지 처리 실패: {str(e)}")  # 400 Bad Request 에러 발생


# ===== 동영상 프레임 추출 함수 =====
def extract_frames_from_video(video_data, max_frames: int = 10, fps: float = 1.0):
    """
    동영상에서 프레임을 추출하여 PIL Image 리스트로 반환하는 함수

    Args:
        video_data: bytes 형식의 동영상 데이터
        max_frames (int): 추출할 최대 프레임 수 (기본값: 10)
        fps (float): 초당 추출할 프레임 수 (기본값: 1.0, 1초당 1프레임)

    Returns:
        List[PIL.Image]: 추출된 프레임들의 PIL Image 리스트

    Raises:
        HTTPException: 동영상 처리 실패 시 400 에러
    """
    try:
        # ===== 임시 파일로 동영상 저장 =====
        import tempfile
        # 임시 파일 생성 (자동 삭제되지 않도록 delete=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_data)  # 동영상 데이터 쓰기
            tmp_path = tmp_file.name  # 임시 파일 경로 저장

        # ===== OpenCV로 동영상 열기 =====
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():  # 동영상 열기 실패
            raise Exception("동영상 파일을 열 수 없습니다.")

        # 동영상 정보 가져오기
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 전체 프레임 수
        video_fps = cap.get(cv2.CAP_PROP_FPS)  # 동영상 FPS
        duration = total_frames / video_fps if video_fps > 0 else 0  # 동영상 길이(초)

        logger.info(f"동영상 정보 - FPS: {video_fps}, 전체 프레임: {total_frames}, 길이: {duration:.2f}초")

        # ===== 프레임 추출 간격 계산 =====
        # fps 파라미터를 기반으로 몇 프레임마다 추출할지 계산
        frame_interval = int(video_fps / fps) if fps > 0 else 1
        frame_interval = max(1, frame_interval)  # 최소 1 프레임

        frames = []  # 추출된 프레임 리스트
        frame_count = 0  # 현재 프레임 카운터

        # ===== 프레임 추출 루프 =====
        while True:
            ret, frame = cap.read()  # 프레임 읽기
            if not ret:  # 프레임 읽기 실패 (동영상 끝)
                break

            # 지정된 간격마다 프레임 추출
            if frame_count % frame_interval == 0:
                # OpenCV는 BGR 형식이므로 RGB로 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # NumPy 배열을 PIL Image로 변환
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)

                # 최대 프레임 수에 도달하면 중단
                if len(frames) >= max_frames:
                    break

            frame_count += 1

        # ===== 리소스 정리 =====
        cap.release()  # VideoCapture 객체 해제
        os.remove(tmp_path)  # 임시 파일 삭제

        logger.info(f"동영상에서 {len(frames)}개의 프레임 추출 완료")

        if len(frames) == 0:  # 프레임이 하나도 추출되지 않음
            raise Exception("동영상에서 프레임을 추출할 수 없습니다.")

        return frames  # 추출된 프레임 리스트 반환

    except Exception as e:  # 예외 발생 시
        logger.error(f"동영상 처리 오류: {str(e)}")  # 에러 로그 출력
        # 임시 파일이 있으면 삭제
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=400, detail=f"동영상 처리 실패: {str(e)}")  # 400 Bad Request 에러 발생


# ===== 응답 생성 함수 =====
def generate_response(text: str, image=None, images=None, max_tokens: int = 256, temperature: float = 0.7):
    """
    텍스트와 이미지(들)을 기반으로 AI 응답을 생성하는 함수

    Args:
        text (str): 사용자 질문 텍스트
        image (PIL.Image, optional): 단일 이미지 객체 (없으면 None, 하위 호환성)
        images (List[PIL.Image], optional): 여러 이미지 객체 리스트 (없으면 None, 멀티 이미지용)
        max_tokens (int): 생성할 최대 토큰 수 (기본값 256)
        temperature (float): 생성 온도 (0~1, 높을수록 창의적, 기본값 0.7)

    Returns:
        tuple: (응답 텍스트, 추론 시간)

    Raises:
        HTTPException: 모델 미로드 시 503, 추론 실패 시 500 에러
    """
    # ===== 모델 로드 확인 =====
    if model is None or processor is None:  # 모델이나 프로세서가 로드되지 않은 경우
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")  # 503 Service Unavailable

    try:
        # ===== 메시지 구성 =====
        content = []  # 콘텐츠 리스트 초기화

        # ===== 이미지 추가 (멀티 이미지 우선, 단일 이미지는 하위 호환성) =====
        if images is not None and len(images) > 0:  # 멀티 이미지가 있는 경우
            for img in images:  # 각 이미지를 순회하며
                content.append({"type": "image", "image": img})  # content에 이미지 추가
        elif image is not None:  # 단일 이미지가 있는 경우 (하위 호환성)
            content.append({"type": "image", "image": image})  # 이미지 추가

        content.append({"type": "text", "text": text})  # 텍스트 추가

        messages = [{"role": "user", "content": content}]  # 사용자 메시지 형식으로 구성

        # ===== 입력 데이터 준비 =====
        inputs = processor.apply_chat_template(
            messages,                    # 구성한 메시지
            tokenize=True,               # 토큰화 수행
            add_generation_prompt=True,  # 생성 프롬프트 추가
            return_dict=True,            # 딕셔너리 형태로 반환
            return_tensors="pt"          # PyTorch 텐서로 반환
        )
        inputs = inputs.to(model.device)  # 모델이 있는 디바이스(GPU/CPU)로 이동

        # ===== 추론 시작 =====
        start_time = time.time()  # 시작 시간 기록

        with torch.inference_mode():  # 추론 모드 (그래디언트 계산 안 함, 속도 향상)
            generated_ids = model.generate(
                **inputs,                                        # 입력 텐서들 (input_ids, attention_mask 등)
                max_new_tokens=max_tokens,                       # 생성할 최대 토큰 수
                do_sample=temperature > 0,                       # 샘플링 여부 (temperature > 0이면 True)
                temperature=temperature if temperature > 0 else None,  # 생성 온도 (0이면 greedy decoding)
                top_p=0.9 if temperature > 0 else None,          # nucleus sampling (상위 90% 확률 토큰만 사용)
                use_cache=True,                                  # KV 캐시 사용 (속도 향상)
                pad_token_id=processor.tokenizer.pad_token_id,   # 패딩 토큰 ID
            )

        inference_time = time.time() - start_time  # 추론 시간 계산 (종료 시간 - 시작 시간)

        # ===== 결과 디코딩 =====
        generated_ids_trimmed = [
            out_ids[len(in_ids):]  # 입력 부분 제거하고 생성된 부분만 추출
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,              # 생성된 토큰 ID들
            skip_special_tokens=True,           # 특수 토큰(<s>, </s> 등) 제거
            clean_up_tokenization_spaces=False  # 토큰화 공백 정리 안 함
        )[0]  # 배치의 첫 번째 결과 (배치 크기 1이므로)

        return output_text, inference_time  # 응답 텍스트와 추론 시간 반환

    except Exception as e:  # 예외 발생 시
        logger.error(f"추론 오류: {str(e)}")  # 에러 로그 출력
        raise HTTPException(status_code=500, detail=f"추론 실패: {str(e)}")  # 500 Internal Server Error


# ===== FastAPI 이벤트 핸들러 =====
@app.on_event("startup")  # 서버 시작 시 실행되는 이벤트 핸들러
async def startup_event():
    """
    서버 시작 시 모델을 로드하는 함수
    비동기 함수로 정의되어 서버 시작 시 자동 실행됨
    """
    success = load_model()  # 모델 로딩 시도
    if not success:  # 로딩 실패 시
        logger.warning("⚠️  서버가 시작되었지만 모델 로딩에 실패했습니다.")  # 경고 로그 출력


# ===== 루트 엔드포인트 (웹 인터페이스) =====
@app.get("/", response_class=HTMLResponse)  # GET 메서드, HTML 응답 반환
async def root():
    """
    루트 경로에서 웹 인터페이스를 제공하는 엔드포인트
    static/index.html 파일이 있으면 반환, 없으면 기본 HTML 반환

    Returns:
        HTMLResponse: HTML 콘텐츠
    """
    html_file = os.path.join(os.path.dirname(__file__), "static", "index.html")  # index.html 경로 생성
    if os.path.exists(html_file):  # 파일이 존재하는지 확인
        with open(html_file, 'r', encoding='utf-8') as f:  # 파일 열기 (UTF-8 인코딩)
            return f.read()  # 파일 내용 읽어서 반환
    else:  # 파일이 없는 경우
        return """
        <html>
            <body>
                <h1>Qwen3-VL Chat Server</h1>
                <p>서버가 실행 중입니다.</p>
                <p>API 문서: <a href="/docs">/docs</a></p>
            </body>
        </html>
        """  # 기본 HTML 반환


# ===== API 상태 엔드포인트 =====
@app.get("/api")  # GET 메서드, /api 경로
async def api_root():
    """
    API 서버 상태를 확인하는 엔드포인트

    Returns:
        dict: 서버 상태 정보
    """
    return {
        "status": "running",              # 서버 실행 상태
        "model_loaded": model is not None,  # 모델 로드 여부 (True/False)
        "endpoints": {                    # 사용 가능한 엔드포인트 목록
            "chat": "/chat",              # JSON 방식 채팅 엔드포인트
            "chat_with_file": "/chat/upload",  # 파일 업로드 방식 엔드포인트
            "health": "/health"           # 헬스 체크 엔드포인트
        }
    }


# ===== 헬스 체크 엔드포인트 =====
@app.get("/health")  # GET 메서드, /health 경로
async def health_check():
    """
    서버 및 모델 상태를 확인하는 헬스 체크 엔드포인트
    로드 밸런서나 모니터링 도구에서 서버 상태 확인용

    Returns:
        dict: 서버 헬스 정보
    """
    return {
        "status": "healthy",                # 서버 건강 상태
        "model_loaded": model is not None,  # 모델 로드 여부
        "gpu_available": torch.cuda.is_available(),  # GPU 사용 가능 여부
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0  # 사용 가능한 GPU 개수
    }


# ===== 채팅 엔드포인트 (JSON 방식) =====
@app.post("/chat", response_model=ChatResponse)  # POST 메서드, 응답 모델 지정
async def chat(request: ChatRequest):
    """
    텍스트와 이미지(base64)를 받아 AI 응답을 생성하는 엔드포인트
    JSON 형식으로 데이터를 주고받음

    Args:
        request (ChatRequest): 요청 데이터 (자동으로 검증 및 파싱됨)

    Returns:
        ChatResponse: AI 응답 데이터

    Raises:
        HTTPException: 처리 중 오류 발생 시

    API 문서:
        - **text**: 사용자 질문 (필수)
        - **image_base64**: Base64 인코딩된 이미지 (선택)
        - **max_tokens**: 최대 생성 토큰 수 (기본: 256)
        - **temperature**: 생성 온도 (기본: 0.7, 0이면 greedy decoding)
    """
    try:
        # ===== 이미지 처리 =====
        image = None  # 단일 이미지 초기값 None (하위 호환성)
        images = None  # 멀티 이미지 초기값 None

        # 멀티 이미지가 있는 경우 우선 처리
        if request.images_base64 and len(request.images_base64) > 0:
            images = []  # 이미지 리스트 초기화
            for img_base64 in request.images_base64:  # 각 Base64 이미지를 순회
                images.append(process_image(img_base64))  # Base64를 PIL Image로 변환하여 리스트에 추가
        # 단일 이미지가 있는 경우 (하위 호환성)
        elif request.image_base64:
            image = process_image(request.image_base64)  # Base64를 PIL Image로 변환

        # ===== AI 응답 생성 =====
        response_text, inference_time = generate_response(
            text=request.text,              # 사용자 질문
            image=image,                    # 단일 이미지 (있으면 PIL Image, 없으면 None)
            images=images,                  # 멀티 이미지 (있으면 List[PIL Image], 없으면 None)
            max_tokens=request.max_tokens,  # 최대 토큰 수
            temperature=request.temperature # 생성 온도
        )

        # ===== 응답 반환 =====
        return ChatResponse(
            response=response_text,                # AI가 생성한 텍스트
            inference_time=round(inference_time, 2),  # 추론 시간 (소수점 2자리로 반올림)
            status="success"                       # 성공 상태
        )

    except HTTPException:  # HTTPException은 그대로 전파
        raise
    except Exception as e:  # 기타 예외 발생 시
        logger.error(f"Chat 오류: {str(e)}")  # 에러 로그 출력
        raise HTTPException(status_code=500, detail=str(e))  # 500 에러로 변환


# ===== 채팅 엔드포인트 (파일 업로드 방식) =====
@app.post("/chat/upload", response_model=ChatResponse)  # POST 메서드, 응답 모델 지정
async def chat_with_upload(
    text: str = Form(...),                      # 폼 데이터: 텍스트 (필수, ...는 필수 의미)
    image: Optional[UploadFile] = File(None),   # 파일: 단일 이미지 (선택, 기본값 None, 하위 호환성)
    images: Optional[List[UploadFile]] = File(None),  # 파일: 여러 이미지 (선택, 멀티 이미지용)
    video: Optional[UploadFile] = File(None),   # 파일: 동영상 (선택, 기본값 None, 프레임 추출)
    max_tokens: int = Form(256),                # 폼 데이터: 최대 토큰 수 (기본값 256)
    temperature: float = Form(0.7),             # 폼 데이터: 생성 온도 (기본값 0.7)
    max_frames: int = Form(10),                 # 동영상에서 추출할 최대 프레임 수 (기본값 10)
    frame_fps: float = Form(1.0)                # 초당 추출할 프레임 수 (기본값 1.0)
):
    """
    파일 업로드 방식으로 이미지(들), 동영상, 텍스트를 받아 AI 응답을 생성하는 엔드포인트
    multipart/form-data 형식으로 데이터를 받음 (HTML form이나 curl -F 사용)

    Args:
        text (str): 사용자 질문 (필수)
        image (UploadFile, optional): 단일 이미지 파일 (선택, 하위 호환성)
        images (List[UploadFile], optional): 여러 이미지 파일 (선택, 멀티 이미지용)
        video (UploadFile, optional): 동영상 파일 (선택, 프레임 자동 추출)
        max_tokens (int): 최대 생성 토큰 수 (기본값 256)
        temperature (float): 생성 온도 (기본값 0.7)
        max_frames (int): 동영상에서 추출할 최대 프레임 수 (기본값 10)
        frame_fps (float): 초당 추출할 프레임 수 (기본값 1.0)

    Returns:
        ChatResponse: AI 응답 데이터

    Raises:
        HTTPException: 처리 중 오류 발생 시

    API 문서:
        - **text**: 사용자 질문 (필수)
        - **image**: 단일 이미지 파일 (선택, 하위 호환성)
        - **images**: 여러 이미지 파일 (선택, 멀티 이미지용)
        - **video**: 동영상 파일 (선택, 프레임 자동 추출)
        - **max_tokens**: 최대 생성 토큰 수 (기본: 256)
        - **temperature**: 생성 온도 (기본: 0.7)
        - **max_frames**: 동영상 최대 프레임 수 (기본: 10)
        - **frame_fps**: 프레임 추출 속도 (기본: 1.0fps)
    """
    try:
        # ===== 이미지/동영상 처리 =====
        image_obj = None  # 단일 이미지 객체 초기값 None (하위 호환성)
        images_obj = None  # 멀티 이미지 객체 리스트 초기값 None

        # 동영상이 있는 경우 프레임 추출 (최우선)
        if video:
            logger.info(f"동영상 파일 수신: {video.filename}")
            video_contents = await video.read()  # 동영상 파일 내용 읽기
            # 동영상에서 프레임 추출
            images_obj = extract_frames_from_video(video_contents, max_frames=max_frames, fps=frame_fps)
            logger.info(f"동영상에서 {len(images_obj)}개의 프레임 추출 완료")

        # 멀티 이미지가 있는 경우 처리 (동영상이 없을 때)
        elif images and len(images) > 0:
            images_obj = []  # 이미지 리스트 초기화
            for img_file in images:  # 각 업로드된 파일을 순회
                contents = await img_file.read()  # 파일 내용을 비동기로 읽기 (바이트 데이터)
                images_obj.append(process_image(contents))  # 바이트를 PIL Image로 변환하여 리스트에 추가

        # 단일 이미지가 있는 경우 (하위 호환성, 동영상과 멀티 이미지가 없을 때)
        elif image:
            contents = await image.read()  # 파일 내용을 비동기로 읽기 (바이트 데이터)
            image_obj = process_image(contents)  # 바이트를 PIL Image로 변환

        # ===== AI 응답 생성 =====
        response_text, inference_time = generate_response(
            text=text,              # 사용자 질문
            image=image_obj,        # 단일 이미지 (있으면 PIL Image, 없으면 None)
            images=images_obj,      # 멀티 이미지 (있으면 List[PIL Image], 없으면 None)
            max_tokens=max_tokens,  # 최대 토큰 수
            temperature=temperature # 생성 온도
        )

        # ===== 응답 반환 =====
        return ChatResponse(
            response=response_text,                # AI가 생성한 텍스트
            inference_time=round(inference_time, 2),  # 추론 시간 (소수점 2자리)
            status="success"                       # 성공 상태
        )

    except HTTPException:  # HTTPException은 그대로 전파
        raise
    except Exception as e:  # 기타 예외 발생 시
        logger.error(f"Chat with upload 오류: {str(e)}")  # 에러 로그 출력
        raise HTTPException(status_code=500, detail=str(e))  # 500 에러로 변환


# ===== 메인 실행 블록 =====
if __name__ == "__main__":
    """
    이 파일을 직접 실행했을 때만 실행되는 코드
    'python chat_server.py' 명령으로 실행 시 동작
    다른 파일에서 임포트하면 실행 안 됨
    """
    # ===== Uvicorn 서버 실행 =====
    uvicorn.run(
        "chat_server:app",  # 실행할 앱 (모듈명:변수명)
        host="0.0.0.0",     # 모든 네트워크 인터페이스에서 접속 허용 (외부 접속 가능)
        port=24112,         # 포트 번호 24112번 사용
        reload=False,       # 코드 변경 시 자동 재시작 비활성화 (프로덕션 설정)
        log_level="info"    # 로그 레벨 INFO (INFO, WARNING, ERROR 출력)
    )
