# FastAPI: 현대적인 고성능 웹 프레임워크의 핵심 클래스들을 임포트
# - FastAPI: 메인 애플리케이션 객체를 생성하는 클래스
# - WebSocket: 실시간 양방향 통신을 위한 WebSocket 연결 객체
# - WebSocketDisconnect: WebSocket 연결이 끊어졌을 때 발생하는 예외 처리용
# - HTTPException: HTTP 에러 응답을 생성하기 위한 예외 클래스
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException

# CORS(Cross-Origin Resource Sharing) 미들웨어 임포트
# 웹 브라우저에서 다른 도메인의 API를 호출할 수 있도록 허용하는 보안 정책 설정용
from fastapi.middleware.cors import CORSMiddleware

# Pydantic의 BaseModel 클래스 임포트
# 데이터 검증, 직렬화, 타입 힌팅을 자동으로 처리하는 데이터 모델의 기본 클래스
from pydantic import BaseModel

# OpenAI의 비동기 클라이언트 라이브러리 임포트
# VLLM 서버와 비동기적으로 통신하여 여러 요청을 동시에 처리 가능
from openai import AsyncOpenAI

# Python의 타입 힌팅을 위한 제네릭 타입 임포트
# List: 리스트 타입 명시, Dict: 딕셔너리 타입 명시 (코드 가독성과 IDE 자동완성 향상)
from typing import List, Dict

# UUID(Universally Unique Identifier) 버전 4 생성 함수 임포트
# 각 사용자 세션에 전 세계적으로 고유한 식별자를 부여하기 위해 사용
from uuid import uuid4

# Python 표준 라이브러리의 datetime 클래스 임포트
# 세션 생성 시간, 메시지 타임스탬프 등 시간 정보를 기록하고 처리
from datetime import datetime

# asyncio: Python의 비동기 I/O 처리를 위한 표준 라이브러리
# 여러 작업을 동시에 처리할 수 있는 이벤트 루프 기반 비동기 프로그래밍 지원
import asyncio

# json: JSON(JavaScript Object Notation) 데이터의 인코딩/디코딩을 위한 표준 라이브러리
# WebSocket으로 주고받는 메시지를 JSON 형식으로 파싱하고 생성
import json

# FastAPI 애플리케이션 인스턴스 생성
# title: API 문서에 표시될 애플리케이션 이름
# version: API 버전 정보 (문서 및 클라이언트에서 참조)
app = FastAPI(title="Multi-User Chatbot System", version="1.0.0")

# CORS 미들웨어를 FastAPI 애플리케이션에 추가
# 미들웨어: 요청과 응답 사이에서 동작하는 중간 처리 계층
app.add_middleware(
    # CORSMiddleware 클래스를 미들웨어로 등록
    CORSMiddleware,
    # allow_origins: 허용할 출처(도메인) 목록
    # ["*"]는 모든 도메인에서의 요청을 허용 (개발 환경용, 프로덕션에서는 특정 도메인만 허용 권장)
    allow_origins=["*"],
    # allow_credentials: 쿠키, 인증 헤더 등 인증 정보를 포함한 요청 허용 여부
    # True로 설정하면 클라이언트가 쿠키를 서버에 전송할 수 있음
    allow_credentials=True,
    # allow_methods: 허용할 HTTP 메서드 목록
    # ["*"]는 GET, POST, PUT, DELETE 등 모든 HTTP 메서드 허용
    allow_methods=["*"],
    # allow_headers: 허용할 HTTP 헤더 목록
    # ["*"]는 모든 커스텀 헤더를 포함한 모든 헤더 허용
    allow_headers=["*"],
)

# AsyncOpenAI 클라이언트 인스턴스 생성
# 로컬에서 실행 중인 VLLM 서버와 OpenAI API 호환 방식으로 통신
client = AsyncOpenAI(
    # base_url: VLLM 서버의 API 엔드포인트 주소
    # localhost:8000은 로컬 컴퓨터의 8000번 포트에서 실행 중인 VLLM 서버를 의미
    # /v1은 OpenAI API v1 호환 엔드포인트
    base_url="http://localhost:8000/v1",
    # api_key: API 인증 키
    # 로컬 VLLM 서버는 인증이 필요 없으므로 "EMPTY" 문자열 사용
    api_key="EMPTY"
)

# sessions: 각 사용자의 대화 세션을 메모리에 저장하는 전역 딕셔너리
# 키(str): 세션 ID (UUID 문자열)
# 값(List[Dict[str, str]]): 해당 세션의 대화 메시지 리스트
#   각 메시지는 {"role": "user/assistant/system", "content": "메시지 내용"} 형태
sessions: Dict[str, List[Dict[str, str]]] = {}

# active_connections: 현재 활성화된 WebSocket 연결을 관리하는 전역 딕셔너리
# 키(str): 세션 ID
# 값(WebSocket): 해당 세션의 WebSocket 연결 객체 (실시간 메시지 전송용)
active_connections: Dict[str, WebSocket] = {}


# 현재 날짜와 시간 정보를 반환하는 함수 (AI가 호출할 수 있는 도구)
def get_current_time() -> str:
    """
    현재 날짜, 시간, 요일 정보를 반환하는 함수
    AI가 사용자가 시간을 물어보면 이 함수를 호출하여 정확한 시간을 얻을 수 있음
    """
    # datetime.now()로 현재 시스템 시간 가져오기
    now = datetime.now()
    # 날짜와 시간을 "YYYY-MM-DD HH:MM:SS" 형식으로 포맷팅
    # %Y: 4자리 연도, %m: 2자리 월, %d: 2자리 일
    # %H: 24시간 형식 시간, %M: 분, %S: 초
    datetime_str = now.strftime("%Y-%m-%d %H:%M:%S")
    # 요일을 영어 전체 이름으로 가져오기 (예: Monday, Tuesday)
    day_of_week_en = now.strftime("%A")
    # 영어 요일명을 한국어로 매핑하는 딕셔너리
    days_kr = {
        "Monday": "월요일",
        "Tuesday": "화요일",
        "Wednesday": "수요일",
        "Thursday": "목요일",
        "Friday": "금요일",
        "Saturday": "토요일",
        "Sunday": "일요일"
    }
    # 한국어 요일명 가져오기
    day_of_week_kr = days_kr[day_of_week_en]
    # 현재 날짜, 시간, 요일 정보를 포함한 문자열 반환
    return f"현재 시간: {datetime_str} ({day_of_week_kr})"


# OpenAI Function Calling을 위한 도구(Tool) 정의
# AI가 현재 시간을 알아야 할 때 호출할 수 있는 함수 스키마
tools = [
    {
        # 도구의 타입 (function: 함수 호출)
        "type": "function",
        # 함수에 대한 상세 정보
        "function": {
            # 함수 이름 (AI가 호출할 때 사용하는 식별자)
            "name": "get_current_time",
            # 함수 설명 (AI가 언제 이 함수를 호출해야 하는지 판단하는 기준)
            "description": "현재 날짜, 시간, 요일 정보를 가져옵니다. 사용자가 현재 시간, 날짜, 요일을 물어볼 때 이 함수를 호출하세요.",
            # 함수 파라미터 스키마 (이 함수는 파라미터가 없음)
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


# ChatRequest: 클라이언트가 채팅 요청 시 전송하는 데이터의 구조를 정의하는 Pydantic 모델
# Pydantic은 자동으로 데이터 타입 검증, JSON 직렬화/역직렬화를 처리
class ChatRequest(BaseModel):
    # session_id: 기존 세션을 이어가기 위한 세션 식별자
    # str = None: 선택 사항이며 기본값은 None (새 세션 생성)
    session_id: str = None

    # message: 사용자가 입력한 채팅 메시지
    # str 타입이며 필수 필드 (기본값 없음)
    message: str

    # system_prompt: AI의 역할과 행동을 정의하는 시스템 프롬프트
    # 기본값으로 "당신은 유능한 상담 전문가입니다." 제공
    system_prompt: str = "당신은 유능한 상담 전문가입니다."


# ChatResponse: 서버가 클라이언트에게 응답할 때 사용하는 데이터 구조 정의
# response_model로 지정하면 FastAPI가 자동으로 응답을 이 형식으로 직렬화
class ChatResponse(BaseModel):
    # session_id: 현재 대화의 세션 식별자 (클라이언트가 다음 요청에 사용)
    session_id: str

    # response: AI가 생성한 응답 메시지 텍스트
    response: str

    # conversation_history: 지금까지의 전체 대화 내역
    # 클라이언트가 대화 컨텍스트를 확인하거나 UI에 표시할 수 있음
    conversation_history: List[Dict[str, str]]


# 루트 경로("/")에 대한 GET 요청 핸들러 정의
# @app.get("/"): FastAPI 데코레이터로 GET 메서드와 경로를 바인딩
@app.get("/")
# async def: 비동기 함수 정의 (다른 요청과 동시에 처리 가능)
async def root():
    # 딕셔너리 형태로 JSON 응답 반환
    # FastAPI가 자동으로 JSON으로 직렬화하여 클라이언트에 전송
    return {
        # status: 서버의 현재 상태 (항상 "running")
        "status": "running",
        # message: API 서버의 설명 메시지
        "message": "Multi-User Chatbot API Server",
        # active_sessions: 현재 메모리에 저장된 세션의 총 개수
        "active_sessions": len(sessions),
        # active_connections: 현재 WebSocket으로 연결된 클라이언트 수
        "active_connections": len(active_connections)
    }


# "/session/create" 경로에 대한 POST 요청 핸들러
# 새로운 채팅 세션을 생성하고 고유 ID를 반환
@app.post("/session/create")
# 비동기 함수로 정의하여 여러 세션 생성 요청을 동시에 처리 가능
async def create_session():
    # uuid4() 함수로 랜덤 UUID(버전 4) 생성 후 문자열로 변환
    # 예: "550e8400-e29b-41d4-a716-446655440000"
    session_id = str(uuid4())

    # sessions 딕셔너리에 새 세션 추가
    # 초기값은 빈 리스트 (아직 메시지가 없음)
    sessions[session_id] = []

    # 클라이언트에게 반환할 응답 데이터 생성
    return {
        # session_id: 생성된 세션의 고유 식별자
        "session_id": session_id,
        # created_at: 세션 생성 시각을 ISO 8601 형식 문자열로 변환
        # 예: "2024-11-26T10:30:00.123456"
        "created_at": datetime.now().isoformat(),
        # message: 성공 메시지
        "message": "Session created successfully"
    }


# "/session/{session_id}/history" 경로에 대한 GET 요청 핸들러
# {session_id}는 경로 파라미터로 URL에서 동적으로 받아옴
@app.get("/session/{session_id}/history")
# session_id: str - 경로 파라미터로 전달된 세션 ID
async def get_session_history(session_id: str):
    # 요청한 세션 ID가 sessions 딕셔너리에 존재하지 않는 경우
    if session_id not in sessions:
        # HTTPException을 발생시켜 404 Not Found 에러 응답
        # detail: 에러 상세 메시지
        raise HTTPException(status_code=404, detail="Session not found")

    # 세션이 존재하면 해당 세션의 정보를 반환
    return {
        # session_id: 요청한 세션 ID (확인용)
        "session_id": session_id,
        # history: 해당 세션의 전체 대화 메시지 리스트
        "history": sessions[session_id],
        # message_count: 대화 메시지의 총 개수
        "message_count": len(sessions[session_id])
    }


# "/session/{session_id}" 경로에 대한 DELETE 요청 핸들러
# 특정 세션과 관련된 모든 데이터를 삭제
@app.delete("/session/{session_id}")
# session_id: 삭제할 세션의 ID (경로 파라미터)
async def delete_session(session_id: str):
    # 세션이 존재하지 않으면 404 에러 반환
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # sessions 딕셔너리에서 해당 세션의 대화 히스토리 삭제
    del sessions[session_id]

    # 해당 세션의 WebSocket 연결이 활성화되어 있는지 확인
    if session_id in active_connections:
        # 활성 연결 목록에서도 제거 (메모리 누수 방지)
        del active_connections[session_id]

    # 삭제 성공 메시지 반환
    return {"message": "Session deleted successfully"}


# "/chat" 경로에 대한 POST 요청 핸들러 (REST API 방식 채팅)
# response_model=ChatResponse: 응답이 ChatResponse 모델 형식임을 명시
@app.post("/chat", response_model=ChatResponse)
# request: ChatRequest - 요청 본문이 ChatRequest 모델로 자동 파싱됨
async def chat(request: ChatRequest):
    # 세션 ID가 제공되지 않았거나 존재하지 않는 세션인 경우
    if not request.session_id or request.session_id not in sessions:
        # 새로운 UUID 생성하여 세션 ID로 사용
        session_id = str(uuid4())

        # 새 세션 생성 시 시스템 프롬프트를 첫 메시지로 추가
        # Function calling을 사용하므로 시스템 프롬프트는 간단하게 유지
        sessions[session_id] = [
            {"role": "system", "content": request.system_prompt}
        ]
    # 기존 세션이 존재하는 경우
    else:
        # 요청에서 제공된 세션 ID를 그대로 사용
        session_id = request.session_id

    # 사용자가 보낸 메시지를 대화 히스토리에 추가
    # role: "user" - 메시지 발신자가 사용자임을 표시
    # content: 실제 메시지 내용
    sessions[session_id].append({
        "role": "user",
        "content": request.message
    })

    # try 블록: VLLM 서버 통신 중 발생할 수 있는 예외 처리
    try:
        # VLLM 서버에 비동기 채팅 완성 요청 전송 (tools 파라미터 포함)
        # await: 응답이 올 때까지 다른 작업을 처리할 수 있도록 대기
        response = await client.chat.completions.create(
            # model: 사용할 언어 모델의 이름 (VLLM 서버에 등록된 모델)
            model="openai/gpt-oss-20b",
            # messages: 현재까지의 전체 대화 히스토리 전송
            messages=sessions[session_id],
            # tools: AI가 사용할 수 있는 함수 목록 (현재 시간 조회 함수)
            tools=tools,
            # tool_choice: AI가 필요할 때 자동으로 도구를 선택하도록 설정
            tool_choice="auto",
            # max_tokens: AI가 생성할 최대 토큰(단어 조각) 수
            max_tokens=512,
            # temperature: 샘플링 온도 (0~2 사이)
            # 0.7은 적당히 창의적이면서도 일관성 있는 응답 생성
            temperature=0.7
        )

        # AI의 응답 메시지 객체 가져오기
        response_message = response.choices[0].message

        # AI가 도구(함수)를 호출했는지 확인
        # tool_calls가 있으면 AI가 함수를 호출하고 싶어 함
        if response_message.tool_calls:
            # AI의 응답(함수 호출 요청)을 대화 히스토리에 추가
            sessions[session_id].append({
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in response_message.tool_calls
                ]
            })

            # 각 함수 호출을 처리
            for tool_call in response_message.tool_calls:
                # 함수 이름 가져오기
                function_name = tool_call.function.name

                # 현재는 get_current_time 함수만 지원
                if function_name == "get_current_time":
                    # 실제 함수를 호출하여 현재 시간 가져오기
                    function_response = get_current_time()

                    # 함수 실행 결과를 대화 히스토리에 추가
                    # role: "tool" - 도구(함수) 실행 결과임을 표시
                    sessions[session_id].append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": function_response
                    })

            # 함수 실행 결과를 포함하여 AI에게 다시 요청
            # AI가 함수 실행 결과를 바탕으로 최종 응답 생성
            second_response = await client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=sessions[session_id],
                max_tokens=512,
                temperature=0.7
            )

            # 최종 AI 응답 메시지 추출
            ai_message = second_response.choices[0].message.content
        else:
            # 함수 호출 없이 바로 응답한 경우
            ai_message = response_message.content

        # AI의 최종 응답을 대화 히스토리에 추가 (다음 대화의 컨텍스트로 사용)
        # role: "assistant" - AI 어시스턴트가 보낸 메시지임을 표시
        sessions[session_id].append({
            "role": "assistant",
            "content": ai_message
        })

        # ChatResponse 모델 인스턴스를 생성하여 반환
        # FastAPI가 자동으로 JSON으로 직렬화
        return ChatResponse(
            # session_id: 클라이언트가 다음 요청에 사용할 세션 ID
            session_id=session_id,
            # response: AI의 응답 메시지
            response=ai_message,
            # conversation_history: 전체 대화 히스토리 (시스템, 사용자, AI 메시지 포함)
            conversation_history=sessions[session_id]
        )

    # VLLM 서버 통신 중 예외 발생 시 처리
    except Exception as e:
        # 500 Internal Server Error 응답 반환
        # detail: 에러의 상세 내용을 문자열로 포함
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


# WebSocket 엔드포인트 정의 (실시간 양방향 통신용)
# "/ws/{session_id}" 경로로 WebSocket 연결 수립
@app.websocket("/ws/{session_id}")
# websocket: WebSocket 연결 객체
# session_id: URL 경로에서 추출한 세션 ID
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    # 클라이언트의 WebSocket 연결 요청을 수락
    # 이 시점부터 양방향 실시간 통신 가능
    await websocket.accept()

    # 해당 세션이 아직 생성되지 않은 경우
    if session_id not in sessions:
        # 새 세션 생성 및 기본 시스템 프롬프트 추가
        # Function calling을 사용하므로 시스템 프롬프트는 간단하게 유지
        sessions[session_id] = [
            {"role": "system", "content": "당신은 유능한 상담 전문가입니다."}
        ]

    # 활성 WebSocket 연결 목록에 현재 연결 추가
    # 나중에 서버에서 클라이언트로 메시지를 푸시할 때 사용
    active_connections[session_id] = websocket

    # 연결 성공을 알리는 초기 메시지를 클라이언트에게 전송
    # send_json: Python 딕셔너리를 자동으로 JSON으로 변환하여 전송
    await websocket.send_json({
        # type: 메시지 유형 (클라이언트가 메시지 종류를 구분하기 위해 사용)
        "type": "connection",
        # message: 연결 성공 메시지
        "message": "Connected to chatbot",
        # session_id: 현재 세션 ID (클라이언트 확인용)
        "session_id": session_id
    })

    # try 블록: WebSocket 통신 중 발생하는 예외 처리
    try:
        # 무한 루프: 클라이언트로부터 계속해서 메시지 수신
        while True:
            # 클라이언트가 보낸 텍스트 메시지를 비동기로 수신
            # 메시지가 올 때까지 대기 (블로킹되지 않음)
            data = await websocket.receive_text()

            # 내부 try 블록: JSON 파싱 시도
            try:
                # 받은 텍스트를 JSON으로 파싱
                message_data = json.loads(data)
                # "message" 키의 값을 추출, 없으면 원본 데이터 사용
                user_message = message_data.get("message", data)
            # JSON 파싱 실패 시 (일반 텍스트인 경우)
            except:
                # 받은 데이터를 그대로 메시지로 사용
                user_message = data

            # 사용자 메시지를 세션의 대화 히스토리에 추가
            sessions[session_id].append({
                "role": "user",
                "content": user_message
            })

            # AI가 응답을 생성 중임을 클라이언트에 알림
            # UI에서 "타이핑 중..." 표시를 위해 사용
            await websocket.send_json({
                "type": "typing",
                "message": "AI is typing..."
            })

            # 내부 try 블록: VLLM 서버 통신 중 예외 처리
            try:
                # VLLM 서버에 비동기 채팅 완성 요청 (tools 파라미터 포함)
                response = await client.chat.completions.create(
                    # 사용할 모델 이름
                    model="openai/gpt-oss-20b",
                    # 전체 대화 히스토리 전송 (컨텍스트 유지)
                    messages=sessions[session_id],
                    # tools: AI가 사용할 수 있는 함수 목록 (현재 시간 조회 함수)
                    tools=tools,
                    # tool_choice: AI가 필요할 때 자동으로 도구를 선택하도록 설정
                    tool_choice="auto",
                    # 최대 생성 토큰 수 (응답 길이 제한)
                    max_tokens=512,
                    # 샘플링 온도 (창의성 조절)
                    temperature=0.7
                )

                # AI의 응답 메시지 객체 가져오기
                response_message = response.choices[0].message

                # AI가 도구(함수)를 호출했는지 확인
                if response_message.tool_calls:
                    # AI의 응답(함수 호출 요청)을 대화 히스토리에 추가
                    sessions[session_id].append({
                        "role": "assistant",
                        "content": response_message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            } for tc in response_message.tool_calls
                        ]
                    })

                    # 각 함수 호출을 처리
                    for tool_call in response_message.tool_calls:
                        # 함수 이름 가져오기
                        function_name = tool_call.function.name

                        # 현재는 get_current_time 함수만 지원
                        if function_name == "get_current_time":
                            # 실제 함수를 호출하여 현재 시간 가져오기
                            function_response = get_current_time()

                            # 함수 실행 결과를 대화 히스토리에 추가
                            sessions[session_id].append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": function_name,
                                "content": function_response
                            })

                    # 함수 실행 결과를 포함하여 AI에게 다시 요청
                    second_response = await client.chat.completions.create(
                        model="openai/gpt-oss-20b",
                        messages=sessions[session_id],
                        max_tokens=512,
                        temperature=0.7
                    )

                    # 최종 AI 응답 메시지 추출
                    ai_message = second_response.choices[0].message.content
                else:
                    # 함수 호출 없이 바로 응답한 경우
                    ai_message = response_message.content

                # AI의 최종 응답을 대화 히스토리에 추가
                sessions[session_id].append({
                    "role": "assistant",
                    "content": ai_message
                })

                # AI 응답을 클라이언트에게 실시간으로 전송
                await websocket.send_json({
                    # type: 일반 메시지 타입
                    "type": "message",
                    # role: 메시지 발신자 (AI 어시스턴트)
                    "role": "assistant",
                    # content: AI가 생성한 응답 텍스트
                    "content": ai_message,
                    # timestamp: 메시지 생성 시각 (ISO 8601 형식)
                    "timestamp": datetime.now().isoformat()
                })

            # VLLM 서버 통신 중 에러 발생 시
            except Exception as e:
                # 에러 메시지를 클라이언트에게 전송
                # 클라이언트가 에러를 UI에 표시할 수 있도록 함
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error: {str(e)}"
                })

    # WebSocket 연결이 끊어졌을 때 발생하는 예외 처리
    except WebSocketDisconnect:
        # 활성 연결 목록에서 해당 세션 제거
        if session_id in active_connections:
            del active_connections[session_id]
        # 서버 콘솔에 연결 해제 로그 출력 (디버깅용)
        print(f"Client {session_id} disconnected")


# "/sessions" 경로에 대한 GET 요청 핸들러
# 현재 서버에 저장된 모든 세션의 목록과 정보를 조회
@app.get("/sessions")
async def list_sessions():
    # 세션 목록과 통계를 포함한 응답 반환
    return {
        # total_sessions: 전체 세션 개수
        "total_sessions": len(sessions),
        # sessions: 각 세션의 상세 정보를 담은 리스트
        # 리스트 컴프리헨션으로 딕셔너리를 순회하며 정보 생성
        "sessions": [
            {
                # session_id: 세션의 고유 식별자
                "session_id": sid,
                # message_count: 해당 세션의 메시지 개수
                "message_count": len(messages),
                # is_connected: WebSocket으로 현재 연결되어 있는지 여부
                "is_connected": sid in active_connections
            }
            # for 루프: sessions 딕셔너리의 모든 항목(세션)을 순회
            # sid: 세션 ID, messages: 메시지 리스트
            for sid, messages in sessions.items()
        ]
    }


# "/stats" 경로에 대한 GET 요청 핸들러
# 서버의 전반적인 통계 정보를 조회
@app.get("/stats")
async def get_stats():
    # 모든 세션의 메시지 수를 합산
    # sum(): 리스트의 모든 요소를 더함
    # for ... in sessions.values(): 모든 세션의 메시지 리스트를 순회
    total_messages = sum(len(messages) for messages in sessions.values())

    # 서버 통계 정보를 딕셔너리로 반환
    return {
        # total_sessions: 생성된 전체 세션 수
        "total_sessions": len(sessions),
        # active_websocket_connections: 현재 WebSocket으로 연결된 클라이언트 수
        "active_websocket_connections": len(active_connections),
        # total_messages: 모든 세션의 총 메시지 수
        "total_messages": total_messages,
        # timestamp: 통계 조회 시각 (ISO 8601 형식)
        "timestamp": datetime.now().isoformat()
    }


# 이 스크립트가 직접 실행될 때만 실행되는 블록
# 다른 모듈에서 임포트될 때는 실행되지 않음
if __name__ == "__main__":
    # uvicorn 모듈을 임포트
    # uvicorn: ASGI 웹 서버로 FastAPI 애플리케이션을 실행
    import uvicorn

    # uvicorn.run(): FastAPI 애플리케이션을 ASGI 서버로 실행
    # app: 실행할 FastAPI 애플리케이션 인스턴스
    # host="0.0.0.0": 모든 네트워크 인터페이스에서 접속 허용
    #   (localhost뿐만 아니라 외부 네트워크에서도 접근 가능)
    # port=8080: 서버가 수신 대기할 포트 번호
    uvicorn.run(app, host="0.0.0.0", port=8080)
