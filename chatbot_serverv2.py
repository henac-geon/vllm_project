from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from openai import AsyncOpenAI

from typing import List, Dict
from uuid import uuid4
from datetime import datetime

# Agent SDK 임포트
from agents import Agent, Runner, function_tool, OpenAIResponsesModel, set_tracing_disabled

import asyncio
import json

set_tracing_disabled(True)

app = FastAPI(title="Multi-User Chatbot System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

sessions: Dict[str, List[Dict[str, str]]] = {}

active_connections: Dict[str, WebSocket] = {}


@function_tool
def get_current_time() -> str:
    """
    현재 날짜, 시간, 요일 정보를 반환하는 함수
    AI가 사용자가 시간을 물어보면 이 함수를 호출하여 정확한 시간을 얻을 수 있음
    """
    print("=" * 60)
    print("[FUNCTION CALL] get_current_time() 함수가 호출되었습니다!")
    print("=" * 60)

    now = datetime.now()
    datetime_str = now.strftime("%Y-%m-%d %H:%M:%S")
    day_of_week_en = now.strftime("%A")
    days_kr = {
        "Monday": "월요일",
        "Tuesday": "화요일",
        "Wednesday": "수요일",
        "Thursday": "목요일",
        "Friday": "금요일",
        "Saturday": "토요일",
        "Sunday": "일요일"
    }
    day_of_week_kr = days_kr[day_of_week_en]
    result = f"현재 시간: {datetime_str} ({day_of_week_kr})"

    print(f"[FUNCTION RESULT] {result}")
    print("=" * 60)

    return result

# 자기소개
@function_tool
def get_myinfo() -> str:
    """
    나의 정보를 반환하는 함수
    AI가 사용자가 나에 대해 물어보면 이 함수를 호출하여 정확한 정보를 얻을 수 있음
    """
    print("=" * 60)
    print("[FUNCTION CALL] get_myinfo() 함수가 호출되었습니다!")
    print("=" * 60)

    result = "저는 제건이라고 합니다. 인생 날로 먹고 싶은 개발자입니다."

    print(f"[FUNCTION RESULT] {result}")
    print("=" * 60)

    return result


def create_agent(system_prompt: str = "당신은 유능한 AI 어시스턴트입니다.") -> Agent:
    """
    Agent SDK를 사용하여 AI 에이전트 인스턴스를 생성

    Args:
        system_prompt: 에이전트의 역할과 행동을 정의하는 시스템 프롬프트

    Returns:
        Agent: 설정된 Agent 인스턴스
    """
    print(f"[AGENT CREATE] 새로운 Agent 인스턴스 생성 중...")

    agent = Agent(
        name="ChatbotAssistant",
        instructions=system_prompt,
        model=OpenAIResponsesModel(
            model="openai/gpt-oss-20b",
            openai_client=client,
        ),
        tools=[get_current_time, get_myinfo],
    )

    print(f"[AGENT CREATE] ✅ Agent 생성 완료 - tools: [get_current_time, get_myinfo]")
    return agent


class ChatRequest(BaseModel):
    session_id: str = None
    message: str
    system_prompt: str = "당신은 유능한 상담 전문가입니다."


class ChatResponse(BaseModel):
    session_id: str

    response: str

    conversation_history: List[Dict[str, str]]


@app.get("/")
async def root():
    return {
        "status": "running",
        "message": "Multi-User Chatbot API Server",
        "active_sessions": len(sessions),
        "active_connections": len(active_connections)
    }


@app.post("/session/create")
async def create_session():
    session_id = str(uuid4())

    sessions[session_id] = []

    return {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "message": "Session created successfully"
    }

@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "history": sessions[session_id],
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

        # 새 세션 생성 시 빈 대화 히스토리로 시작
        # Agent SDK가 시스템 프롬프트를 관리하므로 별도로 저장하지 않음
        sessions[session_id] = []
    # 기존 세션이 존재하는 경우
    else:
        # 요청에서 제공된 세션 ID를 그대로 사용
        session_id = request.session_id

    # 사용자가 보낸 메시지 로그 출력
    print("\n" + "" * 30)
    print(f"[USER MESSAGE] 사용자 메시지: {request.message}")
    print("" * 30 + "\n")

    # 사용자 메시지를 대화 히스토리에 추가 (기록용)
    sessions[session_id].append({
        "role": "user",
        "content": request.message
    })

    # try 블록: Agent 실행 중 발생할 수 있는 예외 처리
    try:
        # Agent 인스턴스 생성 (시스템 프롬프트 전달)
        agent = create_agent(system_prompt=request.system_prompt)

        print(f"[AGENT RUN] Agent 실행 중... 질문: {request.message}")

        # Agent SDK의 Runner를 사용하여 Agent 실행
        # Agent가 자동으로 필요한 경우 get_current_time() 함수를 호출
        result = await Runner.run(agent, request.message)

        # Agent의 최종 응답 추출
        ai_message = result.final_output or "응답을 생성할 수 없습니다."

        print("\n" + "" * 30)
        print(f"[AI RESPONSE] AI 응답 생성 완료")
        print(f"[RESPONSE] {ai_message[:100]}...")
        print("" * 30 + "\n")

        # AI의 최종 응답을 대화 히스토리에 추가 (기록용)
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
        # 새 세션 생성 - 빈 대화 히스토리로 시작
        # Agent SDK가 시스템 프롬프트를 관리
        sessions[session_id] = []

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

            # 사용자 메시지 로그 출력
            print("\n" + "" * 30)
            print(f"[WebSocket USER] 사용자 메시지: {user_message}")
            print("" * 30 + "\n")

            # 사용자 메시지를 대화 히스토리에 추가 (기록용)
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

            # 내부 try 블록: Agent 실행 중 예외 처리
            try:
                # Agent 인스턴스 생성 (기본 시스템 프롬프트 사용)
                agent = create_agent(system_prompt="당신은 유능한 상담 전문가입니다.")

                print(f"[AGENT RUN] Agent 실행 중... 질문: {user_message}")

                # Agent SDK의 Runner를 사용하여 Agent 실행
                # Agent가 자동으로 필요한 경우 get_current_time() 함수를 호출
                result = await Runner.run(agent, user_message)

                # Agent의 최종 응답 추출
                ai_message = result.final_output or "응답을 생성할 수 없습니다."

                print("\n" + "" * 30)
                print(f"[WebSocket] AI 응답 생성 완료")
                print(f"[RESPONSE] {ai_message[:100]}...")
                print("" * 30 + "\n")

                # AI의 최종 응답을 대화 히스토리에 추가 (기록용)
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

            # Agent 실행 중 에러 발생 시
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

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=24100)
