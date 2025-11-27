from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI

from typing import List, Dict
from uuid import uuid4
from datetime import datetime
import asyncio
import json

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

def get_current_time() -> str:
    """
    현재 날짜, 시간, 요일 정보를 반환하는 함수
    AI가 사용자가 시간을 물어보면 이 함수를 호출하여 정확한 시간을 얻을 수 있음
    """

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

    return f"현재 시간: {datetime_str} ({day_of_week_kr})"


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "현재 날짜, 시간, 요일 정보를 가져옵니다. 사용자가 현재 시간, 날짜, 요일을 물어볼 때 이 함수를 호출하세요.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


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


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # del: delete 키워드로 세션 데이터 삭제
    del sessions[session_id]

    if session_id in active_connections:
        del active_connections[session_id]

    return {"message": "Session deleted successfully"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.session_id or request.session_id not in sessions:
        session_id = str(uuid4())
        sessions[session_id] = [
            {"role": "system", "content": request.system_prompt}
        ]
    else:
        session_id = request.session_id

    sessions[session_id].append({
        "role": "user",
        "content": request.message
    })

    try:
        response = await client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=sessions[session_id],
            tools=tools,
            tool_choice="auto",
            max_tokens=512,
            temperature=0.7
        )

        response_message = response.choices[0].message

        if response_message.tool_calls:
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

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name

                if function_name == "get_current_time":
                    function_response = get_current_time()
                    sessions[session_id].append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": function_response
                    })

            second_response = await client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=sessions[session_id],
                max_tokens=512,
                temperature=0.7
            )

            ai_message = second_response.choices[0].message.content
        else:
            ai_message = response_message.content

        sessions[session_id].append({
            "role": "assistant",
            "content": ai_message
        })

        return ChatResponse(
            session_id=session_id,
            response=ai_message,
            conversation_history=sessions[session_id]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()

    if session_id not in sessions:
        sessions[session_id] = [
            {"role": "system", "content": "당신은 유능한 상담 전문가입니다."}
        ]

    active_connections[session_id] = websocket

    await websocket.send_json({
        "type": "connection",
        "message": "Connected to chatbot",
        "session_id": session_id
    })

    try:
        while True:
            data = await websocket.receive_text()

            try:
                message_data = json.loads(data)
                user_message = message_data.get("message", data)
            except:
                user_message = data

            sessions[session_id].append({
                "role": "user",
                "content": user_message
            })

            await websocket.send_json({
                "type": "typing",
                "message": "AI is typing..."
            })

            try:
                response = await client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=sessions[session_id],
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=512,
                    temperature=0.7
                )

                response_message = response.choices[0].message

                if response_message.tool_calls:
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

                    for tool_call in response_message.tool_calls:
                        function_name = tool_call.function.name

                        if function_name == "get_current_time":
                            function_response = get_current_time()

                            sessions[session_id].append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": function_name,
                                "content": function_response
                            })

                    second_response = await client.chat.completions.create(
                        model="openai/gpt-oss-20b",
                        messages=sessions[session_id],
                        max_tokens=512,
                        temperature=0.7
                    )

                    ai_message = second_response.choices[0].message.content
                else:
                    ai_message = response_message.content

                sessions[session_id].append({
                    "role": "assistant",
                    "content": ai_message
                })

                await websocket.send_json({
                    "type": "message",
                    "role": "assistant",
                    "content": ai_message,
                    "timestamp": datetime.now().isoformat()
                })

            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error: {str(e)}"
                })

    except WebSocketDisconnect:
        if session_id in active_connections:
            del active_connections[session_id]
        print(f"Client {session_id} disconnected")


@app.get("/sessions")
async def list_sessions():
    return {
        "total_sessions": len(sessions),
        "sessions": [
            {
                "session_id": sid,
                "message_count": len(messages),
                "is_connected": sid in active_connections
            }
            for sid, messages in sessions.items()
        ]
    }


@app.get("/stats")
async def get_stats():
    total_messages = sum(len(messages) for messages in sessions.values())

    return {
        "total_sessions": len(sessions),
        "active_websocket_connections": len(active_connections),
        "total_messages": total_messages,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
