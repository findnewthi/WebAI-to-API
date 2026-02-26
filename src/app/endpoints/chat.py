# src/app/endpoints/chat.py
import time
from fastapi import APIRouter, HTTPException, Header, Depends
from app.logger import logger
from schemas.request import GeminiRequest, OpenAIChatRequest
from app.services.gemini_client import get_gemini_client, GeminiClientNotInitializedError
from app.services.session_manager import get_translate_session_manager
import os
from fastapi.responses import StreamingResponse
from typing import Optional
import json
import asyncio

router = APIRouter()

@router.post("/translate")
async def translate_chat(request: GeminiRequest):
    try:
        gemini_client = get_gemini_client()
    except GeminiClientNotInitializedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    session_manager = get_translate_session_manager()
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager is not initialized.")
    try:
        # This call now correctly uses the fixed session manager
        response = await session_manager.get_response(request.model, request.message, request.files)
        return {"response": response.text}
    except Exception as e:
        logger.error(f"Error in /translate endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during translation: {str(e)}")

def convert_to_openai_format(response_text: str, model: str, stream: bool = False):
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk" if False else "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": len(response_text),
            "total_tokens": len(response_text),
        },
    }

API_KEY = os.getenv("OPENAI_API_KEY", "default")
def verify_openai_bearer(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(401, detail="Missing Authorization header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token != API_KEY:
        raise HTTPException(401, detail="Invalid authentication credentials")
    return token          # 校验通过即可往下走
# ---------- 1. 非流/流共用：生成一个 chunk 的字典 ----------
def make_chunk(content: str, model: str, finish_reason: Optional[str] = None):
    return {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content else {},
            "finish_reason": finish_reason
        }]
    }

# ---------- 2. 流式生成器 ----------
async def stream_generate(answer, model: str):
    # 按字/句切片模拟流
    chunk_size = 4
    for i in range(0, len(answer.text), chunk_size):
        yield f"data: {json.dumps(make_chunk(answer.text[i:i+chunk_size], model))}\n\n"
        await asyncio.sleep(0.02)          # 控制打字效果，可删
    # 发送 stop
    yield f"data: {json.dumps(make_chunk('', model, finish_reason='stop'))}\n\n"
    # 官方结束标记
    yield "data: [DONE]\n\n"

@router.post("/v1/chat/completions", dependencies=[Depends(verify_openai_bearer)])
async def chat_completions(request: OpenAIChatRequest):
    try:
        gemini_client = get_gemini_client()
    except GeminiClientNotInitializedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    is_stream = request.stream if request.stream is not None else False

    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided.")

    # Build conversation prompt with system prompt and full history
    conversation_parts = []

    for msg in request.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not content:
            continue

        if role == "system":
            conversation_parts.append(f"System: {content}")
        elif role == "user":
            conversation_parts.append(f"User: {content}")
        elif role == "assistant":
            conversation_parts.append(f"Assistant: {content}")

    if not conversation_parts:
        raise HTTPException(status_code=400, detail="No valid messages found.")

    # Join all parts with newlines
    final_prompt = "\n\n".join(conversation_parts)

    if request.model:
        try:
            response = await gemini_client.generate_content(message=final_prompt, model=request.model.value, files=None)
            if is_stream:
                return StreamingResponse(
                    stream_generate(response, request.model.value),
                    media_type="text/plain; charset=utf-8"
                )
            return convert_to_openai_format(response.text, request.model.value, is_stream)
        except Exception as e:
            logger.error(f"Error in /v1/chat/completions endpoint: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error processing chat completion: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Model not specified in the request.")
