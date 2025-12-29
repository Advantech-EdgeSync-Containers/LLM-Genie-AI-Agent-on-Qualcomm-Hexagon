import os
import re
import time
import uuid
import asyncio
import json
import requests

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Literal, Optional, Tuple, Dict, Union
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler

from llm_loader import get_genie_llm
from agent_setup import MyLLMDecisionAgent
from tools import tools

app = FastAPI()
TIMEOUT_SECONDS = int(os.getenv("MAX_PROMPT_GENERATION_TIMEOUT_IN_MIN", 20)) * 60


class Message(BaseModel):
    role: Literal["user", "system", "assistant"]
    content: str


class BackgroundTasks(BaseModel):
    title_generation: Optional[bool] = False
    tags_generation: Optional[bool] = False


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    background_tasks: Optional[Dict[str, bool]] = None
    stream: Optional[bool] = False


def build_chat_response(content: Union[str, dict], model: str, role: str = "assistant", is_json: bool = False):
    content_str = json.dumps(content) if is_json else content

    response = {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "message": {
                    "role": role,
                    "content": content_str
                },
                "finish_reason": "stop",
                "index": 0
            }
        ]
    }

    return response



async def token_stream(agent,model_name, prompt, callback, request: Request):
    task = asyncio.create_task(agent._chain.ainvoke({"question": prompt}))
    print(f"task: {task}")
    try:
        buffer = ""
        buffer_after_think = ""
        think_complete = False
        tool_triggered = False
        start_time = time.time()

        async for token in callback.aiter():
            buffer += token
            token = token.replace("\\", "\\\\").replace('"', '\\"')
            if not think_complete:
                if "</think>" in buffer.lower():
                    think_complete = True
            else:
                buffer_after_think += token
            data = {
                "id": "think",
                "object": "chat.completion.chunk",
                "choices": [
                    {"delta": {"content": token}, "index": 0, "finish_reason": None}
                ],
            }

            if time.time() - start_time > TIMEOUT_SECONDS:
                print("Stream Chat Timeout: Stopping stream...")
                task.cancel()
                agent.stop(model_name)
                break
            yield f"data: {json.dumps(data)}\n\n"

        if not task.done():
            await task

        if not think_complete:
            buffer_after_think = buffer

        if not tool_triggered:
            tool_data = agent.extract_tool(buffer_after_think)
            if tool_data and tool_data.get("tool_name", None):

                tool_lookup = {tool.name.lower(): tool for tool in tools}
                tool_triggered = True
                result = tool_lookup[tool_data["tool_name"]].run({"input": tool_data})
                for chunk in re.findall(r"\S+\s*|\n+", result):
                    escaped_text = chunk.replace("\\", "\\\\").replace('"', '\\"')
                    data = {
                        "id": str(uuid.uuid4()),
                        "object": "chat.completion.chunk",
                        "choices": [
                            {"delta": {"content": escaped_text}, "index": 0, "finish_reason": None}
                        ],
                    }

                    yield f"data: {json.dumps(data)}\n\n"
                    await asyncio.sleep(0.005)  # simulate stream

                yield 'data: [DONE]\n\n'

        yield f'data: {json.dumps({"id": "finish", "object": "chat.completion.chunk", "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]})}\n\n'
        yield 'data: [DONE]\n\n'
    except asyncio.CancelledError:
        print("Agent task cancelled cleanly due to user stop.")
        task.cancel()
    except Exception as e:
        print(f"Error during streaming: {e}")
        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completion(request: Request, chat_request: ChatRequest):
    prompt = chat_request.messages[-1].content
    model_name = chat_request.model

    callback = AsyncIteratorCallbackHandler()
    llm = get_genie_llm(callback_handler=callback)
    agent = MyLLMDecisionAgent(tools=tools, llm=llm)

    print(f"prompt:{prompt}")

    if "### Task:" in prompt:
        result = await llm.ainvoke(prompt)
        result = agent.remove_think_block(result)
        result = agent.extract_json_key(result, "title")

        return build_chat_response({"title": result}, model_name, is_json=True)
    else:
        if chat_request.stream:
            return StreamingResponse(
                token_stream(agent,model_name, prompt, callback, request),
                media_type="text/event-stream"
            )
        else:
            try:
                result = await asyncio.wait_for(llm.ainvoke(prompt), timeout=TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                print("No Stream LLM call timed out.")
                # Optionally stop the model if still running
                result = "The request took too long and was stopped."
            return build_chat_response(result, model_name)


# ---- Models Listing Endpoint for OpenWebUI Compatibility

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": os.getenv("MODEL_NAME"),
                "object": "model",
                "owned_by": "user",
                "size": "1B",
                "modified": time.strftime('%Y-%m-%dT%H:%M:%S')
            }
        ]
    }
