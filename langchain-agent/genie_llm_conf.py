import json
import os
import re

import httpx
import asyncio
from typing import Any, List, Optional, Literal
from pydantic import BaseModel

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatResult,
    ChatGeneration,
    HumanMessage,
    SystemMessage,
)
from langchain.chat_models.base import BaseChatModel


MODEL_CONFIG = os.getenv("MODEL_CONFIG")

CONTROL_TOKEN_RE = re.compile(
    r"(?:</?s>|<\|?(?:system|user|assistant|end|bos|eos)\|?>)",
    flags=re.IGNORECASE,
)

class ChatMessage(BaseModel):
    role: Literal["user", "system", "assistant"]
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False

# ----------------- Prompt Formatter --------------
def build_prompt(messages: List[ChatMessage]) -> str:
    system_msg = next(
        (msg.content for msg in messages if msg.role == "system"),
        "You are a helpful assistant."
    )
    last_user_msg = next(
        (msg.content for msg in reversed(messages) if msg.role == "user"),
        None
    )

    if not last_user_msg:
        raise ValueError("No user message found in the request.")

    return f"<|system|>{system_msg}</s><|user|>{last_user_msg}</s><|assistant|>"

def clean_model_output(text: str) -> str:
    text = CONTROL_TOKEN_RE.sub("", text)
    text = text.split("<|assistant|>", 1)[0]
    return text.strip()


def build_genie_prompt(messages: list[BaseMessage]) -> str:
    system_msg = None
    user_msg = None

    for m in messages:
        if isinstance(m, SystemMessage) and not system_msg:
            system_msg = m.content
        elif isinstance(m, HumanMessage):
            user_msg = m.content  # last HumanMessage wins

    if not user_msg:
        raise ValueError("No user message found.")

    return f"<|system|>{system_msg or 'You are a helpful assistant.'}</s><|user|>{user_msg}</s><|assistant|>"

class GenieLLM(BaseChatModel):
    """LangChain wrapper around `genie-t2t-run` with streaming support."""

    model: str = "llama32-1b-gqa"
    callback_handler: Optional[Any] = None

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = build_genie_prompt(messages)
        cmd = ["genie-t2t-run", "-c", MODEL_CONFIG, "-p", prompt]

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )

        collected_text = ""
        buf = ""
        capturing = False

        while True:
            chunk = await process.stdout.read(64)  # read a bit more than 1 byte
            if not chunk:
                break

            buf += chunk.decode("utf-8", errors="ignore")

            if not capturing and "[BEGIN]:" in buf:
                buf = buf.split("[BEGIN]:", 1)[1]
                capturing = True

            if not capturing:
                continue

            # Stop conditions
            if "[END]" in buf or "[ABORT]" in buf:
                stop_tag = "[END]" if "[END]" in buf else "[ABORT]"
                text_to_use = buf.split(stop_tag, 1)[0]
                new_text = clean_model_output(text_to_use)

                # final flush
                delta = new_text[len(collected_text):]
                if delta:
                    collected_text += delta
                    if run_manager:
                        await run_manager.on_llm_new_token(delta, verbose=False)
                    if self.callback_handler and hasattr(self.callback_handler, "on_llm_new_token"):
                        maybe_coro = self.callback_handler.on_llm_new_token(delta, verbose=False)
                        if asyncio.iscoroutine(maybe_coro):
                            await maybe_coro

                break

            # ✅ Stream intermediate chunks
            cleaned = clean_model_output(buf)
            delta = cleaned[len(collected_text):]
            if delta:
                collected_text += delta
                if run_manager:
                    await run_manager.on_llm_new_token(delta, verbose=False)
                if self.callback_handler and hasattr(self.callback_handler, "on_llm_new_token"):
                    maybe_coro = self.callback_handler.on_llm_new_token(delta, verbose=False)
                    if asyncio.iscoroutine(maybe_coro):
                        await maybe_coro

        # finalize
        if self.callback_handler and hasattr(self.callback_handler, "on_llm_end"):
            maybe_coro = self.callback_handler.on_llm_end({}, verbose=False)
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro

        message = AIMessage(content=collected_text.strip())
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _generate(self, messages: List[BaseMessage], **kwargs: Any) -> ChatResult:
        return asyncio.run(self._agenerate(messages, **kwargs))

    @property
    def _llm_type(self) -> str:
        return "genie"