"""
OpenAI LLM Interface Module
==========================

This module provides interfaces for interacting with openai's language models,
including text generation and embedding capabilities.

Author: Lightrag team
Created: 2024-01-24
License: MIT License

Copyright (c) 2024 Lightrag

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

Version: 1.0.0

Change Log:
- 1.0.0 (2024-01-24): Initial release
    * Added async chat completion support
    * Added embedding generation
    * Added stream response capability

Dependencies:
    - openai
    - numpy
    - pipmaster
    - Python >= 3.10

Usage:
    from llm_interfaces.openai import openai_model_complete, openai_embed
"""

__version__ = "1.0.0"
__author__ = "lightrag Team"
__status__ = "Production"


import sys
import os

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator
import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed("openai"):
    pm.install("openai")

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from minirag.utils import (
    wrap_embedding_func_with_attrs,
    locate_json_string_body_from_string,
    safe_unicode_decode,
    logger,
)
from pydantic import BaseModel
from typing import List
import numpy as np
from typing import Union


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)

class GPTKeywordExtractionFormat(BaseModel):
    high_level_keywords: List[str]
    low_level_keywords: List[str]

async def openai_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    base_url=None,
    api_key=None,
    **kwargs,
) -> str:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    # Respect an explicitly passed base_url; otherwise, try environment var without raising if absent
    if base_url is None:
        base_url = os.environ.get("OPENAI_API_BASE")
    openai_async_client = (
        AsyncOpenAI() if base_url is None else AsyncOpenAI(base_url=base_url)
    )
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # 添加日志输出
    logger.debug("===== Query Input to LLM =====")
    logger.debug(f"Query: {prompt}")
    logger.debug(f"System prompt: {system_prompt}")
    logger.debug("Full context:")
    if "response_format" in kwargs:
        response = await openai_async_client.beta.chat.completions.parse(
            model=model, messages=messages, **kwargs
        )
    else:
        response = await openai_async_client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )

    if hasattr(response, "__aiter__"):

        async def inner():
            async for chunk in response:
                content = chunk.choices[0].delta.content
                if content is None:
                    continue
                if r"\u" in content:
                    content = safe_unicode_decode(content.encode("utf-8"))
                yield content

        return inner()
    else:
        if not response or not hasattr(response, "choices") or not response.choices:
            logger.error("No valid choices returned. Full response: %s", response)
            return ""  # or raise a more specific exception
        content = response.choices[0].message.content
        if content is None:
            logger.error("The message content is None. Full response: %s", response)
            return ""
        if r"\u" in content:
            content = safe_unicode_decode(content.encode("utf-8"))
        return content


async def openai_complete(
    prompt, model_name=None, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> Union[str, AsyncIterator[str]]:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = "json"
    if model_name is None:
        model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def nvidia_openai_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    result = await openai_complete_if_cache(
        "nvidia/llama-3.1-nemotron-70b-instruct",  # context length 128k
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url="https://integrate.api.nvidia.com/v1",
        **kwargs,
    )
    if keyword_extraction:  # TODO: use JSON API
        return locate_json_string_body_from_string(result)
    return result

async def openrouter_openai_complete(
    prompt, 
    system_prompt=None, 
    history_messages=[], 
    keyword_extraction=False, 
    api_key: str = None, 
    **kwargs,
) -> str:
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key

    keyword_extraction = kwargs.pop("keyword_extraction", None)
    result = await openai_complete_if_cache(
        "google/gemini-2.0-flash-001",  # change accordingly
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        **kwargs,
    )
    if keyword_extraction:  # TODO: use JSON API
        return locate_json_string_body_from_string(result)
    return result

@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def openai_embed(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str = None,
    api_key: str = None,
) -> np.ndarray:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    openai_async_client = (
        AsyncOpenAI() if base_url is None else AsyncOpenAI(base_url=base_url)
    )
    response = await openai_async_client.embeddings.create(
        model=model, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


# ==============================================================
# Batch API Support (asynchronous queued chat completions)
# ==============================================================
# This allows deferring many LLM calls (e.g., entity/relationship extraction)
# into OpenAI's Batch API for cost savings and higher rate limits. Each queued
# request returns a Future that resolves only after the batch finishes.

import asyncio
import time
import json
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict

try:  # sync client
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None

FINAL_BATCH_STATES = {"completed", "failed", "cancelled", "expired"}

@dataclass
class _QueuedRequest:
    custom_id: str
    body: Dict[str, Any]
    future: asyncio.Future

@dataclass
class OpenAIBatchManager:
    model: str = "gpt-4o-mini"
    max_batch_size: int = 400  # defensive default < 50k
    flush_interval: float = 180.0  # seconds
    poll_interval: float = 20.0
    organization: str | None = None
    api_key: str | None = None
    _queue: list[_QueuedRequest] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _last_flush: float = field(default_factory=time.time)
    _running: bool = True
    _background_task: asyncio.Task | None = None
    _client: Any = None
    # Track batch statuses for external progress reporting
    _batches: dict[str, str] = field(default_factory=dict)  # batch_id -> status
    _total_submitted_reqs: int = 0
    _total_batches: int = 0
    _batch_history: list[dict] = field(default_factory=list)
    _listeners: list[Any] = field(default_factory=list)  # callables notified on new batch submit

    @property
    def queue_len(self) -> int:
        """Public read-only length of the pending request queue."""
        return len(self._queue)

    def ensure_client(self):
        if self._client is None:
            if OpenAI is None:
                raise RuntimeError("OpenAI client unavailable. Install openai >=1.0.")
            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.organization:
                kwargs["organization"] = self.organization
            self._client = OpenAI(**kwargs)  # type: ignore

    async def start(self):
        if self._background_task is None:
            self._background_task = asyncio.create_task(self._periodic_flush())

    async def stop(self):  # pragma: no cover
        self._running = False
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except Exception:
                pass

    async def _periodic_flush(self):  # pragma: no cover
        while self._running:
            try:
                await asyncio.sleep(1.0)
                if (time.time() - self._last_flush) >= self.flush_interval:
                    await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch periodic flush error: {e}")

    async def queue_chat_completion(
        self,
        prompt: str,
        system_prompt: str | None,
        history_messages: list[dict] | None,
        **kwargs,
    ) -> str:
        """Queue a chat completion; returns awaited content after batch completes."""
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": kwargs.pop("model", self.model),
            "messages": messages,
        }
        for k in ["temperature", "max_tokens", "top_p", "response_format"]:
            if k in kwargs:
                body[k] = kwargs[k]
        custom_id = f"chat-{int(time.time()*1000)}-{len(self._queue)+1}"
        async with self._lock:
            self._queue.append(_QueuedRequest(custom_id, body, fut))
            if len(self._queue) >= self.max_batch_size:
                await self.flush()
        return await fut

    async def flush(self):
        async with self._lock:
            if not self._queue:
                return
            snapshot = self._queue
            self._queue = []
            self._last_flush = time.time()
        self.ensure_client()
        fd, path = tempfile.mkstemp(prefix="openai_batch_input_", suffix=".jsonl")
        with open(fd, "w", encoding="utf-8") as f:
            for req in snapshot:
                rec = {
                    "custom_id": req.custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": req.body,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        loop = asyncio.get_event_loop()
        try:
            file_obj = await loop.run_in_executor(None, lambda: self._client.files.create(file=open(path, "rb"), purpose="batch"))
            batch = await loop.run_in_executor(
                None,
                lambda: self._client.batches.create(
                    input_file_id=file_obj.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                ),
            )
            # record batch initial status
            try:
                bid = getattr(batch, "id", None) or batch["id"]
                bstatus = getattr(batch, "status", None) or batch.get("status") or "submitted"
                self._batches[bid] = bstatus
                size = len(snapshot)
                self._total_submitted_reqs += size
                self._total_batches += 1
                meta = {"id": bid, "status": bstatus, "size": size, "ts": time.time(), "total_reqs": self._total_submitted_reqs}
                self._batch_history.append(meta)
                # Trim history to last 200
                if len(self._batch_history) > 200:
                    self._batch_history = self._batch_history[-200:]
                print(f"[Batch Submit] id={bid} size={size} total_batches={self._total_batches} total_reqs={self._total_submitted_reqs}")
                for cb in list(self._listeners):
                    try:
                        cb(meta)
                    except Exception:
                        pass
            except Exception:  # pragma: no cover
                pass
        finally:
            try:
                os.remove(path)
            except Exception:
                pass
        mapping = {r.custom_id: r for r in snapshot}
        asyncio.create_task(self._poll_batch(batch.id, mapping))

    async def _poll_batch(self, batch_id: str, mapping: Dict[str, _QueuedRequest]):  # pragma: no cover
        self.ensure_client()
        loop = asyncio.get_event_loop()
        while True:
            batch = await loop.run_in_executor(None, lambda: self._client.batches.retrieve(batch_id))
            status = getattr(batch, "status", None) or batch.get("status")
            # update status snapshot
            self._batches[batch_id] = status
            if status in FINAL_BATCH_STATES:
                break
            await asyncio.sleep(self.poll_interval)
        try:
            status = getattr(batch, "status", None)
            # final status update
            self._batches[batch_id] = status or self._batches.get(batch_id, "unknown")
            if status == "completed" and getattr(batch, "output_file_id", None):
                out_id = batch.output_file_id
                file_resp = await loop.run_in_executor(None, lambda: self._client.files.content(out_id))
                text_content = getattr(file_resp, "text", None) or file_resp
                for line in text_content.splitlines():
                    try:
                        obj = json.loads(line)
                        cid = obj.get("custom_id")
                        if cid in mapping:
                            body = obj.get("response", {}).get("body", {})
                            choices = body.get("choices") or []
                            content = ""
                            if choices:
                                content = choices[0].get("message", {}).get("content", "")
                            if not mapping[cid].future.done():
                                mapping[cid].future.set_result(content)
                    except Exception as e:
                        logger.error(f"Parse batch output line failed: {e}")
            else:
                err_msg = f"Batch {batch_id} finished with status {status}"
                for req in mapping.values():
                    if not req.future.done():
                        req.future.set_exception(RuntimeError(err_msg))
        except Exception as e:
            for req in mapping.values():
                if not req.future.done():
                    req.future.set_exception(e)

    # ----------------------------------------------------------
    # External status helper
    # ----------------------------------------------------------
    def status_snapshot(self) -> dict[str, Any]:  # pragma: no cover
        """Return a lightweight snapshot of queue & batch statuses."""
        return {
            "queue_len": len(self._queue),
            "batches": dict(self._batches),
            "total_submitted_reqs": self._total_submitted_reqs,
            "total_batches": self._total_batches,
        }

    def add_listener(self, fn):  # pragma: no cover
        if fn not in self._listeners:
            self._listeners.append(fn)

    def remove_listener(self, fn):  # pragma: no cover
        if fn in self._listeners:
            self._listeners.remove(fn)

    def batch_history(self) -> list[dict]:  # pragma: no cover
        return list(self._batch_history)


_global_batch_manager: OpenAIBatchManager | None = None

def init_openai_batch_manager(**kwargs) -> OpenAIBatchManager:
    """Initialize (or return) a global singleton batch manager."""
    global _global_batch_manager
    if _global_batch_manager is None:
        _global_batch_manager = OpenAIBatchManager(**kwargs)
        loop = asyncio.get_event_loop()
        loop.create_task(_global_batch_manager.start())
    return _global_batch_manager

async def openai_queue_completion(
    prompt: str,
    model_name: str | None = None,
    system_prompt: str | None = None,
    history_messages: list[dict] | None = None,
    **kwargs,
) -> str:
    """Queue a completion via Batch API (drop-in alternative to openai_complete).

    WARNING: This is high-latency (minutes) and should not be used for interactive calls.
    """
    mgr = init_openai_batch_manager(model=model_name or kwargs.pop("model", "gpt-4o-mini"), api_key=kwargs.get("api_key"))
    return await mgr.queue_chat_completion(prompt, system_prompt, history_messages, **kwargs)

