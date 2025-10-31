#!/usr/bin/env python3
"""
Legal Document Query Agent using OpenAI SDK with Weaviate via MCP (ReAct-style)
"""

import os
import sys
import re
import json
import uuid
import asyncio
#!/usr/bin/env python3
"""
Legal Document Query Agent using OpenAI Agents SDK + MCP (Weaviate)
Session-based ReAct-style via Runner and SQLiteSession.
"""

import os
import sys
import asyncio
import requests
from typing import Optional, List
from dotenv import load_dotenv

from agents import Agent, ModelSettings, Runner, SQLiteSession, mcp, RunConfig

load_dotenv()
from datetime import datetime, timezone
from pathlib import Path
import traceback
import random


def _ensure_sessions_dir() -> Path:
    base = Path(__file__).resolve().parent
    sessions_dir = base / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir


def _save_session_record(record: dict, filename: str = "agent_sessions.jsonl") -> None:
    """Append a JSON-line record. Best-effort; swallow errors."""
    try:
        import json
        sessions_dir = _ensure_sessions_dir()
        fpath = sessions_dir / filename
        record = dict(record)
        record.setdefault("saved_at", datetime.now(timezone.utc).isoformat())
        with fpath.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        # Do not raise from the logger - persistence is best-effort
        traceback.print_exc()


def _save_agent_snapshot(filename_prefix: str = "agent_snapshot") -> None:
    """Save a copy of this file for provenance (timestamped).

    Best-effort; swallow errors.
    """
    try:
        sessions_dir = _ensure_sessions_dir()
        src = Path(__file__).resolve()
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = sessions_dir / f"{filename_prefix}_{ts}.py"
        with src.open("r", encoding="utf-8") as fr, dest.open("w", encoding="utf-8") as fw:
            fw.write(fr.read())
    except Exception:
        traceback.print_exc()

def check_weaviate_connection_silent() -> Optional[str]:
    """Check if Weaviate is running and accessible (silent version)"""
    weaviate_urls = [
        "http://localhost:8080/v1/meta",
        "http://host.docker.internal:8080/v1/meta",
        "http://127.0.0.1:8080/v1/meta",
    ]
    for url in weaviate_urls:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return url.replace("/v1/meta", "").replace("http://", "")
        except requests.exceptions.RequestException:
            pass
    return None


def build_system_prompt(default_collection: str | None = None, default_props: List[str] | None = None) -> str:
    if not default_props:
        default_props = ["text", "file_path"]
    if default_collection is None:
        return ("You are a helpful assistant with access to a Weaviate database via tools.\n\n"
        "Goal:\n- Retrieve relevant information and provide direct, concise answers.\n"
        "- Always ground answers in retrieved content and cite source file paths; include a short quote when helpful.\n\n"
        "Instructions:\n- When you need data, request the 'weaviate-query' tool.\n"
        "- Do not ask for clarification; do your best with the information available.\n"
        "- Parse tool results to extract the key text and file_path.\n"
        "- When you know the final answer, prefix it exactly with: \"FINAL ANSWER: \"\n\n"
        )
    return (
        "You are a helpful assistant with access to a Weaviate database via tools.\n\n"
        "Goal:\n- Retrieve relevant information and provide direct, concise answers in Portuguese.\n"
        "- Always ground answers in retrieved content and cite source file paths; include a short quote when helpful.\n\n"
        "Instructions:\n- When you need data, request the 'weaviate-query' tool.\n"
        "- Do not ask for clarification; do your best with the information available.\n"
        "- Parse tool results to extract the key text and file_path.\n"
        "- When you know the final answer, prefix it exactly with: \"FINAL ANSWER: \"\n\n"
        f"Default tool parameters:\n- collection: \"{default_collection}\"\n- targetProperties: {default_props}\n"
    )


async def get_agent_and_session(verbose: bool = False, transport: str = "http") -> tuple[Agent, SQLiteSession]:
    # Windows subprocess event loop (MCP stdio)
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        if verbose:
            print("Set WindowsProactorEventLoop for subprocess support")
    # Check Weaviate connectivity
    if verbose:
        print("\n🔗 Checking Weaviate connectivity...")
    weaviate_host = check_weaviate_connection_silent()
    if not weaviate_host:
        raise RuntimeError("Weaviate database is not accessible")

    # OpenAI setup
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", "gpt-5")

    # Allow environment override of transport and URL
    env_transport = os.getenv("MCP_TRANSPORT")
    if env_transport:
        transport = env_transport.lower()

    if transport == "stdio":
            # MCP server path
        # mcp_server_path = r"C:\Users\Francisco Azeredo\OneDrive\Documents\tecnico\5 ano\tese\Código\mcp-server-weaviate\client\mcp-server.exe"
        mcp_server_path = r"C:\Users\Francisco Azeredo\OneDrive\Documents\tecnico\5 ano\tese\Código\mcp-server-weaviate-ts\dist\main.js"
        # if not os.path.exists(mcp_server_path):
        #     raise RuntimeError(f"MCP server executable not found at: {mcp_server_path}")
        # if verbose:
        #     print(f"✅ MCP server found at: {mcp_server_path}")
        # Create MCP stdio server
        mcp_args = [mcp_server_path]
        # if weaviate_host:
        #     mcp_args.append(f"-weaviate-host={weaviate_host}")
        stdio_params: mcp.MCPServerStdioParams = {
            "command": "node",
            "args": mcp_args,
        }
        mcp_server = mcp.MCPServerStdio(
            params=stdio_params,
            name="weaviate",
            use_structured_content=False,
        )
    elif transport == "http":
        # Create MCP HTTP server
        http_params: mcp.MCPServerStreamableHttpParams = {
            "url": "http://127.0.0.1:3000/mcp",
            "timeout": 120,
        }
        mcp_server = mcp.MCPServerStreamableHttp(
            params=http_params,
            name="weaviate",
            use_structured_content=False,
            max_retry_attempts=5,
        )
    else:
        raise ValueError(f"Unsupported MCP transport: {transport}")

    # Connect MCP server before using it
    try:
        await mcp_server.connect()
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to MCP HTTP server. Set MCP_HTTP_URL to the correct endpoint or set MCP_STDIO_PATH and MCP_TRANSPORT=stdio. {e}"
        ) from e

    # Agent configuration
    system_prompt = build_system_prompt(default_collection=None, default_props=None)
    system_prompt = "You are a helpful assistant to a journaling company with access to a weaviate database via tools.\n\nGoal:\n- Retrieve relevant information and provide direct, concise answers.\n- Always ground answers in retrieved content; include a short quote when helpful.\n\nInstructions:\n- When you need data, request the 'weaviate-origin' tool, and then follow up with 'weaviate-follow-ref' to navigate the database in a structured way.\n- Do not ask for clarification; do your best with the information available.\n- Parse tool results to extract the key text.\n"
    agent = Agent(
        name="DocSearchAssistant",
        instructions=system_prompt,
        model=model_name,
        model_settings=ModelSettings(
            # OpenAI Responses API accepts service_tier as a top-level arg; pass via extra_args
            extra_args={"service_tier": "flex"}
        ),
        tools=[],
        mcp_servers=[mcp_server],
    )

    # Session (persistent conversation)
    # Use a file-backed SQLite DB so the session survives process restarts and
    # can be inspected after the run. Stored under sessions/doc_conversation.db
    try:
        sessions_dir = _ensure_sessions_dir()
        db_path = sessions_dir / "doc_conversation.db"
        session = SQLiteSession("doc_conversation", db_path=str(db_path))
    except Exception:
        print("⚠️ Warning: Could not create file-backed SQLiteSession; using in-memory only.")
        # Fallback to in-memory session if file-backed DB can't be created
        session = SQLiteSession("doc_conversation")
    return agent, session

async def openai_agent(question: str, verbose: bool = False):
    agent, session = await get_agent_and_session(verbose=verbose)
    try:
        if verbose:
            print("\n🚀 Asking:", question)
        # Per-call override (optional): you can also pass service_tier via RunConfig
        result = await Runner.run(
            agent,
            question,
            session=session,
            max_turns=20,
            run_config=RunConfig(
                model_settings=ModelSettings(extra_args={"service_tier": "flex"})
            ),
        )
        if verbose:
            print("\n🎯 Answer:")
            print(result.final_output)
    finally:
        # Cleanup MCP servers
        for srv in agent.mcp_servers:
            try:
                await srv.cleanup()
            except Exception:
                pass
    return result

async def run_agent(agent, session, question: str, verbose: bool = False):
    try:
        if verbose:
            print("\n🚀 Asking:", question)
        result = await Runner.run(agent, question, session=session, max_turns=20)
        if verbose:
            print("\n🎯 Answer:")
            print(result.final_output)
        # Inspect the returned result for common error patterns and raise so retry logic can act.
        # final = getattr(result, "final_output", None)
        # if isinstance(final, str):
        #     if ("Error code:" in final) or ("\"error\"" in final) or ("request_id" in final) or ("server_error" in final):
        #         # Raise an exception with the agent's error text so the retry wrapper can catch it
        #         raise RuntimeError(f"Agent returned error: {final}")

        return result
    except Exception:
        # Print traceback for diagnostics, then re-raise so caller retry logic can handle it
        traceback.print_exc()
        raise
    
async def try_openai_agent_with_retries(agent, session, question: str, verbose: bool, retries: int, delay: float):
    """Try calling run_agent with retries and exponential backoff.

    Returns the successful result or raises the last exception if all attempts fail.
    """
    attempts = retries + 1  # first try + retries
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return await run_agent(agent, session, question, verbose)
        except Exception as e:
            last_error = e
            msg = str(e)
            # Try to extract OpenAI request id if present in the message
            req_id = None
            try:
                m = re.search(r"(req_[0-9a-fA-F]+)", msg)
                if m:
                    req_id = m.group(1)
            except Exception:
                pass

            print(f"Agent error on attempt {attempt}/{attempts}: {msg}")
            if req_id:
                print(f"OpenAI request_id: {req_id}")
            traceback.print_exc()

            if attempt < attempts:
                backoff = delay * (2 ** (attempt - 1))
                jitter = random.uniform(0, min(1.0, backoff * 0.1))
                wait = backoff + jitter
                print(f"Retrying in {wait:.1f} seconds...")
                try:
                    await asyncio.sleep(wait)
                except Exception:
                    pass
    print(f"All {attempts} attempts failed for question: {question}")
    if last_error:
        raise last_error
    return None
def main():
    verbose = True
    async def run_flow():
        agent, session = await get_agent_and_session(verbose=verbose)
        try:
            # First turn
            question = (
                "How is the war on Russia and Ukraine?"
            )
            print("\n🚀 Asking:", question)
            result = await try_openai_agent_with_retries(agent, session,question, verbose, retries=5, delay=5.0)
            print("\n🎯 Answer:")
            print(result)

            # Save session record + snapshot for provenance
            try:
                _save_session_record({
                    "question": question,
                    "start_time": datetime.now(timezone.utc).isoformat(),
                    "result": result if isinstance(result, dict) else str(result),
                })
                # _save_agent_snapshot("legal_openai_agent")
                # Also export the SQLiteSession contents (messages) for inspection
                try:
                    session_items = await session.get_items()
                    _save_session_record({
                        "question": question,
                        "session_items": session_items,
                        "exported_at": datetime.now(timezone.utc).isoformat(),
                    }, filename="agent_session_messages.jsonl")
                except Exception:
                    pass
            except Exception:
                pass

            # Second turn (context preserved)
            followup ="What is the latest war crime?"

            print("\n🚀 Asking:", followup)
            result2 = await try_openai_agent_with_retries(agent, session, followup, verbose, retries=5, delay=5.0)
            print("\n🎯 Answer:")
            print(result2)

            # Save session record + snapshot for provenance
            try:
                _save_session_record({
                    "question": followup,
                    "start_time": datetime.now(timezone.utc).isoformat(),
                    "result": result2 if isinstance(result2, dict) else str(result2),
                })
                # _save_agent_snapshot("legal_openai_agent")
                try:
                    session_items = await session.get_items()
                    _save_session_record({
                        "question": followup,
                        "session_items": session_items,
                        "exported_at": datetime.now(timezone.utc).isoformat(),
                    }, filename="agent_session_messages.jsonl")
                except Exception:
                    pass
            except Exception:
                pass

            # Third turn (use async runner to avoid event loop conflicts)
            followup2 = "Who reported?"
            print("\n🚀 Asking:", followup2)
            result3 = await try_openai_agent_with_retries(agent, session, followup2, verbose, retries=5, delay=5.0)

            
            print("\n🎯 Answer:")
            print(result3)

            # Save session record + snapshot for provenance
            try:
                _save_session_record({
                    "question": followup2,
                    "start_time": datetime.now(timezone.utc).isoformat(),
                    "result": result3 if isinstance(result3, dict) else str(result3),
                })
                _save_agent_snapshot("legal_openai_agent")
                try:
                    session_items = await session.get_items()
                    _save_session_record({
                        "question": followup2,
                        "session_items": session_items,
                        "exported_at": datetime.now(timezone.utc).isoformat(),
                    }, filename="agent_session_messages.jsonl")
                except Exception:
                    pass
            except Exception:
                pass
        finally:
            # Cleanup MCP servers
            for srv in agent.mcp_servers:
                try:
                    await srv.cleanup()
                except Exception:
                    pass

    asyncio.run(run_flow())


if __name__ == "__main__":
    main()
