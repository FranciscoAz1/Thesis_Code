#!/usr/bin/env python3
"""
Legal Document Query Agent using ReAct with Ollama and Weaviate via MCP
"""

import os
from ssl import OP_ENABLE_MIDDLEBOX_COMPAT
import sys
import asyncio
import requests
import json
import re
import uuid
from typing import Dict, List, Optional
from pydantic import SecretStr

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
load_dotenv()

def check_weaviate_connection_silent() -> Optional[str]:
    """Check if Weaviate is running and accessible (silent version)"""
    weaviate_urls = [
        "http://localhost:8080/v1/meta",
        "http://host.docker.internal:8080/v1/meta",
        "http://127.0.0.1:8080/v1/meta"
    ]
    
    for url in weaviate_urls:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return url.replace("/v1/meta", "").replace("http://", "")
        except requests.exceptions.RequestException:
            pass
    
    return None


async def query_legal_documents(question: str, verbose: bool = False, model: str = "ollama", log_steps: bool = False) -> Dict:
    """
    Query Portuguese legal documents using ReAct agent with Weaviate database.
    
    Args:
        question: The question to ask about legal documents
        verbose: Whether to print debug information
        
    Returns:
        dict: {
            'answer': str,           # The agent's final answer
            'context': list,         # List of retrieved documents
            'success': bool,         # Whether the query was successful
            'error': str or None     # Error message if any
        }
    """
    try:
        messages_log = [] if log_steps else None
        # Set ProactorEventLoop for Windows subprocess support
        if sys.platform.startswith('win'):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            if verbose:
                print("Set WindowsProactorEventLoop for subprocess support")
        
        # Check if the MCP server executable exists
        mcp_server_path = r"C:\Users\Francisco Azeredo\OneDrive\Documents\tecnico\5 ano\tese\Código\mcp-server-weaviate\client\mcp-server.exe"
        
        if not os.path.exists(mcp_server_path):
            return {
                'answer': '',
                'context': [],
                'success': False,
                'error': f"MCP server executable not found at: {mcp_server_path}"
            }
        
        if verbose:
            print(f"✅ MCP server found at: {mcp_server_path}")
        
        # Check Weaviate connectivity
        if verbose:
            print("\n🔗 Checking Weaviate connectivity...")
        weaviate_host = check_weaviate_connection_silent()
        
        if not weaviate_host:
            return {
                'answer': '',
                'context': [],
                'success': False,
                'error': "Weaviate database is not accessible"
            }
        
        # Create MCP client
        if verbose:
            print("\n🔧 Creating MCP client...")
        client = MultiServerMCPClient({
            "weaviate": {
                "command": mcp_server_path,
                "args": ["-log-level=debug", "-log-output=stderr"] + 
                       ([f"-weaviate-host={weaviate_host}"] if weaviate_host else []),
                "transport": "stdio",
            }
        })
        
        # Get tools and resources
        if verbose:
            print("📡 Getting tools from MCP server...")
        tools = await client.get_tools()
        if verbose:
            print(f"✅ Successfully loaded {len(tools)} tools from MCP server")
        
        # Get available resources to discover collections and schemas
        if verbose:
            print("\n📚 Getting available resources...")
        resources = await client.get_resources("weaviate")
        
        # Extract collection information
        collections = []
        collection_info = {}
        
        for resource in resources:
            try:
                data = resource.data
                # Normalize data to string if possible
                if isinstance(data, bytes):
                    data = data.decode("utf-8", errors="ignore")
                elif data is not None and not isinstance(data, str):
                    data = str(data)

                if isinstance(data, str) and "Properties for collection" in data:
                    # Parse: "Properties for collection 'Dataset': text, file_path"
                    collection_start = data.find("'") + 1
                    collection_end = data.find("'", collection_start)
                    collection_name = data[collection_start:collection_end]
                    
                    properties_start = data.find(": ") + 2
                    properties_text = data[properties_start:]
                    properties = [prop.strip() for prop in properties_text.split(", ")]
                    
                    collections.append(collection_name)
                    collection_info[collection_name] = properties
            except Exception as e:
                if verbose:
                    print(f"  - Error accessing resource: {e}")
        
        if verbose:
            print(f"\n📂 Available collections: {collections}")
            for coll, props in collection_info.items():
                print(f"  {coll}: {props}")

        if model == "ollama":
            # Create Ollama model
            if verbose:
                print("\n🤖 Creating Ollama model...")
            model = ChatOllama(
                model="qwen2.5:latest",
                base_url="http://localhost:11434",
                temperature=0.9,
            )
        # Create ReAct agent
        if verbose:
            print("🔧 Creating ReAct agent with Ollama...")
        # Create OpenAI model
        if model == "openai":
            if verbose:
                print("\n🤖 Creating OpenAI model...")
            OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            model = ChatOpenAI(
                model="gpt-5-nano",
                temperature=0.9,
                api_key=SecretStr(OPENAI_API_KEY) if OPENAI_API_KEY else None,
            )

        memory = MemorySaver()
        agent = create_react_agent(model, tools, checkpointer=memory)
        if verbose:
            print("✅ ReAct agent created successfully!")
        
        # Create system prompt with discovered collections and properties
        system_prompt = f"""You are a helpful assistant with access to a Weaviate database.

Goal:
- Retrieve relevant information from Weaviate and provide direct, concise answers.
- Always ground answers in retrieved content and cite source file paths; include a short quote when helpful.

Instructions:
- Use the weaviate-query tool (and refine queries if needed) to fetch relevant items
- Don't ask for clarifications; do your best with the information available
- Parse tool results to extract the key text and file_path
- When you know the final answer, prefix it exactly with: "FINAL ANSWER: "

Tool parameters:
- collection: "Dataset"
- targetProperties: ["text", "file_path"]
"""
        if verbose:
            print(f"\n🧠 Starting ReAct query for: {question}")
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]
        
        retrieved_contexts = []
        # Provide a stable thread_id so MemorySaver can track checkpoints for this conversation
        thread_id = f"legal-docs-{uuid.uuid4()}"
        max_steps = 50
        
        for step in range(max_steps):
            # Pass the required configurable keys for the checkpointer (MemorySaver)
            response = await agent.ainvoke(
                {"messages": messages},
                config={
                    "configurable": {
                        "thread_id": thread_id
                    }
                }
            )
            new_messages = response.get('messages', [])
            if not new_messages:
                break
            if log_steps:
                step_log = f"--- Step {step + 1} ---\n"
                step_log += f"New messages received: {len(new_messages)}\n"
                for i, msg in enumerate(new_messages):
                    step_log += f"--- Message {i + 1} ---\n{type(msg).__name__}: {msg.content[:200]}...\n"
                step_log += f"Total messages so far: {len(messages) + len(new_messages)}\n"
                if messages_log is not None:
                    messages_log.append(step_log)
            if verbose:
                print(f"\n--- Step {step + 1} ---")
                print(f"New messages received: {len(new_messages)}")
                # print messages
                for i, msg in enumerate(new_messages):
                    print(f"\n--- Message {i + 1} ---")
                    print(f"{type(msg).__name__}: {msg.content[:200]}...")
                print(f"Total messages so far: {len(messages) + len(new_messages)}")
            # Find the last AI message that's new
            last_ai_msg = None
            prev_nonempty_ai_msg = None
            for msg in reversed(new_messages):
                if type(msg).__name__ == "AIMessage" and msg not in messages:
                    if last_ai_msg is None:
                        last_ai_msg = msg
                    if msg.content.strip() and msg.content.strip() != '...':
                        prev_nonempty_ai_msg = msg
            # If any AIMessage in this batch is just '...', break and return the last non-empty AI message (from new or previous messages)
            def is_empty_or_nonalpha(content):
                stripped = content.strip()
                return not stripped or not any(c.isalpha() for c in stripped)

            found_ellipsis_or_empty = any(
                type(msg).__name__ == "AIMessage" and (
                    msg.content.strip() == '...'
                    or is_empty_or_nonalpha(msg.content)
                )
                for msg in new_messages
            )
            # if found_ellipsis_or_empty:
            if False:
                if verbose:
                    print("Agent returned '...', empty, or non-alphabetic message in this batch, treating previous non-empty AI message as final answer.")
                # Prefer the last non-empty AI message in this batch, else search previous messages
                if prev_nonempty_ai_msg:
                    result = {
                        'answer': prev_nonempty_ai_msg.content,
                        'context': retrieved_contexts,
                        'success': True,
                        'error': None
                    }
                    if log_steps:
                        result['messages'] = messages_log
                    return result
                # Fallback: search all previous messages
                for msg in reversed(messages):
                    if type(msg).__name__ == "AIMessage" and msg.content.strip() and not is_empty_or_nonalpha(msg.content):
                        result = {
                            'answer': msg.content,
                            'context': retrieved_contexts,
                            'success': True,
                            'error': None
                        }
                        if log_steps:
                            result['messages'] = messages_log
                        return result
                if verbose:
                    print("No previous non-empty AI message found, ending process.")
                break

            if not last_ai_msg:
                if verbose:
                    print("No new AI message found, ending process.")
                break

            # Add this message to our conversation
            messages.append(last_ai_msg)
            
            # Check for tool calls
            tool_calls = getattr(last_ai_msg, 'tool_calls', None)
            
            # If no structured tool calls, try parsing from content
            if not tool_calls and last_ai_msg.content and '<tool_call>' in last_ai_msg.content:
                # Extract tool call from XML-like format
                tool_call_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', last_ai_msg.content, re.DOTALL)
                if tool_call_match:
                    try:
                        tool_call_json = json.loads(tool_call_match.group(1))
                        tool_calls = [tool_call_json]
                        if verbose:
                            print(f"Parsed tool call from content: {tool_calls}")
                    except json.JSONDecodeError as e:
                        if verbose:
                            print(f"Failed to parse tool call JSON: {e}")
            
            if tool_calls:
                if verbose:
                    print(f"Processing {len(tool_calls)} tool calls...")
                for tool_call in tool_calls:
                    tool_name = tool_call.get('name')
                    tool_args = tool_call.get('arguments', tool_call.get('args'))  # Try both keys
                    tool_obj = next((t for t in tools if t.name == tool_name), None)
                    if tool_obj:
                        if verbose:
                            print(f"Executing tool '{tool_name}' with args: {tool_args}")
                        tool_result = await tool_obj.ainvoke(tool_args)
                        if verbose:
                            print(f"Tool result (first 200 chars): {str(tool_result)[:200]}...")
                        
                        # Store the context
                        retrieved_contexts.append({
                            'tool': tool_name,
                            'args': tool_args,
                            'result': tool_result
                        })
                        
                        # Create tool message
                        tool_msg = ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call.get('id', f'tool_{step}'),
                            name=tool_name
                        )
                        # Don't append tool messages
                        # messages.append(tool_msg)
                continue  # Go to next step to process tool result
            else:
                # Check if the agent explicitly marked a final answer
                content = last_ai_msg.content.strip()
                if content.startswith('FINAL ANSWER:'):
                    if verbose:
                        print("Agent provided explicit FINAL ANSWER, ending process.")
                    result = {
                        'answer': content[len('FINAL ANSWER:'):].strip(),
                        'context': retrieved_contexts,
                        'success': True,
                        'error': None
                    }
                    if log_steps:
                        result['messages'] = messages_log
                    return result
                # Otherwise check for proper bullet-point responsibilities
                if content.startswith('-') and ('responsabilidades' in content.lower() or 'empregador' in content.lower()):
                    if verbose:
                        print("Agent provided proper answer format, ending process.")
                    result = {
                        'answer': last_ai_msg.content,
                        'context': retrieved_contexts,
                        'success': True,
                        'error': None
                    }
                    if log_steps:
                        result['messages'] = messages_log
                    return result
                # Not a final answer, keep iterating
                if verbose:
                    print("Agent answer doesn't look like proper final answer, continuing to next step.")
                continue  # Continue to next step instead of stopping
        
        # If we reach here, we hit max steps
        if verbose:
            print(f"Reached maximum steps ({max_steps}) without final answer.")
        result = {
            'answer': "Sorry, I couldn't find a complete answer within the step limit.",
            'context': retrieved_contexts,
            'success': False,
            'error': "Reached maximum steps without final answer"
        }
        if log_steps:
            result['messages'] = messages_log
        return result
        
    except Exception as e:
        return {
            'answer': '',
            'context': [],
            'success': False,
            'error': f"{type(e).__name__}: {e}"
        }


async def test_agent():
    """Test the agent with a sample question"""
    question = "Quais são as responsabilidades do empregador segundo a Norma Técnica nº 123/2023 analisada no parecer da Direcção-Geral do Trabalho?"
    
    print("🚀 Testing Legal Document Agent")
    print("=" * 80)
    print(f"Question: {question}")
    print("=" * 80)
    
    result = await query_legal_documents(question, verbose=True)
    
    print("\n📋 RESULT:")
    print("=" * 80)
    print(f"Success: {result['success']}")
    if result['error']:
        print(f"Error: {result['error']}")
    
    print(f"\n🎯 Answer:")
    print("-" * 40)
    print(result['answer'])
    print("-" * 40)
    
    print(f"\n📚 Context ({len(result['context'])} items):")
    for i, ctx in enumerate(result['context'], 1):
        print(f"\n{i}. Tool: {ctx['tool']}")
        print(f"   Args: {ctx['args']}")
        result_data = ctx['result']
        
        # Extract file paths from Weaviate response structure
        file_paths = []
        if isinstance(result_data, dict):
            data = result_data.get('data', {})
            get_data = data.get('Get', {})
            dataset = get_data.get('Dataset', [])
            if isinstance(dataset, list):
                file_paths = [item.get('file_path', 'N/A') for item in dataset if isinstance(item, dict)]
        
        if file_paths:
            print(f"   Retrieved files: {file_paths}")
        print(f"   Result (first 300 chars): {str(result_data)[:300]}...")
    
    print("\n" + "=" * 80)
    return result


if __name__ == "__main__":
    print("🚀 Starting Legal Document Agent test...")
    result = asyncio.run(test_agent())
    
    if result['success']:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed.")