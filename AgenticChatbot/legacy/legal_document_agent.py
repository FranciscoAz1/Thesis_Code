#!/usr/bin/env python3
"""
Legal Document Query Agent using ReAct with Ollama and Weaviate via MCP
Enhanced with BART summarization for better context management
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

# Import for BART summarization
try:
    from transformers import pipeline
    BART_AVAILABLE = True
except ImportError:
    BART_AVAILABLE = False
    print("[WARN] transformers not available - using simple text truncation instead")

load_dotenv()

# Initialize BART summarizer (lazy loaded on first use)
_summarizer = None

def get_summarizer():
    """Get or initialize the BART summarizer (lazy loading)"""
    global _summarizer
    if _summarizer is None and BART_AVAILABLE:
        try:
            _summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception as e:
            print(f"[WARN] Failed to load BART summarizer: {e}")
            return None
    return _summarizer


def summarize_text(text: str, max_length: int = 150, min_length: int = 50, verbose: bool = False) -> str:
    """
    Summarize text using BART if available, otherwise return original text truncated.
    
    Args:
        text: Text to summarize
        max_length: Maximum summary length in tokens
        min_length: Minimum summary length in tokens
        verbose: Whether to print debug info
    
    Returns:
        Summarized text or truncated original if summarization not available
    """
    if not text or len(text) < 50:
        return text  # Text too short to summarize
    
    summarizer = get_summarizer()
    if summarizer is None:
        # Fallback: truncate to first 500 chars
        if verbose:
            print("[*] BART not available, truncating text instead")
        return text[:500] + "..." if len(text) > 500 else text
    
    try:
        if verbose:
            print(f"[*] Summarizing {len(text)} chars to ~{max_length} tokens...")
        
        # BART requires input between 55-1024 tokens
        # Adjust if needed
        if len(text.split()) < 55:
            return text
        
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        summarized = summary[0]['summary_text']
        
        if verbose:
            print(f"[OK] Summarized to {len(summarized)} chars")
        
        return summarized
    except Exception as e:
        if verbose:
            print(f"[WARN] Summarization failed: {e}, returning truncated text")
        return text[:500] + "..." if len(text) > 500 else text


def summarize_document_texts(retrieved_items: List[Dict], verbose: bool = False) -> List[Dict]:
    """
    Summarize the 'text' field of retrieved documents to reduce context size.
    ALWAYS summarizes to keep responses concise.
    
    Args:
        retrieved_items: List of document dictionaries from Weaviate
        verbose: Whether to print debug info
    
    Returns:
        List with 'text' fields summarized
    """
    summarized = []
    for i, item in enumerate(retrieved_items):
        summarized_item = item.copy()
        if isinstance(summarized_item, dict) and 'text' in summarized_item:
            original_text = summarized_item['text']
            if isinstance(original_text, str) and len(original_text) > 0:
                # Summarize the text, except when it is very short
                orig_len = len(original_text)
                summarized_item['text'] = summarize_text(original_text, max_length=200, min_length=40, verbose=verbose)
                new_len = len(summarized_item['text'])
                if verbose:
                    print(f"  Doc {i+1}: {orig_len} chars -> {new_len} chars (reduction: {100*(1-new_len/orig_len):.1f}%)")
        summarized.append(summarized_item)
    return summarized


def format_tool_results_for_llm(tool_result: Dict) -> str:
    """
    Format tool results into a readable format for the LLM with clear file paths.
    
    Args:
        tool_result: Dictionary with Weaviate results
        
    Returns:
        Formatted string for the LLM
    """
    formatted = "=== RESULTADOS DO WEAVIATE ===\n"
    
    try:
        # Handle case where tool_result might be a string already
        if isinstance(tool_result, str):
            import json
            try:
                tool_result = json.loads(tool_result)
            except json.JSONDecodeError:
                formatted += f"\nConteúdo bruto:\n{tool_result}\n"
                return formatted
        
        data = tool_result.get('data', {}) if isinstance(tool_result, dict) else {}
        get_data = data.get('Get', {})
        # Use AdministrativeDocuments collection name
        dataset = get_data.get('AdministrativeDocuments', [])
        
        if isinstance(dataset, list) and len(dataset) > 0:
            for i, doc in enumerate(dataset, 1):
                formatted += f"\n--- Documento {i} ---\n"
                
                # File path
                file_path = doc.get('file_path', 'Caminho não disponível') if isinstance(doc, dict) else 'Caminho não disponível'
                formatted += f"Arquivo: {file_path}\n"
                
                # Text content
                text = doc.get('text', 'Conteúdo não disponível') if isinstance(doc, dict) else 'Conteúdo não disponível'
                formatted += f"Conteúdo:\n{text}\n"
        else:
            formatted += "\nNenhum documento encontrado.\n"
    except Exception as e:
        formatted += f"\nErro ao processar resultados: {e}\n"
    
    formatted += "\n=== FIM DOS RESULTADOS ===\n"
    return formatted


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


def get_mcp_client_config(transport: str = "stdio", verbose: bool = False) -> Dict:
    """
    Generate MCP client configuration based on transport type.
    
    Args:
        transport: One of "stdio", "streamable_http", "sse", or "websocket"
        verbose: Whether to print debug information
    
    Returns:
        Dictionary configuration for MultiServerMCPClient
    """
    # Validate transport type
    valid_transports = ["stdio", "streamable_http", "sse", "websocket"]
    if transport not in valid_transports:
        raise ValueError(f"Unsupported MCP transport: {transport}. Must be one of: {', '.join(repr(t) for t in valid_transports)}")
    
    if transport == "stdio":
        # Use the TypeScript MCP server (better than the old one)
        mcp_server_path = r"C:\Users\Francisco Azeredo\OneDrive\Documents\tecnico\5 ano\tese\Código\mcp-server-weaviate-ts"
        
        # Check if Node.js can run the server
        main_file = os.path.join(mcp_server_path, "dist", "main.js")
        if not os.path.exists(main_file):
            raise RuntimeError(f"MCP server main.js not found at: {main_file}")
        
        if verbose:
            print(f"[OK] MCP TypeScript server found at: {mcp_server_path}")
        
        # Use node to run the TypeScript compiled server
        return {
            "weaviate": {
                "command": "node",
                "args": [main_file],
                "transport": "stdio",
            }
        }
    
    else:
        # For HTTP-based transports (streamable_http, sse, websocket)
        mcp_url = os.getenv("MCP_URL", "http://127.0.0.1:3000/mcp")
        
        if verbose:
            print(f"[OK] Using MCP {transport} transport at: {mcp_url}")
        
        return {
            "weaviate": {
                "url": mcp_url,
                "transport": transport,
            }
        }


async def query_legal_documents(question: str, verbose: bool = False, model: str = "ollama", log_steps: bool = False, transport: str = "stdio") -> Dict:
    """
    Query Portuguese legal documents using ReAct agent with Weaviate database.
    
    Args:
        question: The question to ask about legal documents
        verbose: Whether to print debug information
        model: LLM to use ("ollama" or "openai")
        log_steps: Whether to log step-by-step execution
        transport: MCP transport type ("stdio" or "http")
        
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
        
        # Check Weaviate connectivity
        if verbose:
            print("\n[*] Checking Weaviate connectivity...")
        weaviate_host = check_weaviate_connection_silent()
        
        if not weaviate_host:
            return {
                'answer': '',
                'context': [],
                'success': False,
                'error': "Weaviate database is not accessible"
            }
        
        # Get MCP client configuration based on transport
        if verbose:
            print(f"\n[CONFIG] Configuring MCP client with transport: {transport}")
        try:
            mcp_config = get_mcp_client_config(transport=transport, verbose=verbose)
        except RuntimeError as e:
            return {
                'answer': '',
                'context': [],
                'success': False,
                'error': str(e)
            }
        
        # Create MCP client
        if verbose:
            print("[*] Creating MCP client...")
        client = MultiServerMCPClient(mcp_config)
        
        # Get tools and resources
        if verbose:
            print("[*] Getting tools from MCP server...")
        tools = await client.get_tools()
        if verbose:
            print(f"[OK] Successfully loaded {len(tools)} tools from MCP server")
        
        # Get available resources to discover collections and schemas
        if verbose:
            print("\n[*] Getting available resources...")
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
            print(f"\n[*] Available collections: {collections}")
            for coll, props in collection_info.items():
                print(f"  {coll}: {props}")

        if model == "ollama":
            # Create Ollama model
            if verbose:
                print("\n[*] Creating Ollama model...")
            model = ChatOllama(
                model="qwen2.5:latest",
                base_url="http://localhost:11434",
                temperature=0.3,  # Reduced from 0.9 for faster, more deterministic responses
            )
        # Create ReAct agent
        if verbose:
            print("[*] Creating ReAct agent with Ollama...")
        # Create OpenAI model
        if model == "openai":
            if verbose:
                print("\n[*] Creating OpenAI model...")
            OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            model = ChatOpenAI(
                model="gpt-5-nano",
                temperature=0.9,
                api_key=SecretStr(OPENAI_API_KEY) if OPENAI_API_KEY else None,
            )

        memory = MemorySaver()
        agent = create_react_agent(model, tools, checkpointer=memory)
        if verbose:
            print("[OK] ReAct agent created successfully!")
        
        # Create system prompt with discovered collections and properties
        system_prompt = f"""Você é um assistente que responde perguntas usando um banco de dados Weaviate.

INSTRUÇÕES OBRIGATÓRIAS:
1. SEMPRE use weaviate-query PRIMEIRO
2. Use EXATAMENTE targetProperties: ["text", "file_path"]
3. DEPOIS responda CONCISAMENTE em 1-2 frases
4. Inclua o arquivo (file_path) real dos resultados

FORMATO EXATO DA FERRAMENTA:
{{"name": "weaviate-query", "arguments": {{"query": "[sua pergunta]", "collection": "AdministrativeDocuments", "targetProperties": ["text", "file_path"]}}}}

Não altere targetProperties. Não responda sem usar a ferramenta.
"""
        if verbose:
            print(f"\n[*] Starting ReAct query for: {question}")
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]
        
        retrieved_contexts = []
        # Provide a stable thread_id so MemorySaver can track checkpoints for this conversation
        thread_id = f"legal-docs-{uuid.uuid4()}"
        max_steps = 10  # Reduced from 50 for faster responses
        tool_calls_count = 0
        max_tool_calls = 2  # Reduced from 3 for faster responses
        
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
                    # Prevent infinite tool call loops
                    tool_calls_count += 1
                    if tool_calls_count > max_tool_calls:
                        if verbose:
                            print(f"[WARN] Reached maximum tool calls ({max_tool_calls}), stopping to prevent infinite loops.")
                        # Return the best answer we have so far
                        for msg in reversed(messages):
                            if type(msg).__name__ == "AIMessage" and msg.content.strip():
                                result = {
                                    'answer': msg.content,
                                    'context': retrieved_contexts,
                                    'success': True,
                                    'error': None
                                }
                                if log_steps:
                                    result['messages'] = messages_log
                                return result
                    
                    tool_name = tool_call.get('name')
                    tool_args = tool_call.get('arguments', tool_call.get('args'))  # Try both keys
                    tool_obj = next((t for t in tools if t.name == tool_name), None)
                    if tool_obj:
                        if verbose:
                            print(f"Executing tool '{tool_name}' with args: {tool_args}")
                        tool_result = await tool_obj.ainvoke(tool_args)
                        if verbose:
                            print(f"Tool result (first 200 chars): {str(tool_result)[:200]}...")
                        
                        # Summarize document texts to reduce context size
                        summarized_result = tool_result
                        if verbose:
                            print(f"[DEBUG-A] Tool result type: {type(tool_result).__name__}")
                            if isinstance(tool_result, str):
                                print(f"[DEBUG-A] Tool result (first 300 chars): {tool_result[:300]}")
                        if isinstance(tool_result, dict) and 'data' in tool_result:
                            try:
                                # Extract texts from Weaviate response structure
                                # Response uses: {"data": {"Get": {"AdministrativeDocuments": [...]}}
                                data = tool_result.get('data', {})
                                get_data = data.get('Get', {})
                                dataset = get_data.get('AdministrativeDocuments', [])
                                
                                if verbose:
                                    print(f"[DEBUG] Found {len(dataset)} documents in AdministrativeDocuments")
                                
                                if isinstance(dataset, list) and len(dataset) > 0:
                                    if verbose:
                                        print(f"[*] Summarizing {len(dataset)} document(s)...")
                                    
                                    # Summarize each document
                                    summarized_docs = summarize_document_texts(dataset, verbose=verbose)
                                    
                                    # Update the result with summarized documents
                                    summarized_result = dict(tool_result)
                                    summarized_result['data']['Get']['AdministrativeDocuments'] = summarized_docs
                            except Exception as e:
                                if verbose:
                                    print(f"[WARN] Summarization failed: {e}, using original result")
                                summarized_result = tool_result
                        
                        # Store the context (with summarized content)
                        retrieved_contexts.append({
                            'tool': tool_name,
                            'args': tool_args,
                            'result': summarized_result,
                            'original_result': tool_result  # Keep original for reference
                        })
                        
                        # Create tool message with formatted results for LLM readability
                        # Use formatted version so LLM can easily extract file paths
                        formatted_result = format_tool_results_for_llm(summarized_result)
                        tool_msg = ToolMessage(
                            content=formatted_result,
                            tool_call_id=tool_call.get('id', f'tool_{step}'),
                            name=tool_name
                        )
                        # Append tool message so LLM can see and reference the results with file paths
                        messages.append(tool_msg)
                continue  # Go to next step to process tool result
            else:
                # No tool calls - this should be the final answer
                content = last_ai_msg.content.strip()
                
                # Check if response is concise (< 1000 chars is good, > 5000 is too verbose)
                is_too_verbose = len(content) > 5000
                
                if content and len(content) > 5 and not is_too_verbose:  # Even shorter minimum (was 10)
                    if verbose:
                        print(f"Agent provided concise answer ({len(content)} chars), ending process.")
                    result = {
                        'answer': last_ai_msg.content,
                        'context': retrieved_contexts,
                        'success': True,
                        'error': None
                    }
                    if log_steps:
                        result['messages'] = messages_log
                    return result
                
                # Response too verbose or too short - keep iterating
                if is_too_verbose:
                    if verbose:
                        print(f"Agent response too verbose ({len(content)} chars), asking for concise answer.")
                    # Add a message asking for concise answer
                    messages.append(HumanMessage(content="Responda de forma concisa e direta, em poucas frases."))
                else:
                    if verbose:
                        print("Agent message is too short or empty, continuing to next step.")
                continue  # Continue to next step
        
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
        import traceback
        error_details = traceback.format_exc()
        if verbose:
            print(f"\n[ERROR] Exception occurred:")
            print(error_details)
        
        error_msg = f"{type(e).__name__}: {str(e)}"
        
        return {
            'answer': '',
            'context': [],
            'success': False,
            'error': error_msg
        }


async def test_agent():
    """Test the agent with a sample question"""
    question = "Quais são as responsabilidades do empregador segundo a Norma Técnica nº 123/2023 analisada no parecer da Direcção-Geral do Trabalho?"
    
    print("[*] Testing Legal Document Agent")
    print("=" * 80)
    print(f"Question: {question}")
    
    # Use transport from environment or default to stdio
    # Map common names to valid transports
    env_transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
    transport_map = {
        "http": "streamable_http",
        "sse": "sse",
        "websocket": "websocket",
        "stdio": "stdio",
        "streamable_http": "streamable_http"
    }
    transport = transport_map.get(env_transport, "stdio")
    
    print(f"Transport: {transport} (set MCP_TRANSPORT env var to override)")
    print("=" * 80)
    
    result = await query_legal_documents(question, verbose=True, transport=transport)
    
    print("\n[RESULT]:")
    print("=" * 80)
    print(f"Success: {result['success']}")
    if result['error']:
        print(f"Error: {result['error']}")
    
    print(f"\n[ANSWER]:")
    print("-" * 40)
    print(result['answer'])
    print("-" * 40)
    
    print(f"\n[CONTEXT] ({len(result['context'])} items):")
    for i, ctx in enumerate(result['context'], 1):
        print(f"\n{i}. Tool: {ctx['tool']}")
        print(f"   Args: {ctx['args']}")
        result_data = ctx['result']
        
        # Extract file paths from Weaviate response structure
        file_paths = []
        if isinstance(result_data, dict):
            data = result_data.get('data', {})
            get_data = data.get('Get', {})
            dataset = get_data.get('AdministrativeDocuments', [])
            if isinstance(dataset, list):
                file_paths = [item.get('file_path', 'N/A') for item in dataset if isinstance(item, dict)]
        
        if file_paths:
            print(f"   Retrieved files: {file_paths}")
        print(f"   Result (first 300 chars): {str(result_data)[:300]}...")
    
    print("\n" + "=" * 80)
    return result


if __name__ == "__main__":
    print("[*] Starting Legal Document Agent test...")
    print("\n[CONFIG] Transport Configuration:")
    print("   Default: stdio (local MCP server)")
    print("   Available transports: stdio, streamable_http, sse, websocket")
    print("   To use HTTP-based transport, set environment variables:")
    print("     - MCP_TRANSPORT=streamable_http  (or sse, websocket)")
    print("     - MCP_URL=http://127.0.0.1:3000/mcp (optional, default)")
    print()
    
    result = asyncio.run(test_agent())
    
    if result['success']:
        print("[OK] Test completed successfully!")
    else:
        print("[FAIL] Test failed.")