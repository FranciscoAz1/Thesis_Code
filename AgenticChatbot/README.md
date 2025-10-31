# AgenticChatbot - Intelligent Document Retrieval Agent

## Overview

This module implements an autonomous, conversational agent that uses the **ReAct (Reasoning + Acting)** pattern to answer questions about legal documents stored in Weaviate. The agent leverages OpenAI's Agents SDK and communicates with Weaviate through the Model Context Protocol (MCP).

## Key Features

- **Autonomous Reasoning**: Uses GPT-4/5 to plan and execute multi-step retrieval strategies
- **Session Persistence**: Maintains conversation context across multiple turns using SQLite
- **MCP Integration**: Structured communication with Weaviate database through MCP tools
- **Retry Logic**: Automatic retries with exponential backoff for robustness
- **Error Handling**: Comprehensive error detection and recovery
- **Session Logging**: Complete audit trail of agent interactions

## Architecture

```
User Question
    ↓
OpenAI Agent (GPT-4/5)
    ↓
MCP Tools (weaviate-query, weaviate-follow-ref)
    ↓
MCP Server (TypeScript/Go)
    ↓
Weaviate Database
```

## Directory Structure

```
AgenticChatbot/
├── legal_document_agent_openai.py  # ⭐ Main agent implementation
├── README.md                        # This file
├── .env                            # Environment configuration
│
├── scripts/                        # Executable scripts
│   ├── benchmark_openAI_1.py       # Single-question benchmark
│   ├── benchmark_openAI_2.py       # Batch benchmark with Q&A dataset
│   ├── benchmark_example_usage.py  # Example benchmark setup
│   ├── benchmark_weaviate.ipynb    # Benchmark notebook
│   ├── demo.py                     # Simple demo script
│   ├── example_usage.py            # Usage examples
│   ├── test_mcp_standalone.py      # MCP connection testing
│   ├── test_ollama_simple.py       # Ollama connectivity test
│   └── merge_datasets.py           # Dataset consolidation
│
├── datasets/                       # Q&A datasets
│   ├── merged_qa_dataset.json      # Combined Q&A dataset
│   ├── merged_qa_dataset1.json     # Alternative dataset
│   ├── qa_dataset.json             # Original Q&A pairs
│   ├── qa_dataset300.json          # Extended 300-question dataset
│   └── query_set_Lihua.json        # LiHua benchmark queries
│
├── results/                        # Benchmark results
│   ├── benchmark_results_openai.csv     # Main benchmark results
│   ├── benchmark_results_openai20.csv   # 20-question subset
│   ├── comparison_report_1.csv          # Single configuration
│   ├── comparison_report_20.csv         # 20-question comparison
│   ├── comparison_report_300.csv        # 300-question comparison
│   └── comparison_report_mixed.csv      # Mixed dataset comparison
│
├── logs/                           # Output logs
│   ├── output.txt                  # Main output log
│   ├── output25.txt                # Dated output
│   ├── output.26.txt               # Dated output
│   ├── output.log                  # Application log
│   ├── output_ts.log               # TypeScript MCP log
│   └── output_ts2.log              # Alternative TS log
│
├── sessions/                       # Persistent session storage
│   ├── agent_sessions.jsonl        # Session metadata
│   ├── agent_session_messages.jsonl # Complete message logs
│   ├── analyze_session.py          # Session analysis tools
│   └── ref_graph.dot               # Reference graph visualization
│
├── legacy/                         # Old/deprecated files
│   ├── legal_document_agent.py     # Legacy agent version
│   └── legal_document_agent copy 2.py  # Older backup
│
└── .vscode/                        # VS Code configuration
    ├── launch.json                 # Debug configuration
    └── settings.json               # Editor settings
```

## Setup

### Prerequisites

1. **OpenAI API Key**:
   ```powershell
   $env:OPENAI_API_KEY="sk-your-key-here"
   ```

2. **Weaviate Running**:
   ```powershell
   cd ..\weaviate
   docker-compose up -d
   ```

3. **MCP Server Running**:
   ```powershell
   cd ..\mcp-server-weaviate-ts
   npm start
   ```

### Installation

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install agents openai python-dotenv requests
```

### Configuration

Create `.env` file:
```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4  # or gpt-5
MCP_TRANSPORT=http  # or stdio
MCP_HTTP_URL=http://127.0.0.1:3000/mcp
```

## Usage

### Basic Usage

```python
import asyncio
from legal_document_agent_openai import openai_agent

async def main():
    result = await openai_agent(
        "What are the requirements for contract approval?",
        verbose=True
    )
    print(result.final_output)

asyncio.run(main())
```

### Multi-Turn Conversation

```python
import asyncio
from legal_document_agent_openai import get_agent_and_session, try_openai_agent_with_retries

async def conversation():
    agent, session = await get_agent_and_session(verbose=True)
    
    try:
        # First question
        result1 = await try_openai_agent_with_retries(
            agent, session,
            "What documents mention contract approval?",
            verbose=True, retries=5, delay=5.0
        )
        print("Answer 1:", result1.final_output)
        
        # Follow-up (context preserved)
        result2 = await try_openai_agent_with_retries(
            agent, session,
            "What are the specific requirements mentioned?",
            verbose=True, retries=5, delay=5.0
        )
        print("Answer 2:", result2.final_output)
        
    finally:
        for srv in agent.mcp_servers:
            await srv.cleanup()

asyncio.run(conversation())
```

### Running Benchmarks

```powershell
# Single question benchmark
python scripts/benchmark_openAI_1.py

# Full Q&A dataset benchmark
python scripts/benchmark_openAI_2.py

# Interactive notebook
jupyter notebook scripts/benchmark_weaviate.ipynb
```

## Agent Configuration

### System Prompt

The agent uses a structured system prompt that:
- Defines the task (retrieve and answer)
- Specifies tools (weaviate-query, weaviate-follow-ref)
- Enforces Portuguese language output
- Requires source citations
- Uses "FINAL ANSWER:" marker for completion

### Model Settings

```python
ModelSettings(
    extra_args={"service_tier": "flex"}  # Cost optimization
)
```

### Retry Strategy

- **Max Retries**: 5
- **Initial Delay**: 5 seconds
- **Backoff**: Exponential (2^attempt)
- **Jitter**: Random 0-10% of backoff time

### Tool Configuration

```python
mcp_server = mcp.MCPServerStreamableHttp(
    params={
        "url": "http://127.0.0.1:3000/mcp",
        "timeout": 120,
    },
    name="weaviate",
    max_retry_attempts=5,
)
```

## Session Management

### Session Storage

Sessions are persisted in SQLite:
```
sessions/doc_conversation.db
```

### Session Records

Two JSONL files track activity:
- **`agent_sessions.jsonl`**: High-level session metadata
- **`agent_session_messages.jsonl`**: Complete message logs

Example session record:
```json
{
  "question": "What are the approval requirements?",
  "start_time": "2025-10-31T12:34:56Z",
  "result": "...",
  "saved_at": "2025-10-31T12:35:12Z"
}
```

### Session Analysis

```python
from sessions.analyze_session import load_sessions, analyze_patterns

sessions = load_sessions("sessions/agent_sessions.jsonl")
analyze_patterns(sessions)
```

## Benchmarking

### Metrics Collected

1. **Latency**: Response time per question
2. **Accuracy**: Correctness of retrieved documents
3. **Completeness**: Coverage of expected information
4. **Citation Quality**: Presence and relevance of sources
5. **Conversation Context**: Ability to maintain multi-turn context

### Benchmark Output

Results saved to CSV with columns:
- `question`: Input question
- `expected_answer`: Ground truth
- `agent_answer`: Agent response
- `latency_ms`: Response time
- `correct_docs`: Whether correct documents retrieved
- `has_citation`: Whether sources cited
- `session_id`: Session identifier

### Comparison Reports

Compare different configurations:
```powershell
python compare_results.py benchmark_results_openai.csv benchmark_results_openai20.csv
```

## Advanced Features

### Custom Tool Selection

Filter available MCP tools:
```python
agent = Agent(
    # ... other params
    tools=[],  # Empty = use all MCP tools
    tool_filter=lambda tool: "weaviate" in tool.name.lower()
)
```

### Structured Content

Parse tool results as structured data:
```python
mcp_server = mcp.MCPServerStreamableHttp(
    # ... params
    use_structured_content=True  # Parse JSON tool results
)
```

### Transport Options

#### HTTP (Recommended)
```python
mcp_server = mcp.MCPServerStreamableHttp(
    params={"url": "http://127.0.0.1:3000/mcp"}
)
```

#### STDIO
```python
mcp_server = mcp.MCPServerStdio(
    params={
        "command": "node",
        "args": ["../mcp-server-weaviate-ts/dist/main.js"]
    }
)
```

## Troubleshooting

### Agent Returns Errors

Check agent response for OpenAI error patterns:
```python
if "Error code:" in result.final_output or "request_id" in result.final_output:
    # Retry with backoff
```

### MCP Connection Fails

1. Verify MCP server is running:
   ```powershell
   curl http://127.0.0.1:3000/health
   ```

2. Check Weaviate connectivity:
   ```powershell
   python test_mcp_standalone.py
   ```

3. Review logs:
   ```powershell
   cat ../mcp-server-weaviate-ts/logs/app.log
   ```

### Session Not Persisting

Ensure sessions directory exists:
```python
from pathlib import Path
sessions_dir = Path(__file__).parent / "sessions"
sessions_dir.mkdir(exist_ok=True)
```

### Windows Event Loop Issues

For subprocess support on Windows:
```python
import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
```

## Performance Considerations

### Average Latencies

- **Simple Question**: 2-4 seconds
- **Multi-Step Reasoning**: 5-8 seconds
- **Follow-up Questions**: 1-3 seconds (context cached)

### Optimization Tips

1. **Use HTTP transport**: Faster than stdio
2. **Limit result count**: Reduce token usage
3. **Cache embeddings**: Reuse for similar queries
4. **Service tier "flex"**: Lower costs, acceptable latency
5. **Batch questions**: Reuse session across queries

## Best Practices

1. **Always use retry logic** for production deployments
2. **Log sessions** for debugging and analysis
3. **Monitor token usage** to control costs
4. **Validate tool results** before using in prompts
5. **Use verbose mode** during development
6. **Clean up MCP servers** in finally blocks
7. **Set reasonable timeouts** (120s for MCP, 900s for queries)

## Related Components

- **MCP Server**: `../mcp-server-weaviate-ts/` (recommended) or `../mcp-server-weaviate/`
- **Weaviate Setup**: `../weaviate/weaviate_manager.py`
- **Benchmark Framework**: `../gerador_documentos_gpt_azure (1)/gerador_documentos_gpt_azure/`

## References

- [OpenAI Agents SDK Documentation](https://github.com/openai/agents-sdk)
- [Model Context Protocol Specification](https://modelcontextprotocol.io)
- [Weaviate Python Client](https://weaviate.io/developers/weaviate/client-libraries/python)

---

**Last Updated**: October 31, 2025
