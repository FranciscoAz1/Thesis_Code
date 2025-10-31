#!/usr/bin/env python3
"""
Standalone test script for MCP ReAct Agent on Windows

This script runs outside of Ju        print("🤖 Creating ReAct agent with Ollama...")
        
        # Create Ollama chat model
        ollama_model = ChatOllama(
            model="qwen2.5:latest",
            base_url="http://localhost:11434",
            temperature=0.1,
        )
        
        agent = create_react_agent(
            ollama_model,
            tools
        )
        print("✅ ReAct agent created successfully!")to avoid Windows subprocess limitations.
Run this from the command line: python test_mcp_standalone.py
"""

import os
import asyncio
import sys
import requests
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama

def check_weaviate_connection():
    """Check if Weaviate is running and accessible"""
    weaviate_urls = [
        "http://localhost:8080/v1/meta",
        "http://host.docker.internal:8080/v1/meta",
        "http://127.0.0.1:8080/v1/meta"
    ]
    
    for url in weaviate_urls:
        try:
            print(f"  Checking Weaviate at: {url}")
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"  ✅ Weaviate is running at: {url}")
                return url.replace("/v1/meta", "").replace("http://", "")
            else:
                print(f"  ❌ Weaviate returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Connection failed: {e}")
    
    print("  ⚠️ No accessible Weaviate instance found")
    return None

async def main():
    """Main async function to test MCP ReAct Agent"""
    
    # Set ProactorEventLoop for Windows subprocess support
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        print("Set WindowsProactorEventLoopPolicy for subprocess support")
    
    # Check if the MCP server executable exists
    mcp_server_path = r"C:\Users\Francisco Azeredo\OneDrive\Documents\tecnico\5 ano\tese\Código\mcp-server-weaviate\client\mcp-server.exe"
    
    if not os.path.exists(mcp_server_path):
        print(f"❌ MCP server executable not found at: {mcp_server_path}")
        print("Please check the path or build the MCP server first.")
        return
    
    print(f"✅ MCP server found at: {mcp_server_path}")
    
    # Check Weaviate connectivity
    print("\n🔗 Checking Weaviate connectivity...")
    weaviate_host = check_weaviate_connection()
    
    try:
        # Create MCP client
        print("\n🔧 Creating MCP client...")
        
        # Try different configurations to debug the connection issue
        mcp_configs = [
            {
                "name": "weaviate-debug",
                "config": {
                    "command": mcp_server_path,
                    "args": ["-log-level=debug", "-log-output=stderr"] + 
                           ([f"-weaviate-host={weaviate_host}"] if weaviate_host else []),
                    "transport": "stdio",
                }
            },
            {
                "name": "weaviate-default", 
                "config": {
                    "command": mcp_server_path,
                    "args": ["-log-level=debug"],
                    "transport": "stdio",
                }
            }
        ]
        
        client = None
        for config_info in mcp_configs:
            try:
                print(f"  Trying configuration: {config_info['name']}")
                client = MultiServerMCPClient({
                    "weaviate": config_info["config"]
                })
                break
            except Exception as e:
                print(f"  ❌ Failed with {config_info['name']}: {e}")
                continue
        
        if client is None:
            print("❌ All MCP client configurations failed")
            return False
        
        # Get tools from MCP server
        print("📡 Getting tools from MCP server...")
        tools = await client.get_tools()
        print(f"✅ Successfully loaded {len(tools)} tools from MCP server")
        
        # List available tools
        print("\n📋 Available tools:")
        for i, tool in enumerate(tools, 1):
            print(f"  {i}. {tool.name}: {tool.description}")
        
        # Create ReAct agent
        print("\n🤖 Creating ReAct agent...")
        agent = create_react_agent(
            "anthropic:claude-3-5-sonnet-20241022",
            tools
        )
        print("✅ ReAct agent created successfully!")
        
        # Test with a question in Portuguese about Portuguese labor law
        print("\n🧪 Testing with Weaviate question...")
        test_question = "Quais são as responsabilidades do empregador segundo a Norma Técnica nº 123/2023 analisada no parecer da Direcção-Geral do Trabalho?"
        
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": test_question}]}
        )
        
        print(f"\n📝 Response to: {test_question}")
        print("=" * 80)
        print(response)
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting MCP ReAct Agent test...")
    print("=" * 80)
    
    # Run the async main function
    success = asyncio.run(main())
    
    print("\n" + "=" * 80)
    if success:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed. Check the error messages above.")
    
    print("\nTo run this script:")
    print("1. Open a terminal/command prompt")
    print("2. Navigate to the project directory")
    print("3. Activate your virtual environment if needed")
    print("4. Run: python test_mcp_standalone.py")