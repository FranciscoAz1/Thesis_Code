#!/usr/bin/env python3
"""
Simple test to verify Ollama integration works
"""

import asyncio
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

async def test_ollama():
    print("🧪 Testing Ollama connection...")
    
    try:
        # Create Ollama chat model
        ollama_model = ChatOllama(
            model="qwen2.5:latest",
            base_url="http://localhost:11434",
            temperature=0.1,
        )
        
        print("📡 Testing simple chat with Ollama...")
        
        # Test a simple message
        message = HumanMessage(content="Hello! Please respond with 'Ollama is working correctly'")
        response = await ollama_model.ainvoke([message])
        
        print(f"✅ Ollama response: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ Ollama test failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_ollama())
    if success:
        print("\n✅ Ollama is working correctly!")
    else:
        print("\n❌ Ollama test failed.")