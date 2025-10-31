#!/usr/bin/env python3
"""
Simple Demo: Legal Document Query Function

This demonstrates the clean function interface:
    query_legal_documents(question: str, verbose: bool = False) -> dict

Returns:
    {
        'answer': str,           # The agent's final answer
        'context': list,         # List of retrieved documents  
        'success': bool,         # Whether the query was successful
        'error': str or None     # Error message if any
    }
"""

import asyncio
from legal_document_agent import query_legal_documents


async def demo():
    """Simple demo of the legal document query function"""
    
    print("🚀 Legal Document Query Agent Demo")
    print("="*60)
    
    # Example question
    question = "Quais são as responsabilidades do empregador na segurança do trabalho?"
    
    print(f"❓ Question: {question}")
    print("-"*60)
    
    # Call the function (with verbose=False for clean output)
    result = await query_legal_documents(question, verbose=False)
    
    # Display results
    print(f"✅ Success: {result['success']}")
    
    if result['success']:
        print(f"\n💬 Answer:")
        print(result['answer'])
        
        print(f"\n📚 Retrieved {len(result['context'])} document(s)")
        for i, ctx in enumerate(result['context'], 1):
            query_used = ctx['args'].get('query', 'N/A')
            print(f"  {i}. Search: '{query_used}'")
    else:
        print(f"\n❌ Error: {result['error']}")
    
    print("\n" + "="*60)
    print("Demo completed!")
    
    return result


if __name__ == "__main__":
    # Run the demo
    result = asyncio.run(demo())
    
    # Show the function interface
    print("\n" + "🔧 FUNCTION INTERFACE:")
    print("="*60)
    print("from legal_document_agent import query_legal_documents")
    print("")
    print("# Usage:")
    print("result = await query_legal_documents(")
    print("    question='Your legal question in Portuguese',")
    print("    verbose=False  # Set to True for debug info")
    print(")")
    print("")
    print("# Returns dict with keys:")
    print("# - answer: str (final answer)")
    print("# - context: list (retrieved documents)")  
    print("# - success: bool")
    print("# - error: str or None")
    print("="*60)