#!/usr/bin/env python3
"""
Example usage of the Legal Document Query Agent
"""

import asyncio

import ollama
from legal_document_agent import query_legal_documents


async def example_usage():
    """Example of how to use the legal document agent function"""
    
    # Example questions
    questions = [
        "Quais são as responsabilidades do empregador segundo a Norma Técnica nº 123/2023?",
        "O que diz a legislação sobre teletrabalho?",
        "Quais são os requisitos para equipamentos de proteção individual?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"QUESTION {i}: {question}")
        print('='*80)
        
        # Query the legal documents
        result = await query_legal_documents(question, verbose=False)
        
        # Display results
        if result['success']:
            print(f"✅ SUCCESS")
            print(f"\n📋 ANSWER:")
            print("-" * 40)
            print(result['answer'])
            print("-" * 40)
            
            print(f"\n📚 RETRIEVED CONTEXT ({len(result['context'])} documents):")
            for j, ctx in enumerate(result['context'], 1):
                print(f"\n{j}. Search Query: {ctx['args'].get('query', 'N/A')}")
                # Extract document info from result
                import json
                try:
                    data = json.loads(ctx['result'])
                    docs = data.get('data', {}).get('Get', {}).get('Dataset', [])
                    for k, doc in enumerate(docs, 1):
                        file_path = doc.get('file_path', 'Unknown file')
                        text_preview = doc.get('text', '')[:200] + '...'
                        print(f"   Document {k}: {file_path}")
                        print(f"   Preview: {text_preview}")
                except:
                    print(f"   Raw result: {str(ctx['result'])[:200]}...")
        else:
            print(f"❌ FAILED: {result['error']}")
        
        print("\n" + "="*80)


async def single_question_example():
    """Example with a single question"""
    question = "Quais são as responsabilidades do empregador segundo normas de segurança no trabalho?"
    
    print("🚀 Legal Document Agent - Single Query Example")
    print("="*80)
    print(f"Question: {question}")
    print("="*80)
    
    # Query with verbose output for debugging
    result = await query_legal_documents(question, verbose=True, model="openai")
    
    print("\n📊 FINAL RESULT:")
    print("="*80)
    
    if result['success']:
        print("✅ Status: SUCCESS")
        print(f"\n🎯 Answer:")
        print(result['answer'])
        
        print(f"\n📚 Context: {len(result['context'])} items retrieved")
        for i, ctx in enumerate(result['context'], 1):
            print(f"  {i}. Tool: {ctx['tool']}")
            print(f"     Query: {ctx['args'].get('query', 'N/A')}")
            
            # Extract and display retrieved files
            import json
            try:
                data = json.loads(ctx['result'])
                docs = data.get('data', {}).get('Get', {}).get('Dataset', [])
                if docs:
                    file_paths = [doc.get('file_path', 'Unknown file') for doc in docs]
                    print(f"     Retrieved files: {file_paths}")
            except:
                pass
    else:
        print(f"❌ Status: FAILED")
        print(f"Error: {result['error']}")
    
    print("="*80)
    return result


if __name__ == "__main__":
    # Run the single question example
    print("Starting Legal Document Agent Example...")
    result = asyncio.run(single_question_example())
    
    if result['success']:
        print("\n✅ Example completed successfully!")
        print(f"Answer length: {len(result['answer'])} characters")
        print(f"Context items: {len(result['context'])}")
    else:
        print("\n❌ Example failed.")