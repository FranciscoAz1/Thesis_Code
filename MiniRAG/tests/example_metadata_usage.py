"""
Example: Using Metadata in MiniRAG Pipeline

This example demonstrates how to:
1. Insert documents with metadata
2. Query documents using metadata filters
3. Use metadata in RAG queries for better context
"""

import asyncio
from minirag import MiniRAG, QueryParam
from minirag.llm.openai import openai_complete_if_cache
from minirag.utils import EmbeddingFunc

# Example embedding function (replace with your preferred embedding service)
async def simple_embedding_func(texts):
    # This is a placeholder - replace with actual embedding service
    import numpy as np
    return [np.random.rand(384).tolist() for _ in texts]

async def main():
    # Initialize MiniRAG
    rag = MiniRAG(
        working_dir="./minirag_metadata_example",
        embedding_func=simple_embedding_func,
        llm_model_func=openai_complete_if_cache,  # Replace with your LLM
    )

    # Example 1: Insert documents with metadata
    documents = [
        "Artificial intelligence is transforming healthcare by enabling faster diagnosis.",
        "Machine learning algorithms are being used in financial fraud detection.",
        "Deep learning models have revolutionized computer vision applications."
    ]
    
    metadata_list = [
        {
            "domain": "healthcare", 
            "topic": "AI", 
            "source": "medical_journal",
            "date": "2024-01-15",
            "keywords": ["AI", "healthcare", "diagnosis"]
        },
        {
            "domain": "finance", 
            "topic": "ML", 
            "source": "fintech_report",
            "date": "2024-02-10",
            "keywords": ["ML", "finance", "fraud"]
        },
        {
            "domain": "computer_vision", 
            "topic": "deep_learning", 
            "source": "tech_blog",
            "date": "2024-03-05",
            "keywords": ["deep learning", "computer vision", "AI"]
        }
    ]
    
    print("Inserting documents with metadata...")
    await rag.ainsert(documents, metadata=metadata_list)
    print("Documents inserted successfully!")

    # Example 2: Query documents by metadata
    print("\n=== Querying by Metadata ===")
    
    # Query documents from healthcare domain
    healthcare_docs = await rag.query_by_metadata({"domain": "healthcare"})
    print(f"Healthcare documents found: {len(healthcare_docs)}")
    for doc in healthcare_docs:
        print(f"  - {doc['content_summary']}")
        print(f"    Metadata: {doc['metadata']}")

    # Query documents with specific keywords
    ai_docs = await rag.query_by_metadata({"keywords": ["AI"]})  # Note: this may need adjustment based on implementation
    print(f"\nAI-related documents found: {len(ai_docs)}")

    # Example 3: Enhanced RAG query with metadata context
    print("\n=== Enhanced RAG Query ===")
    
    # Create query parameter with metadata filter
    query_param = QueryParam(
        mode="mini",
        metadata_filter={"domain": "healthcare"},  # Focus on healthcare documents
        top_k=20
    )
    
    # Perform RAG query
    query = "How is AI being used in medical applications?"
    response = await rag.aquery(query, query_param)
    print(f"Query: {query}")
    print(f"Response: {response}")

    # Example 4: Filter by multiple metadata criteria
    print("\n=== Multi-criteria Metadata Query ===")
    
    complex_filter = {
        "domain": "finance",
        "source": "fintech_report"
    }
    
    finance_docs = await rag.query_by_metadata(complex_filter)
    print(f"Finance documents from fintech reports: {len(finance_docs)}")
    for doc in finance_docs:
        print(f"  - Content: {doc['content_summary']}")
        print(f"    Date: {doc['metadata'].get('date', 'N/A')}")

def run_example():
    """Run the metadata example"""
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error running example: {e}")
        print("Make sure you have configured your embedding and LLM functions correctly.")

if __name__ == "__main__":
    run_example()
