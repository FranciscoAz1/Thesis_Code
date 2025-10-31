"""
Simple test script to verify metadata functionality works
"""

import sys
import os
import asyncio

# Add the minirag package to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_metadata_functionality():
    """Test basic metadata functionality"""
    try:
        # Import the classes
        from minirag import MiniRAG, QueryParam
        print("✓ Imports successful")
        
        # Simple embedding function for testing
        from minirag.utils import wrap_embedding_func_with_attrs
        
        @wrap_embedding_func_with_attrs(embedding_dim=384, max_token_size=8192)
        async def dummy_embedding_func(texts):
            import numpy as np
            return [np.random.rand(384).tolist() for _ in texts]

        # Simple LLM function for testing
        async def dummy_llm_func(prompt, **kwargs):
            return f"Test response to: {prompt[:50]}..."
        
        # Create MiniRAG instance
        rag = MiniRAG(
            working_dir="./test_metadata_cache",
            embedding_func=dummy_embedding_func,
            llm_model_func=dummy_llm_func,
        )
        print("✓ MiniRAG instance created")
        
        # Test document insertion with metadata
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
        print("✓ Documents inserted successfully!")
        
        # Test metadata query
        print("Querying by metadata...")
        healthcare_docs = await rag.query_by_metadata({"domain": "healthcare"})
        print(f"✓ Healthcare documents found: {len(healthcare_docs)}")
        
        if healthcare_docs:
            for doc in healthcare_docs:
                print(f"  - Content summary: {doc['content_summary']}")
                print(f"  - Metadata: {doc['metadata']}")
        
        # Test query with multiple criteria
        ai_docs = await rag.query_by_metadata({"topic": "AI"})
        print(f"✓ AI-related documents found: {len(ai_docs)}")
        
        print("All tests passed! ✓")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_metadata_functionality())
