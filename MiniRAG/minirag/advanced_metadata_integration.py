"""
Advanced Metadata Integration for RAG Queries

This file demonstrates how to enhance the existing query functions 
to incorporate metadata filtering for more precise document retrieval.
"""

from typing import Dict, Any, List
from minirag.base import QueryParam, BaseKVStorage, BaseVectorStorage, TextChunkSchema

async def enhanced_chunk_retrieval_with_metadata(
    query: str,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    metadata_filter: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Enhanced chunk retrieval that respects metadata filters
    
    Args:
        query: The search query
        chunks_vdb: Vector database for chunks
        text_chunks_db: Key-value storage for text chunks
        query_param: Query parameters
        metadata_filter: Optional metadata filter
        
    Returns:
        List of filtered and ranked chunks
    """
    # Step 1: Get initial chunk candidates from vector search
    initial_results = await chunks_vdb.query(query, top_k=query_param.top_k * 2)  # Get more to allow for filtering
    
    # Step 2: Apply metadata filtering if specified
    if metadata_filter:
        filtered_chunks = []
        
        for result in initial_results:
            chunk_id = result["id"]
            chunk_data = await text_chunks_db.get_by_id(chunk_id)
            
            if chunk_data and _matches_metadata_filter(chunk_data.get("metadata", {}), metadata_filter):
                filtered_chunks.append({
                    **result,
                    "chunk_data": chunk_data
                })
        
        # Trim to requested top_k after filtering
        filtered_chunks = filtered_chunks[:query_param.top_k]
        return filtered_chunks
    else:
        # No metadata filtering, return original results
        enhanced_results = []
        for result in initial_results[:query_param.top_k]:
            chunk_data = await text_chunks_db.get_by_id(result["id"])
            enhanced_results.append({
                **result,
                "chunk_data": chunk_data
            })
        return enhanced_results

def _matches_metadata_filter(doc_metadata: Dict[str, Any], filter_criteria: Dict[str, Any]) -> bool:
    """Helper function to check metadata matching"""
    for key, expected_value in filter_criteria.items():
        if key not in doc_metadata:
            return False
        
        doc_value = doc_metadata[key]
        
        # Handle different matching patterns
        if isinstance(expected_value, list) and isinstance(doc_value, list):
            # List intersection matching
            if not any(item in doc_value for item in expected_value):
                return False
        elif isinstance(expected_value, list):
            # Value in list matching
            if doc_value not in expected_value:
                return False
        elif isinstance(doc_value, list):
            # Expected value in document's list
            if expected_value not in doc_value:
                return False
        else:
            # Exact matching
            if doc_value != expected_value:
                return False
    
    return True

# Example of enhanced query function that uses metadata
async def metadata_aware_minirag_query(
    query: str,
    knowledge_graph_inst,
    entities_vdb: BaseVectorStorage,
    entity_name_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    embedder,
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """
    Enhanced minirag query with metadata awareness
    
    This function demonstrates how to modify the existing query pipeline
    to incorporate metadata filtering at the chunk retrieval level.
    """
    # Extract metadata filter from query parameters
    metadata_filter = getattr(query_param, 'metadata_filter', {})
    
    # If metadata filter is provided, pre-filter chunks
    if metadata_filter:
        # Get chunks that match metadata criteria
        filtered_chunks = await enhanced_chunk_retrieval_with_metadata(
            query, chunks_vdb, text_chunks_db, query_param, metadata_filter
        )
        
        # Extract just the chunk IDs for use in existing pipeline
        filtered_chunk_ids = [chunk["id"] for chunk in filtered_chunks]
        
        return f"Query processed with metadata filter: {metadata_filter}. Found {len(filtered_chunks)} matching chunks."
    
    else:
        # No metadata filtering, use standard minirag query
        # ... (call original minirag_query function)
        return "Standard query without metadata filtering"

# Usage examples:

query_examples = {
    "domain_specific": {
        "query": "What are the latest developments in cancer treatment?",
        "metadata_filter": {"domain": "healthcare", "topic": "oncology"},
        "description": "Search only in healthcare documents about oncology"
    },
    
    "source_specific": {
        "query": "Financial market trends in 2024",
        "metadata_filter": {"source": "bloomberg", "date_range": "2024"},
        "description": "Search only in Bloomberg articles from 2024"
    },
    
    "multi_criteria": {
        "query": "AI applications in education",
        "metadata_filter": {
            "domain": ["education", "technology"],
            "keywords": ["AI", "artificial intelligence"],
            "confidence_score": {"$gte": 0.8}  # Could support MongoDB-style operators
        },
        "description": "Complex metadata filtering with multiple criteria"
    }
}

# Integration with existing query modes:
def create_metadata_aware_query_param(
    mode: str = "mini",
    metadata_filter: Dict[str, Any] = None,
    **kwargs
) -> QueryParam:
    """
    Helper function to create QueryParam with metadata filtering
    """
    param = QueryParam(mode=mode, **kwargs)
    if metadata_filter:
        param.metadata_filter = metadata_filter
    return param
