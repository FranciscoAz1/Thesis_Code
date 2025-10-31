"""
Example implementation of metadata filtering for JsonKVStorage

This shows how to implement the filter_by_metadata method for the JSON-based storage.
Similar implementations would be needed for other storage backends.
"""

import json
import os
from typing import Any, Dict, List

class JsonKVStorageWithMetadata:
    """
    Example enhancement to JsonKVStorage to support metadata filtering
    This would be integrated into the actual JsonKVStorage implementation
    """
    
    def __init__(self, namespace: str, global_config: dict):
        self.namespace = namespace
        self.global_config = global_config
        self.working_dir = global_config.get("working_dir", "./minirag_cache")
        self.file_path = os.path.join(self.working_dir, f"kv_store_{namespace}.json")
    
    async def filter_by_metadata(self, metadata_filter: Dict[str, Any]) -> List[str]:
        """
        Filter documents by metadata criteria
        
        Args:
            metadata_filter: Dictionary containing metadata criteria to match
            
        Returns:
            List of document IDs that match the criteria
        """
        try:
            if not os.path.exists(self.file_path):
                return []
            
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            matching_ids = []
            
            for doc_id, doc_data in data.items():
                doc_metadata = doc_data.get('metadata', {})
                
                # Check if all filter criteria match
                if self._matches_filter(doc_metadata, metadata_filter):
                    matching_ids.append(doc_id)
            
            return matching_ids
        
        except Exception as e:
            print(f"Error filtering by metadata: {e}")
            return []
    
    def _matches_filter(self, doc_metadata: Dict[str, Any], filter_criteria: Dict[str, Any]) -> bool:
        """
        Check if document metadata matches filter criteria
        
        Args:
            doc_metadata: The document's metadata
            filter_criteria: The filter criteria to match against
            
        Returns:
            True if all criteria match, False otherwise
        """
        for key, expected_value in filter_criteria.items():
            if key not in doc_metadata:
                return False
            
            doc_value = doc_metadata[key]
            
            # Handle different types of matching
            if isinstance(expected_value, list) and isinstance(doc_value, list):
                # Check if any item in expected_value exists in doc_value
                if not any(item in doc_value for item in expected_value):
                    return False
            elif isinstance(expected_value, list):
                # Check if doc_value is in expected_value list
                if doc_value not in expected_value:
                    return False
            elif isinstance(doc_value, list):
                # Check if expected_value is in doc_value list
                if expected_value not in doc_value:
                    return False
            else:
                # Direct comparison
                if doc_value != expected_value:
                    return False
        
        return True

# Example usage patterns for metadata filtering:

metadata_examples = {
    "exact_match": {
        "domain": "healthcare",
        "status": "published"
    },
    
    "list_contains": {
        "keywords": ["AI", "machine learning"],  # Documents containing any of these keywords
        "authors": ["Dr. Smith"]
    },
    
    "date_range": {
        "date": "2024-01-15"  # Could be extended to support date ranges
    },
    
    "complex_filter": {
        "domain": ["healthcare", "finance"],  # Documents from either domain
        "confidence_score": 0.8,  # Exact match for numeric values
        "tags": ["peer-reviewed"]  # Must contain this tag
    }
}

# For SQL-based storage backends, the implementation would use SQL queries:
sql_metadata_filter_examples = {
    "postgresql": '''
        SELECT id FROM documents 
        WHERE workspace = $1 
        AND metadata @> $2::jsonb
    ''',
    
    "oracle": '''
        SELECT id FROM documents 
        WHERE workspace = :workspace 
        AND JSON_EXISTS(metadata, '$.domain' EQUALS :domain)
    '''
}
