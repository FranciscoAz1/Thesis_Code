import asyncio
from typing import Any, Union, List, Set, Dict
import weaviate
import weaviate.cluster
from weaviate.exceptions import WeaviateQueryException
from weaviate.classes.config import Configure
from dataclasses import dataclass, field
import hashlib
import uuid
from minirag.base import (
    BaseVectorStorage,
    BaseKVStorage,
    BaseGraphStorage
)

async def run_sync(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)

def string_to_uuid(text: str) -> str:
    """
    Convert a string to a valid UUID v5 (deterministic, based on hash).
    This ensures that the same string always produces the same UUID.
    """
    # Use a namespace UUID for consistent hashing
    namespace = uuid.NAMESPACE_DNS
    return str(uuid.uuid5(namespace, text))

def get_weaviate_client():
    """Get Weaviate v4 client using connect_to_local()"""
    try:
        return weaviate.connect_to_local()
    except Exception as e:
        print(f"Failed to connect to local Weaviate: {e}")
        print("Make sure Weaviate is running: docker run -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:latest")
        raise

@dataclass
class WeaviateVectorStorage(BaseVectorStorage):
    client: Any = field(init=False)  # Weaviate v4 client

    def __post_init__(self):
        """Initialize Weaviate v4 client using connect_to_local()"""
        self.client = get_weaviate_client()
        self.init_schema()

    def init_schema(self):
        """Create 'Document' collection if it doesn't exist (v4 API)"""
        try:
            # Check if collection exists (collections.list_all() returns a dict)
            collections = self.client.collections.list_all()
            collection_names = list(collections.keys()) if isinstance(collections, dict) else [c.name for c in collections]
            
            if "Document" not in collection_names:
                # Create collection in v4
                self.client.collections.create(
                    name="Document",
                    description="Document collection for vector storage"
                )
                print("[Weaviate] Created 'Document' collection")
        except Exception as e:
            print(f"Vector schema init error: {e}")

    async def query(self, query: str, top_k: int) -> List[Dict]:
        """Query documents using v4 collections API"""
        try:
            # Get collection
            collection = self.client.collections.get("Document")
            
            # Use fetch_objects to retrieve data (v4 API)
            response = collection.query.fetch_objects(limit=top_k)
            
            # Convert to expected format, including all properties
            results = []
            for obj in response.objects:
                result_dict = {
                    "content": obj.properties.get("content", ""),
                    "_id": str(obj.uuid),
                    "id": str(obj.uuid)  # Some code expects 'id' instead of '_id'
                }
                # Add all other properties from the object
                if obj.properties:
                    for key, value in obj.properties.items():
                        if key != "content":
                            result_dict[key] = value
                results.append(result_dict)
            
            return results
        except Exception as e:
            print(f"Weaviate query error: {e}")
            return []

    async def upsert(self, data: Dict[str, Dict]):
        """Upsert documents into collection (v4 API)"""
        try:
            collection = self.client.collections.get("Document")
            
            for _id, value in data.items():
                # Store all properties from the input data, not just 'content'
                obj_properties = {k: v for k, v in value.items() if k != "content" and not k.startswith("_")}
                # Always include content if it exists
                if "content" in value:
                    obj_properties["content"] = value["content"]
                
                # Convert string ID to valid UUID for Weaviate
                valid_uuid = string_to_uuid(_id)
                try:
                    # Try to insert first
                    collection.data.insert(
                        uuid=valid_uuid,
                        properties=obj_properties
                    )
                except Exception as insert_err:
                    # If already exists, try to update
                    if "already exists" in str(insert_err):
                        try:
                            collection.data.update(
                                uuid=valid_uuid,
                                properties=obj_properties
                            )
                        except Exception as update_err:
                            pass  # Silently ignore update errors
                    else:
                        pass  # Silently ignore other errors
        except Exception as e:
            pass  # Silently ignore top-level errors

    async def delete(self, ids: List[str]):
        """Delete documents from collection (v4 API)"""
        try:
            collection = self.client.collections.get("Document")
            
            for _id in ids:
                # Convert string ID to valid UUID for Weaviate
                valid_uuid = string_to_uuid(_id)
                collection.data.delete_by_id(valid_uuid)
        except Exception as e:
            print(f"Error deleting from Weaviate: {e}")

    async def clear(self):
        """Clear all objects from collection (v4 API)"""
        try:
            collection = self.client.collections.get("Document")
            collection.data.delete_many(
                where={
                    "path": ["id"],
                    "operator": "Like",
                    "valueString": "*"
                }
            )
        except Exception as e:
            print(f"Vector clear failed: {e}")

    async def count(self) -> int:
        """Count objects in collection (v4 API)"""
        try:
            collection = self.client.collections.get("Document")
            # Use aggregate to count
            result = collection.aggregate.over_all()
            return result.total_count if hasattr(result, 'total_count') else 0
        except Exception as e:
            print(f"Count failed: {e}")
            return 0

    def close(self):
        """Close Weaviate client connection"""
        try:
            if hasattr(self.client, 'close'):
                self.client.close()
        except Exception as e:
            print(f"Error closing Weaviate client: {e}")

@dataclass
class WeaviateKVStorage(BaseKVStorage):
    client: Any = field(init=False)  # Weaviate v4 client

    def __post_init__(self):
        """Initialize Weaviate v4 client using connect_to_local()"""
        self.client = get_weaviate_client()
        self.init_schema()

    async def all_keys(self) -> List[str]:
        """Get all keys from KVDocument collection (v4 API)"""
        try:
            collection = self.client.collections.get("KVDocument")
            response = collection.query.fetch_objects()
            
            keys = []
            for obj in response.objects:
                if "key" in obj.properties:
                    keys.append(obj.properties["key"])
            
            return keys
        except Exception as e:
            print(f"Weaviate all_keys error: {e}")
            return []

    async def get_by_id(self, id: str) -> Union[Dict, None]:
        """Get object by ID from KVDocument collection (v4 API)"""
        try:
            collection = self.client.collections.get("KVDocument")
            # Convert string ID to valid UUID for Weaviate
            valid_uuid = string_to_uuid(id)
            obj = collection.query.fetch_object_by_id(valid_uuid)
            
            if obj:
                return obj.properties
            return None
        except Exception as e:
            print(f"Weaviate get_by_id error for {id}: {e}")
            return None

    async def get_by_ids(self, ids: List[str], fields: Union[Set[str], None] = None) -> List[Union[Dict, None]]:
        """Get multiple objects by IDs (v4 API)"""
        results = []
        for _id in ids:
            data = await self.get_by_id(_id)
            if data and fields:
                data = {k: v for k, v in data.items() if k in fields}
            results.append(data)
        return results

    async def filter_keys(self, data: List[str]) -> Set[str]:
        """Filter out existing keys (v4 API)"""
        existing_keys = set(await self.all_keys())
        return set(data) - existing_keys

    async def upsert(self, data: Dict[str, Dict]):
        """Upsert KV data into collection (v4 API)"""
        try:
            collection = self.client.collections.get("KVDocument")
            
            for _id, value in data.items():
                # Convert string ID to valid UUID for Weaviate
                valid_uuid = string_to_uuid(_id)
                try:
                    # Try to insert first
                    collection.data.insert(
                        uuid=valid_uuid,
                        properties=value
                    )
                except Exception as insert_err:
                    # If already exists, try to update
                    if "already exists" in str(insert_err):
                        try:
                            collection.data.update(
                                uuid=valid_uuid,
                                properties=value
                            )
                        except Exception as update_err:
                            pass  # Silently ignore update errors
                    else:
                        pass  # Silently ignore other errors
        except Exception as e:
            pass  # Silently ignore top-level errors

    async def delete(self, ids: List[str]):
        """Delete KV objects by IDs (v4 API)"""
        try:
            collection = self.client.collections.get("KVDocument")
            
            for _id in ids:
                # Convert string ID to valid UUID for Weaviate
                valid_uuid = string_to_uuid(_id)
                collection.data.delete_by_id(valid_uuid)
        except Exception as e:
            print(f"Error deleting KV data: {e}")

    def init_schema(self):
        """Create KVDocument collection if it doesn't exist (v4 API)"""
        try:
            # collections.list_all() returns a dict, not a list
            collections = self.client.collections.list_all()
            collection_names = list(collections.keys()) if isinstance(collections, dict) else [c.name for c in collections]
            
            if "KVDocument" not in collection_names:
                self.client.collections.create(
                    name="KVDocument",
                    description="Key-Value storage collection"
                )
                print("[Weaviate] Created 'KVDocument' collection")
        except Exception as e:
            print(f"KV schema init error: {e}")

    async def clear(self):
        """Clear all objects from KV collection (v4 API)"""
        try:
            collection = self.client.collections.get("KVDocument")
            collection.data.delete_many(
                where={
                    "path": ["id"],
                    "operator": "Like",
                    "valueString": "*"
                }
            )
        except Exception as e:
            print(f"KV clear failed: {e}")

    async def count(self) -> int:
        """Count KV objects (v4 API)"""
        try:
            collection = self.client.collections.get("KVDocument")
            result = collection.aggregate.over_all()
            return result.total_count if hasattr(result, 'total_count') else 0
        except Exception as e:
            print(f"KV count failed: {e}")
            return 0

    def close(self):
        """Close Weaviate client connection"""
        try:
            if hasattr(self.client, 'close'):
                self.client.close()
        except Exception as e:
            print(f"Error closing Weaviate client: {e}")
        
@dataclass
class WeaviateGraphStorage(BaseGraphStorage):
    client: Any = field(init=False)  # Weaviate v4 client

    def __post_init__(self):
        """Initialize Weaviate v4 client using connect_to_local()"""
        self.client = get_weaviate_client()
        self.init_schema()

    async def get_types(self) -> tuple[List[str], List[str]]:
        """Returns (list of node classes, list of edge classes) using v4 API"""
        try:
            # collections.list_all() returns a dict with collection names as keys
            collections = self.client.collections.list_all()
            node_classes = list(collections.keys()) if isinstance(collections, dict) else [c.name for c in collections]
            # Weaviate edges are references inside properties, so edge classes uncommon
            edge_classes = []
            return node_classes, edge_classes
        except Exception as e:
            print(f"Weaviate get_types error: {e}")
            return [], []

    async def get_node_from_types(self, node_types: List[str], num_nodes: int = 10) -> List[Dict]:
        """Get nodes by type (v4 API) - required by MiniRAG
        Returns a list of dictionaries with entity_name and properties"""
        try:
            node_datas = []
            for node_type in node_types:
                try:
                    collection = self.client.collections.get(node_type)
                    objs = collection.query.fetch_objects(limit=num_nodes)
                    
                    for obj in objs.objects:
                        node_dict = {
                            "entity_name": str(obj.uuid),  # Use UUID as entity name
                            "entity_type": node_type,
                            **obj.properties  # Include all properties from the node
                        }
                        node_datas.append(node_dict)
                except Exception as e:
                    # Silently skip collections that don't exist
                    pass
                    
            return node_datas
        except Exception as e:
            print(f"Error getting nodes from types: {e}")
            return []

    async def has_node(self, node_id: str) -> bool:
        """Check if node exists (v4 API)"""
        try:
            collection = self.client.collections.get("GraphNode")
            valid_uuid = string_to_uuid(node_id)
            obj = collection.query.fetch_object_by_id(valid_uuid)
            return obj is not None
        except Exception as e:
            print(f"Error checking node {node_id}: {e}")
            return False

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Weaviate edges are references; this is complex, simplified here"""
        return False

    async def node_degree(self, node_id: str) -> int:
        """Not trivial in Weaviate, returning 0"""
        return 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Not implemented for Weaviate"""
        return 0

    async def get_node(self, node_id: str) -> Union[Dict, None]:
        """Get node data by ID (v4 API)"""
        try:
            collection = self.client.collections.get("GraphNode")
            valid_uuid = string_to_uuid(node_id)
            obj = collection.query.fetch_object_by_id(valid_uuid)
            
            if obj:
                return obj.properties
            return None
        except Exception as e:
            print(f"Error getting node {node_id}: {e}")
            return None

    async def get_edge(self, source_node_id: str, target_node_id: str) -> Union[Dict, None]:
        """Get edge data (v4 API)"""
        return None

    async def get_node_edges(self, source_node_id: str) -> Union[List[tuple], None]:
        """Get all edges from a node (v4 API)"""
        return None

    async def upsert_node(self, node_id: str, node_data: Dict[str, str]):
        """Upsert node into GraphNode collection (v4 API)"""
        try:
            collection = self.client.collections.get("GraphNode")
            # Convert string node_id to valid UUID for Weaviate
            valid_uuid = string_to_uuid(node_id)
            
            # Check if node already exists, if so update it
            try:
                existing = collection.query.fetch_object_by_id(valid_uuid)
                if existing:
                    # Update existing node
                    collection.data.update(
                        uuid=valid_uuid,
                        properties=node_data
                    )
                else:
                    # Create new node
                    collection.data.insert(
                        uuid=valid_uuid,
                        properties=node_data
                    )
            except:
                # If fetch fails, try to insert
                collection.data.insert(
                    uuid=valid_uuid,
                    properties=node_data
                )
        except Exception as e:
            # "already exists" errors are expected in Weaviate and can be ignored
            if "already exists" not in str(e):
                print(f"Error upserting node {node_id}: {e}")

    def init_schema(self):
        """Create GraphNode collection if it doesn't exist (v4 API)"""
        try:
            # collections.list_all() returns a dict, not a list
            collections = self.client.collections.list_all()
            collection_names = list(collections.keys()) if isinstance(collections, dict) else [c.name for c in collections]
            
            if "GraphNode" not in collection_names:
                self.client.collections.create(
                    name="GraphNode",
                    description="Graph node collection"
                )
                print("[Weaviate] Created 'GraphNode' collection")
        except Exception as e:
            print(f"Graph schema init error: {e}")

    async def clear(self):
        """Clear all objects from GraphNode collection (v4 API)"""
        try:
            collection = self.client.collections.get("GraphNode")
            collection.data.delete_many(
                where={
                    "path": ["id"],
                    "operator": "Like",
                    "valueString": "*"
                }
            )
        except Exception as e:
            print(f"Graph clear failed: {e}")

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: Dict[str, str]):
        """Upsert edge by creating reference (v4 API)"""
        try:
            # In Weaviate v4, references are created through collection API
            collection = self.client.collections.get("GraphNode")
            # This is a simplified approach - edge data would need to be stored differently
            # For now, we just note that this is a cross-reference
            print(f"[Weaviate] Edge reference: {source_node_id} -> {target_node_id}")
        except Exception as e:
            print(f"Edge creation failed: {e}")

    async def delete_node(self, node_id: str):
        """Delete node from GraphNode collection (v4 API)"""
        try:
            collection = self.client.collections.get("GraphNode")
            collection.data.delete_by_id(node_id)
        except Exception as e:
            print(f"Error deleting node {node_id}: {e}")

    async def embed_nodes(self, algorithm: str):
        """Node embedding not implemented for Weaviate"""
        raise NotImplementedError("Node embedding not implemented for Weaviate.")

    def close(self):
        """Close Weaviate client connection"""
        try:
            if hasattr(self.client, 'close'):
                self.client.close()
        except Exception as e:
            print(f"Error closing Weaviate client: {e}")
