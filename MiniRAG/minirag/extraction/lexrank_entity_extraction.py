"""
Example integration of LexRank-based entity extraction into MiniRAG.

This module demonstrates how to integrate the LexRank summarizer into the
entity extraction pipeline to reduce processing time while maintaining quality.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict

from minirag.base import BaseGraphStorage, BaseVectorStorage, TextChunkSchema
from minirag.utils import logger, compute_mdhash_id
from minirag.summarization.lexrank_summarizer import DocumentSummarizer
from minirag.prompt import PROMPTS, GRAPH_FIELD_SEP

# Import entity extraction helpers
from minirag.operate import (
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction,
    _merge_nodes_then_upsert,
    _merge_edges_then_upsert,
)


async def extract_entities_with_lexrank(
    chunks: Dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    entity_name_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: Dict,
) -> Union[BaseGraphStorage, None]:
    """Extract entities from document summaries using LexRank.
    
    Instead of analyzing every chunk, we:
    1. Use LexRank to identify key sentences
    2. Only extract entities from these key sentences
    3. Reduce processing time and improve focus on important content
    
    Args:
        chunks: Dictionary of text chunks
        knowledge_graph_inst: Knowledge graph storage
        entity_vdb: Entity vector storage
        entity_name_vdb: Entity name vector storage
        relationships_vdb: Relationship vector storage
        global_config: Global configuration (contains 'lexrank_summarizer' key if available)
    
    Returns:
        Updated knowledge graph
    """
    
    # Get summarizer from global_config or create one
    summarizer = global_config.get("lexrank_summarizer")
    if summarizer is None:
        summarizer = DocumentSummarizer(language="pt")
    
    lexrank_ratio = global_config.get("lexrank_ratio", 0.3)
    
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config.get("entity_extract_max_gleaning", 1)
    
    # Prepare LLM prompts
    entity_extract_prompt = PROMPTS.get("entity_extraction", "")
    continue_prompt = PROMPTS.get("entiti_continue_extraction", "")
    if_loop_prompt = PROMPTS.get("entiti_if_loop_extraction", "")
    
    context_base = {
        "tuple_delimiter": PROMPTS.get("DEFAULT_TUPLE_DELIMITER", "<|>"),
        "record_delimiter": PROMPTS.get("DEFAULT_RECORD_DELIMITER", "||"),
        "completion_delimiter": PROMPTS.get("DEFAULT_COMPLETION_DELIMITER", "<|COMPLETE|>"),
        "entity_types": ",".join(PROMPTS.get("DEFAULT_ENTITY_TYPES", [])),
    }
    
    ordered_chunks = list(chunks.items())
    
    logger.info(
        f"Starting entity extraction with LexRank on {len(ordered_chunks)} chunks"
    )
    
    async def _process_single_content_lexrank(
        chunk_key_dp: Tuple[str, TextChunkSchema]
    ) -> Tuple[Dict, Dict]:
        """Process a single chunk with LexRank summarization."""
        chunk_key, chunk_data = chunk_key_dp
        content = chunk_data.get("content", "")
        
        if not content:
            return {}, {}
        
        try:
            # Extract key sentences using LexRank
            logger.debug(
                f"Extracting key sentences from chunk {chunk_key} "
                f"(original length: {len(content)} chars)"
            )
            key_sentences = summarizer.extract_key_sentences(
                content,
                ratio=lexrank_ratio
            )
            
            # Join key sentences for analysis
            summarized_content = " ".join(key_sentences)
            
            logger.debug(
                f"Chunk {chunk_key} summarized: "
                f"{len(content)} → {len(summarized_content)} chars "
                f"({100*len(summarized_content)/len(content):.1f}% retained)"
            )
            
            # Use summarized content for entity extraction
            prompt = entity_extract_prompt.format(
                input_text=summarized_content,
                **context_base
            )
            
            # Get LLM response for entity extraction
            response = await use_llm_func(prompt)
            
            # Parse entities and relationships from response
            maybe_nodes = defaultdict(list)
            maybe_edges = defaultdict(list)
            
            # Split response into records
            records = response.split(context_base["record_delimiter"])
            
            for record in records:
                record = record.strip()
                if not record:
                    continue
                
                # Parse attributes
                attributes = record.split(context_base["tuple_delimiter"])
                
                # Handle entities
                if attributes and attributes[0].strip().lower() == "entity":
                    entity_data = await _handle_single_entity_extraction(
                        attributes, chunk_key
                    )
                    if entity_data:
                        entity_name = entity_data.get("entity_name", "")
                        maybe_nodes[entity_name].append(entity_data)
                
                # Handle relationships
                elif attributes and attributes[0].strip().lower() == "relationship":
                    edge_data = await _handle_single_relationship_extraction(
                        attributes, chunk_key
                    )
                    if edge_data:
                        src = edge_data.get("src_id", "")
                        tgt = edge_data.get("tgt_id", "")
                        maybe_edges[(src, tgt)].append(edge_data)
            
            logger.debug(
                f"Extracted {len(maybe_nodes)} entities and {len(maybe_edges)} "
                f"relationships from chunk {chunk_key}"
            )
            
            return maybe_nodes, maybe_edges
            
        except Exception as e:
            logger.error(
                f"Error processing chunk {chunk_key} with LexRank: {e}"
            )
            return {}, {}
    
    # Process all chunks concurrently with LexRank summarization
    logger.info("Processing all chunks with LexRank...")
    results = await asyncio.gather(
        *[_process_single_content_lexrank(c) for c in ordered_chunks],
        return_exceptions=False
    )
    
    # Aggregate results
    all_nodes = defaultdict(list)
    all_edges = defaultdict(list)
    
    for result in results:
        if result is None:
            continue
        
        try:
            m_nodes, m_edges = result
        except (TypeError, ValueError):
            logger.warning(f"Invalid result from chunk processing: {result}")
            continue
        
        for node_name, node_data_list in m_nodes.items():
            all_nodes[node_name].extend(node_data_list)
        
        for edge_key, edge_data_list in m_edges.items():
            all_edges[edge_key].extend(edge_data_list)
    
    logger.info(
        f"Total aggregated: {len(all_nodes)} unique entities, "
        f"{len(all_edges)} unique relationships"
    )
    
    # Merge and upsert nodes
    if all_nodes:
        logger.info(f"Upserting {len(all_nodes)} entity nodes...")
        all_entities_data = await asyncio.gather(
            *[
                _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in all_nodes.items()
            ],
            return_exceptions=False
        )
        logger.info(f"Successfully upserted {len(all_entities_data)} entities")
    else:
        all_entities_data = []
        logger.info("No entities to upsert")
    
    # Merge and upsert edges
    if all_edges:
        logger.info(f"Upserting {len(all_edges)} relationships...")
        all_relationships_data = await asyncio.gather(
            *[
                _merge_edges_then_upsert(
                    k[0], k[1], v, knowledge_graph_inst, global_config
                )
                for k, v in all_edges.items()
            ],
            return_exceptions=False
        )
        logger.info(
            f"Successfully upserted {len(all_relationships_data)} relationships"
        )
    else:
        all_relationships_data = []
        logger.info("No relationships to upsert")
    
    # Upsert to vector databases
    if entity_vdb and all_entities_data:
        logger.info("Upserting entities to vector database...")
        try:
            entities_for_vdb = {
                compute_mdhash_id(e["entity_name"], prefix="ent-"): {
                    "content": e["entity_name"],
                    "embedding": await entity_vdb.embedding_func(
                        e["entity_name"]
                    ) if entity_vdb.embedding_func else None,
                    "metadata": {
                        "entity_name": e["entity_name"],
                        "entity_type": e.get("entity_type", "UNKNOWN"),
                    },
                }
                for e in all_entities_data
            }
            await entity_vdb.upsert(entities_for_vdb)
        except Exception as e:
            logger.warning(f"Error upserting entities to VDB: {e}")
    
    if entity_name_vdb and all_entities_data:
        logger.info("Upserting entity names to vector database...")
        try:
            entity_names_for_vdb = {
                compute_mdhash_id(e["entity_name"], prefix="ent-name-"): {
                    "content": e["entity_name"],
                    "embedding": await entity_name_vdb.embedding_func(
                        e["entity_name"]
                    ) if entity_name_vdb.embedding_func else None,
                    "metadata": {"entity_name": e["entity_name"]},
                }
                for e in all_entities_data
            }
            await entity_name_vdb.upsert(entity_names_for_vdb)
        except Exception as e:
            logger.warning(f"Error upserting entity names to VDB: {e}")
    
    if relationships_vdb and all_relationships_data:
        logger.info("Upserting relationships to vector database...")
        try:
            relationships_for_vdb = {
                compute_mdhash_id(
                    f"{r['src_id']}-{r['tgt_id']}", prefix="rel-"
                ): {
                    "content": r.get("description", ""),
                    "embedding": await relationships_vdb.embedding_func(
                        r.get("description", "")
                    ) if relationships_vdb.embedding_func else None,
                    "metadata": {
                        "src_id": r["src_id"],
                        "tgt_id": r["tgt_id"],
                    },
                }
                for r in all_relationships_data
            }
            await relationships_vdb.upsert(relationships_for_vdb)
        except Exception as e:
            logger.warning(f"Error upserting relationships to VDB: {e}")
    
    logger.info(
        f"Entity extraction with LexRank completed: "
        f"{len(all_entities_data)} entities, "
        f"{len(all_relationships_data)} relationships"
    )
    
    return knowledge_graph_inst
