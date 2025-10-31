"""
BART-based entity extraction with summarized descriptions.

This module extends the entity extraction pipeline by:
1. Summarizing chunks using BART before entity extraction
2. Summarizing entity descriptions using BART
3. Providing configurable BART usage

Key differences from standard extract_entities:
- Chunks are summarized before LLM extraction (better quality, lower token usage)
- Entity descriptions are summarized for consistency
- Optional BART usage (can be disabled via BART_ENABLED flag)
"""

import asyncio
import logging
from typing import Union, Optional
from collections import Counter, defaultdict

from .operate import (
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction,
    _merge_edges_then_upsert,
)
from .summarization.bart_summarizer import BARTSummarizer, TRANSFORMERS_AVAILABLE
from .utils import (
    split_string_by_multi_markers,
    logger,
    pack_user_ass_to_openai_messages,
    compute_mdhash_id,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
)
from .metadata_plugin import minirag_generate_metadata
from .prompt import GRAPH_FIELD_SEP, PROMPTS
import re
from collections import defaultdict

# Global flag for BART usage
BART_ENABLED = TRANSFORMERS_AVAILABLE
CHUNK_SUMMARY_RATIO = 0.3  # Summarize chunks to 30% of original length
DESCRIPTION_MAX_LENGTH = 200  # Max tokens for entity descriptions


async def summarize_chunk_for_extraction(
    content: str,
    summarizer: Optional[BARTSummarizer] = None,
) -> str:
    """
    Summarize a chunk before entity extraction to improve quality and reduce tokens.
    
    Args:
        content: Text chunk to summarize
        summarizer: BARTSummarizer instance (created if None)
    
    Returns:
        Summarized text or original if BART disabled/failed
    """
    if not BART_ENABLED or summarizer is None:
        return content
    
    try:
        # Extract key sentences (30% of original)
        key_sentences = summarizer.extract_key_sentences(
            content, 
            ratio=CHUNK_SUMMARY_RATIO
        )
        
        if not key_sentences:
            return content
        
        summarized = " ".join(key_sentences)
        logger.debug(f"Summarized chunk: {len(content)} → {len(summarized)} chars")
        return summarized
    except Exception as e:
        logger.warning(f"Chunk summarization failed: {e}, using original")
        return content


async def summarize_entity_description(
    description: str,
    summarizer: Optional[BARTSummarizer] = None,
) -> str:
    """
    Summarize entity descriptions using BART for consistency.
    
    Args:
        description: Multi-line entity description
        summarizer: BARTSummarizer instance
    
    Returns:
        Summarized description or original if BART disabled/failed
    """
    if not BART_ENABLED or summarizer is None or len(description) < 100:
        return description
    
    try:
        # For descriptions, use shorter ratio
        key_sentences = summarizer.extract_key_sentences(
            description,
            ratio=0.5,  # Keep 50% for descriptions
        )
        
        if not key_sentences:
            return description
        
        summarized = " ".join(key_sentences)
        logger.debug(f"Summarized description: {len(description)} → {len(summarized)} chars")
        return summarized
    except Exception as e:
        logger.warning(f"Description summarization failed: {e}, using original")
        return description


async def _merge_nodes_then_upsert_with_bart(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
    summarizer: Optional[BARTSummarizer] = None,
):
    """
    Merge nodes and upsert with BART-summarized descriptions.
    
    Similar to _merge_nodes_then_upsert but applies BART summarization
    to entity descriptions.
    """
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]

    # Combine descriptions
    combined_description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    
    # Summarize combined description if BART enabled
    description = await summarize_entity_description(
        combined_description,
        summarizer=summarizer,
    )

    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )

    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def extract_entities_with_bart(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    entity_name_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
    use_bart: bool = True,
) -> Union[BaseGraphStorage, None]:
    """
    Extract entities from chunks with BART-based chunk and description summarization.
    
    This is the main BART-enhanced entity extraction function. It:
    1. Optionally summarizes chunks before extraction (reduces tokens, improves quality)
    2. Extracts entities and relationships using LLM
    3. Summarizes entity descriptions using BART
    4. Stores entities and relationships in vector DBs
    
    Args:
        chunks: Dictionary of text chunks
        knowledge_graph_inst: Knowledge graph storage instance
        entity_vdb: Entity vector database
        entity_name_vdb: Entity name vector database
        relationships_vdb: Relationship vector database
        global_config: Global configuration
        use_bart: Whether to use BART for summarization (default: True if available)
    
    Returns:
        Knowledge graph instance or None if extraction failed
    """
    use_bart = use_bart and BART_ENABLED
    
    # Initialize BART summarizer if enabled
    summarizer = None
    if use_bart:
        try:
            summarizer = BARTSummarizer(language="pt")
            logger.info("BART summarizer initialized for chunk and description summarization")
        except Exception as e:
            logger.warning(f"Failed to initialize BART: {e}, proceeding without BART")
            summarizer = None
    
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())
    entity_extract_prompt = PROMPTS["entity_extraction"]

    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content_with_bart(chunk_key_dp: tuple[str, TextChunkSchema]):
        """Process single chunk with BART summarization."""
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        
        # Summarize chunk if BART enabled
        if use_bart and summarizer is not None:
            content = await summarize_chunk_for_extraction(content, summarizer)
        
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        final_result = await use_llm_func(hint_prompt)

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)
            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    results = await asyncio.gather(
        *[_process_single_content_with_bart(c) for c in ordered_chunks]
    )
    
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)
    
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert_with_bart(
                k, v, knowledge_graph_inst, global_config, summarizer
            )
            for k, v in maybe_nodes.items()
        ]
    )
    
    all_relationships_data = await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if not len(all_relationships_data):
        logger.warning(
            "Didn't extract any relationships, maybe your LLM is not working"
        )
        return None

    # Insert into database
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    if entity_name_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_name_vdb.upsert(data_for_vdb)

    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(
                dp["src_id"] + dp["tgt_id"] + dp["description"], prefix="rel-"
            ): {
                "content": dp["src_id"]
                + " "
                + dp["description"]
                + " "
                + dp["tgt_id"],
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
            }
            for dp in all_relationships_data
        }
        await relationships_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst
