import asyncio
import json
import re
from typing import Union
from collections import Counter, defaultdict
import warnings
import json_repair

from .utils import (
    list_of_list_to_csv,
    truncate_list_by_token_size,
    split_string_by_multi_markers,
    logger,
    locate_json_string_body_from_string,
    process_combine_contexts,
    clean_str,
    edge_vote_path,
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken,
    is_float_regex,
    pack_user_ass_to_openai_messages,
    compute_mdhash_id,
    calculate_similarity,
    cal_path_score_list,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .metadata_plugin import (
    minirag_generate_metadata
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
import math
from dataclasses import dataclass

#############################
# Simple BM25 Implementation #
#############################

_RE_TOKEN = re.compile(r"\w+")

def _tokenize(text: str) -> list[str]:
    return _RE_TOKEN.findall(text.lower())

@dataclass
class _BM25Index:
    corpus_tokens: list[list[str]]
    doc_ids: list[str]
    df: dict[str, int]
    avgdl: float
    k1: float = 1.5
    b: float = 0.75

    def score(self, query: str, top_k: int) -> list[tuple[str, float]]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        N = len(self.corpus_tokens)
        # precompute idf
        idf_cache = {}
        scores = [0.0] * N
        for qt in q_tokens:
            if qt not in self.df:
                continue
            if qt not in idf_cache:
                # standard BM25 idf
                df = self.df[qt]
                idf_cache[qt] = math.log((N - df + 0.5) / (df + 0.5) + 1)
            idf = idf_cache[qt]
            for idx, doc_tokens in enumerate(self.corpus_tokens):
                # compute tf for term in doc
                tf = 0
                # naive loop (acceptable for moderate corpus); could optimize with precomputed term freq maps
                for t in doc_tokens:
                    if t == qt:
                        tf += 1
                if tf == 0:
                    continue
                dl = len(doc_tokens)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[idx] += idf * (tf * (self.k1 + 1)) / denom
        ranked = sorted(((self.doc_ids[i], s) for i, s in enumerate(scores) if s > 0), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

async def build_bm25_index(text_chunks_db: BaseKVStorage[TextChunkSchema]) -> _BM25Index | None:
    try:
        all_ids = await text_chunks_db.all_keys()
    except Exception:
        return None
    corpus_tokens = []
    doc_ids = []
    df = Counter()
    # batch fetch to limit memory overhead of awaiting each individually
    batch_size = 256
    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i:i+batch_size]
        batch = await text_chunks_db.get_by_ids(batch_ids)
        for cid, chunk in zip(batch_ids, batch):
            if not chunk or not chunk.get("content"):
                continue
            tokens = _tokenize(chunk["content"])[:4096]  # truncate very long docs
            if not tokens:
                continue
            corpus_tokens.append(tokens)
            doc_ids.append(cid)
            for tok in set(tokens):
                df[tok] += 1
    if not corpus_tokens:
        return None
    avgdl = sum(len(t) for t in corpus_tokens) / len(corpus_tokens)
    return _BM25Index(corpus_tokens=corpus_tokens, doc_ids=doc_ids, df=df, avgdl=avgdl)

async def bm25_query(
    query: str,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    bm25_index_state: dict,
) -> str:
    """Perform a BM25 retrieval over all text chunks and optionally generate an answer via LLM.

    bm25_index_state: a dict with keys {'index': _BM25Index|None, 'dirty': bool}
    This allows MiniRAG to cache & invalidate the index after insertions without a new class.
    """
    # lazy build or rebuild index if marked dirty
    if bm25_index_state.get("index") is None or bm25_index_state.get("dirty"):
        bm25_index_state["index"] = await build_bm25_index(text_chunks_db)
        bm25_index_state["dirty"] = False
    index: _BM25Index | None = bm25_index_state.get("index")
    if index is None:
        return PROMPTS["fail_response"]
    ranked = index.score(query, top_k=query_param.top_k)
    if not ranked:
        return PROMPTS["fail_response"]
    # fetch chunk contents
    chunk_ids = [cid for cid,_ in ranked]
    chunks = await text_chunks_db.get_by_ids(chunk_ids)
    # keep ordering by score
    id2score = {cid:score for cid,score in ranked}
    enriched = []
    for cid, ch in zip(chunk_ids, chunks):
        if ch:
            enriched.append({"id": cid, "score": id2score[cid], **ch})
    # truncate by token budget
    maybe_trun_chunks = truncate_list_by_token_size(
        enriched,
        key=lambda x: x.get("content", ""),
        max_token_size=query_param.max_token_for_text_unit,
    )
    section = "--BM25 Chunk--\n".join([c.get("content", "") for c in maybe_trun_chunks])
    if query_param.only_need_context:
        return section
    use_model_func = global_config["llm_model_func"]
    sys_prompt_temp = PROMPTS.get("naive_rag_response", PROMPTS["rag_response"])
    sys_prompt = sys_prompt_temp.format(content_data=section, response_type=query_param.response_type) if "content_data" in sys_prompt_temp else sys_prompt_temp.format(context_data=section, response_type=query_param.response_type)
    response = await use_model_func(query, system_prompt=sys_prompt)
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )
    return response


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])

    edge_keywords = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """
    Merge nodes and upsert with optional BART summarization for descriptions.
    
    Args:
        entity_name: Name of the entity
        nodes_data: List of node data dictionaries
        knowledge_graph_inst: Knowledge graph storage instance
        global_config: Global configuration dictionary containing optional 'summarizer' key
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
    
    # Use BART summarization if summarizer is available in global_config
    summarizer = global_config.get("summarizer")
    if summarizer is not None:
        from .operate_bart_entity import summarize_entity_description
        description = await summarize_entity_description(
            combined_description,
            summarizer=summarizer,
        )
    else:
        # Keep original behavior - concatenate descriptions
        description = combined_description
    
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )

    # description = await _handle_entity_relation_summary(
    #     entity_name, description, global_config
    # )
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


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_keywords.extend(
            split_string_by_multi_markers(already_edge["keywords"], [GRAPH_FIELD_SEP])
        )

    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    keywords = GRAPH_FIELD_SEP.join(
        sorted(set([dp["keywords"] for dp in edges_data] + already_keywords))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    # description = await _handle_entity_relation_summary(
    #     (src_id, tgt_id), description, global_config
    # )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
    )

    return edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    entity_name_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    """
    Extract entities from chunks with optional BART summarization or LexRank.
    
    Args:
        chunks: Dictionary of text chunks
        knowledge_graph_inst: Knowledge graph storage instance
        entity_vdb: Entity vector database
        entity_name_vdb: Entity name vector database
        relationships_vdb: Relationship vector database
        global_config: Global configuration dictionary 
                      (contains 'summarizer' key if BART is enabled,
                       'key_sentences_lexrank' if LexRank is enabled,
                       'lexrank_summarizer' if LexRank summarizer available)
    
    Returns:
        Knowledge graph instance or None if extraction failed
    """
    # Standard entity extraction with optional BART summarization
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())
    # if global_config['RAGmode'] == 'minirag':
    #     # entity_extract_prompt = PROMPTS["entity_extraction_noDes"]
    #     entity_extract_prompt = PROMPTS["entity_extraction"]
    # else:
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

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
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
        *[_process_single_content(c) for c in ordered_chunks]
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
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
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

    # Insert in to database
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + " " + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    if entity_name_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="Ename-"): {
                "content": dp["entity_name"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_name_vdb.upsert(data_for_vdb)

    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "content": dp["keywords"]
                + " " + dp["src_id"]
                + " " + dp["tgt_id"]
                + " " + dp["description"],
            }
            for dp in all_relationships_data
        }

        await relationships_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst

async def presidio_entity_extraction(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    entity_name_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    """
    Extract entities from text using Presidio-based metadata extraction.
    Deprecated
    
    Args:
        chunks: Dictionary of text chunks
        knowledge_graph_inst: Knowledge graph storage instance
        entity_vdb: Entity vector database
        entity_name_vdb: Entity name vector database
        relationships_vdb: Relationship vector database
        global_config: Global configuration dictionary (contains 'summarizer' key if BART is enabled)
    """
    logger.info("Using Presidio for entity extraction")

    ordered_chunks = list(chunks.items())
    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_chunk(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
    
        # Use minirag_generate_metadata for regex-based extraction
        metadata = minirag_generate_metadata(
            doc_id=chunk_key,
            text=content,
            path=chunk_key,  # Use chunk_key as path
            extractor=global_config.get("metadata_extractor")
        )
        
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)

        keys = metadata.keys()
        
        # Extract entities from metadata
        entities = metadata.get("entities", [])
        person_names = metadata.get("person_names", [])
        pii_detections = metadata.get("pii_detections", [])
        
        # Process PII detections
        for detection in pii_detections:
            entity_text = detection.get("text", "").strip()
            if not entity_text:
                continue
                
            entity_name = clean_str(entity_text.upper())
            if not entity_name.strip():
                continue

            entity_type = detection.get("entity_type", "PII")

            entity_data = dict(
                entity_name=entity_name,
                entity_type=entity_type,
                description=f"",
                source_id=chunk_key,
            )
            maybe_nodes[entity_name].append(entity_data)
        
        # Create relationships between entities in the same chunk
        entity_names_in_chunk = list(maybe_nodes.keys())
        for i, src_name in enumerate(entity_names_in_chunk):
            for tgt_name in entity_names_in_chunk[i+1:]:
                # Create a relationship between co-occurring entities
                relationship_data = dict(
                    src_id=src_name,
                    tgt_id=tgt_name,
                    weight=1.0,
                    description=f"",
                    keywords="co-occurrence",
                    source_id=chunk_key,
                )
                edge_key = (src_name, tgt_name)
                maybe_edges[edge_key].append(relationship_data)
        
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
        *[_process_single_chunk(c) for c in ordered_chunks]
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
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
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
        logger.warning("Didn't extract any entities, maybe the Presidio extraction failed")
        return None
    if not len(all_relationships_data):
        logger.warning(
            "Didn't extract any relationships, this is normal for Presidio extraction"
        )

    # Insert into vector databases
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
            compute_mdhash_id(dp["entity_name"], prefix="Ename-"): {
                "content": dp["entity_name"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_name_vdb.upsert(data_for_vdb)

    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "content": dp["keywords"]
                + " " + dp["src_id"]
                + " " + dp["tgt_id"]
                + " " + dp["description"],
            }
            for dp in all_relationships_data
        }
        await relationships_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst

async def local_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    context = None
    use_model_func = global_config["llm_model_func"]

    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query)
    result = await use_model_func(kw_prompt)
    json_text = locate_json_string_body_from_string(result)

    try:
        keywords_data = json.loads(json_text)
        keywords = keywords_data.get("low_level_keywords", [])
        keywords = ", ".join(keywords)
    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[1].split("}")[0] + "}"

            keywords_data = json.loads(result)
            keywords = keywords_data.get("low_level_keywords", [])
            keywords = ", ".join(keywords)
        # Handle parsing error
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]
    if keywords:
        context = await _build_local_query_context(
            keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    return response


async def _build_local_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k)

    if not len(results):
        return None
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]  # what is this text_chunks_db doing.  dont remember it in airvx.  check the diagram.
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    logger.info(
        f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  # Add source_id check
    }

    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            if this_edges:  # Add check for None edges
                for e in this_edges:
                    if (
                        e[1] in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[e[1]]
                    ):
                        relation_counts += 1

            chunk_data = await text_chunks_db.get_by_id(c_id)
            if chunk_data is not None and "content" in chunk_data:  # Add content check
                all_text_units_lookup[c_id] = {
                    "data": chunk_data,
                    "order": index,
                    "relation_counts": relation_counts,
                }

    # Filter out None values and ensure data has content
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )

    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = set()
    for this_edges in all_related_edges:
        all_edges.update([tuple(sorted(e)) for e in this_edges])
    all_edges = list(all_edges)
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    return all_edges_data


async def global_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    context = None
    use_model_func = global_config["llm_model_func"]

    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query)
    result = await use_model_func(kw_prompt)
    json_text = locate_json_string_body_from_string(result)

    try:
        keywords_data = json.loads(json_text)
        keywords = keywords_data.get("high_level_keywords", [])
        keywords = ", ".join(keywords)
    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[1].split("}")[0] + "}"

            keywords_data = json.loads(result)
            keywords = keywords_data.get("high_level_keywords", [])
            keywords = ", ".join(keywords)

        except json.JSONDecodeError as e:
            # Handle parsing error
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]
    if keywords:
        context = await _build_global_query_context(
            keywords,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    return response


async def _build_global_query_context(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await relationships_vdb.query(keywords, top_k=query_param.top_k)

    if not len(results):
        return None

    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")
    edge_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"]) for r in results]
    )
    edge_datas = [
        {"src_id": k["src_id"], "tgt_id": k["tgt_id"], "rank": d, **v}
        for k, v, d in zip(results, edge_datas, edge_degree)
        if v is not None
    ]
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )

    use_entities = await _find_most_related_entities_from_relationships(
        edge_datas, query_param, knowledge_graph_inst
    )
    use_text_units = await _find_related_text_unit_from_relationships(
        edge_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    logger.info(
        f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} text units"
    )
    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(edge_datas):
        relations_section_list.append(
            [
                i,
                e["src_id"],
                e["tgt_id"],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(use_entities):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = set()
    for e in edge_datas:
        entity_names.add(e["src_id"])
        entity_names.add(e["tgt_id"])

    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(entity_name) for entity_name in entity_names]
    )

    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(entity_name) for entity_name in entity_names]
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]

    all_text_units_lookup = {}

    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                }

    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]

    return all_text_units


async def hybrid_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    low_level_context = None
    high_level_context = None
    use_model_func = global_config["llm_model_func"]

    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query)

    result = await use_model_func(kw_prompt)
    json_text = locate_json_string_body_from_string(result)
    try:
        keywords_data = json.loads(json_text)
        hl_keywords = keywords_data.get("high_level_keywords", [])
        ll_keywords = keywords_data.get("low_level_keywords", [])
        hl_keywords = ", ".join(hl_keywords)
        ll_keywords = ", ".join(ll_keywords)
    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[1].split("}")[0] + "}"
            keywords_data = json.loads(result)
            hl_keywords = keywords_data.get("high_level_keywords", [])
            ll_keywords = keywords_data.get("low_level_keywords", [])
            hl_keywords = ", ".join(hl_keywords)
            ll_keywords = ", ".join(ll_keywords)
        # Handle parsing error
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]
    if ll_keywords:
        low_level_context = await _build_local_query_context(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )

    if hl_keywords:
        high_level_context = await _build_global_query_context(
            hl_keywords,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )

    context = combine_contexts(high_level_context, low_level_context)

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )
    return response


def combine_contexts(high_level_context, low_level_context):
    # Function to extract entities, relationships, and sources from context strings

    def extract_sections(context):
        entities_match = re.search(
            r"-----Entities-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )
        relationships_match = re.search(
            r"-----Relationships-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )
        sources_match = re.search(
            r"-----Sources-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )

        entities = entities_match.group(1) if entities_match else ""
        relationships = relationships_match.group(1) if relationships_match else ""
        sources = sources_match.group(1) if sources_match else ""

        return entities, relationships, sources

    # Extract sections from both contexts

    if high_level_context is None:
        warnings.warn(
            "High Level context is None. Return empty High entity/relationship/source"
        )
        hl_entities, hl_relationships, hl_sources = "", "", ""
    else:
        hl_entities, hl_relationships, hl_sources = extract_sections(high_level_context)

    if low_level_context is None:
        warnings.warn(
            "Low Level context is None. Return empty Low entity/relationship/source"
        )
        ll_entities, ll_relationships, ll_sources = "", "", ""
    else:
        ll_entities, ll_relationships, ll_sources = extract_sections(low_level_context)

    # Combine and deduplicate the entities

    combined_entities = process_combine_contexts(hl_entities, ll_entities)
    combined_entities = chunking_by_token_size(combined_entities, max_token_size=2000)
    # Combine and deduplicate the relationships
    combined_relationships = process_combine_contexts(
        hl_relationships, ll_relationships
    )
    combined_relationships = chunking_by_token_size(
        combined_relationships, max_token_size=2000
    )
    # Combine and deduplicate the sources
    combined_sources = process_combine_contexts(hl_sources, ll_sources)
    combined_sources = chunking_by_token_size(combined_sources, max_token_size=2000)
    # Format the combined context
    return f"""
-----Entities-----
```csv
{combined_entities}
```
-----Relationships-----
```csv
{combined_relationships}
```
-----Sources-----
```csv
{combined_sources}
```
"""


async def naive_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    use_model_func = global_config["llm_model_func"]
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return PROMPTS["fail_response"]
    chunks_ids = [r["id"] for r in results]

    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])
    if query_param.only_need_context:
        return section
    sys_prompt_temp = PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )

    if len(response) > len(sys_prompt):
        response = (
            response[len(sys_prompt) :]
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    return response

async def doc_query(
    doc_id: str,
    doc_status_storage: BaseKVStorage,  # expected to store DocProcessingStatus-like dicts
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    query_param: QueryParam,
    global_config: dict,
):
    """Retrieve detailed information for a single document.

    Returns a dictionary with:
    - metadata & status fields
    - number of chunks
    - chunk list (optionally truncated by token limits)
    - related entities (graph nodes whose source_id contains any chunk id)
    - related relationships (graph edges whose source_id contains any chunk id)
    """
    # 1. Fetch document status / metadata
    doc_data = await doc_status_storage.get_by_id(doc_id)
    if doc_data is None:
        return {"error": f"Document {doc_id} not found"}

    # 2. Collect chunk ids by scanning text_chunks_db (no direct index available)
    try:
        all_chunk_ids = await text_chunks_db.all_keys()
    except Exception:
        all_chunk_ids = []

    related_chunk_ids = []
    related_chunks = []
    # Fetch in batches to reduce memory usage
    batch_size = 100
    for i in range(0, len(all_chunk_ids), batch_size):
        batch_ids = all_chunk_ids[i : i + batch_size]
        batch = await text_chunks_db.get_by_ids(batch_ids)
        for cid, chunk in zip(batch_ids, batch):
            if not chunk:
                continue
            if chunk.get("full_doc_id") == doc_id:
                related_chunk_ids.append(cid)
                related_chunks.append({
                    "id": cid,
                    "content": chunk.get("content", ""),
                    "tokens": chunk.get("tokens"),
                    "chunk_order_index": chunk.get("chunk_order_index"),
                })

    # Optional truncation by token budget (reuse truncate_list_by_token_size helper)
    try:
        from .utils import truncate_list_by_token_size

        related_chunks = truncate_list_by_token_size(
            related_chunks,
            key=lambda x: x["content"],
            max_token_size=query_param.max_token_for_text_unit,
        )
    except Exception:
        pass

    # 3. Gather entities & relationships from graph (best-effort; depends on storage implementation)
    entities = []
    relationships = []
    chunk_id_set = set(related_chunk_ids)

    # Helper to test if a source_id field (concatenated by GRAPH_FIELD_SEP) intersects chunk ids
    def _source_matches(source_id: str) -> bool:
        if not source_id:
            return False
        parts = [p for p in source_id.split(GRAPH_FIELD_SEP) if p]
        return any(p in chunk_id_set for p in parts)

    # Attempt introspection for nodes/edges if underlying storage exposes a NetworkX-like _graph
    if hasattr(knowledge_graph_inst, "_graph"):
        try:
            nx_graph = getattr(knowledge_graph_inst, "_graph")
            for node_id, data in nx_graph.nodes(data=True):
                source_field = data.get("source_id", "")
                if _source_matches(source_field):
                    entities.append(
                        {
                            "entity_name": node_id,
                            "entity_type": data.get("entity_type"),
                            "description": data.get("description"),
                            "source_id": source_field,
                        }
                    )
            for src, tgt, edata in nx_graph.edges(data=True):
                source_field = edata.get("source_id", "")
                if _source_matches(source_field):
                    relationships.append(
                        {
                            "src_id": src,
                            "tgt_id": tgt,
                            "description": edata.get("description"),
                            "keywords": edata.get("keywords"),
                            "weight": edata.get("weight"),
                            "source_id": source_field,
                        }
                    )
        except Exception as e:  # pragma: no cover - best effort
            logger.error(f"Failed to enumerate graph for doc query: {e}")
    else:
        # Fallback: cannot enumerate entities/relationships for this backend
        logger.warning(
            "Graph storage does not expose '_graph'; entity/relationship listing skipped"
        )

    result = {
        "doc_id": doc_id,
        "status": doc_data.get("status"),
        "file_path": doc_data.get("file_path"),
        "created_at": doc_data.get("created_at"),
        "updated_at": doc_data.get("updated_at"),
        "content_summary": doc_data.get("content_summary"),
        "content_length": doc_data.get("content_length"),
        "chunks_count": doc_data.get("chunks_count") or len(related_chunks),
        "metadata": doc_data.get("metadata", {}),
        "chunks": related_chunks,
        "entities": entities,
        "relationships": relationships,
    }
    return result


async def path2chunk(
    scored_edged_reasoning_path, knowledge_graph_inst, pairs_append, query, max_chunks=5
):
    already_node = {}
    for k, v in scored_edged_reasoning_path.items():
        node_chunk_id = None

        for pathtuple, scorelist in v["Path"].items():
            # collect edge-derived chunk ids
            if pathtuple in pairs_append:
                use_edge = pairs_append[pathtuple]
                edge_datas = await asyncio.gather(
                    *[knowledge_graph_inst.get_edge(r[0], r[1]) for r in use_edge]
                )
                edge_datas = [ed for ed in edge_datas if ed and ed.get("source_id")]
                text_units = []
                for ed in edge_datas:
                    try:
                        text_units.extend(
                            split_string_by_multi_markers(ed["source_id"], [GRAPH_FIELD_SEP])
                        )
                    except Exception:
                        continue
            else:
                text_units = []

            # first node in path tuple
            node_datas = await asyncio.gather(*[knowledge_graph_inst.get_node(pathtuple[0])])
            for dp in node_datas:
                if not dp or not dp.get("source_id"):
                    continue
                try:
                    text_units_node = split_string_by_multi_markers(
                        dp["source_id"], [GRAPH_FIELD_SEP]
                    )
                    text_units.extend(text_units_node)
                except Exception:
                    pass

            # remaining nodes
            node_datas = await asyncio.gather(*[knowledge_graph_inst.get_node(ents) for ents in pathtuple[1:]])
            if query is not None:
                for dp in node_datas:
                    if not dp:
                        continue
                    try:
                        text_units_node = split_string_by_multi_markers(
                            dp.get("source_id", ""), [GRAPH_FIELD_SEP]
                        ) if dp.get("source_id") else []
                        descriptionlist_node = split_string_by_multi_markers(
                            dp.get("description", ""), [GRAPH_FIELD_SEP]
                        ) if dp.get("description") else []
                    except Exception:
                        continue
                    if not descriptionlist_node:
                        continue
                    desc_key = descriptionlist_node[0]
                    if desc_key not in already_node:
                        already_node[desc_key] = None
                        if text_units_node and len(text_units_node) == len(descriptionlist_node):
                            if len(text_units_node) > 5:
                                max_ids = int(max(5, len(text_units_node) / 2))
                                try:
                                    should_consider_idx = calculate_similarity(
                                        descriptionlist_node, query, k=max_ids
                                    )
                                    text_units_node = [text_units_node[i] for i in should_consider_idx]
                                    already_node[desc_key] = text_units_node
                                except Exception:
                                    pass
                    else:
                        text_units_node = already_node.get(desc_key)
                    if text_units_node:
                        text_units.extend(text_units_node)

            count_dict = Counter(text_units)
            total_score = scorelist[0] + scorelist[1] + 1
            for key, value in count_dict.items():
                count_dict[key] = value * total_score
            if node_chunk_id is None:
                node_chunk_id = count_dict
            else:
                node_chunk_id = node_chunk_id + count_dict
        v["Path"] = []
        if node_chunk_id is None:
            # Fallback: no aggregated path info; use the node's own source_id chunks
            node_datas = await asyncio.gather(knowledge_graph_inst.get_node(k))
            fallback_counter = Counter()
            for dp in node_datas:
                if not dp or not dp.get("source_id"):
                    continue
                try:
                    text_units_node = split_string_by_multi_markers(
                        dp["source_id"], [GRAPH_FIELD_SEP]
                    )
                    fallback_counter.update(text_units_node)
                except Exception:
                    continue
            for chunk_id, _ in fallback_counter.most_common(max_chunks):
                v["Path"].append(chunk_id)
        else:
            # Use the aggregated counter built from paths & edges
            for chunk_id, _ in node_chunk_id.most_common(max_chunks):
                v["Path"].append(chunk_id)
            # v['Path'] = node_chunk_id.most_common(max_chunks)
    return scored_edged_reasoning_path


def scorednode2chunk(input_dict, values_dict):
    for key, value_list in input_dict.items():
        input_dict[key] = [
            values_dict.get(val, None) for val in value_list if val in values_dict
        ]
        input_dict[key] = [val for val in input_dict[key] if val is not None]


def kwd2chunk(ent_from_query_dict, chunks_ids, chunk_nums):
    final_chunk = Counter()
    final_chunk_id = []
    for key, list_of_dicts in ent_from_query_dict.items():
        total_id_scores = Counter()
        id_scores_list = []
        id_scores = {}
        for d in list_of_dicts:
            if d == list_of_dicts[0]:
                score = d["Score"] * 2
            else:
                score = d["Score"]
            path = d["Path"]

            for id in path:
                if id == path[0] and id in chunks_ids:
                    score = score * 10
                if id in id_scores:
                    id_scores[id] += score
                else:
                    id_scores[id] = score
        id_scores_list.append(id_scores)

        for scores in id_scores_list:
            total_id_scores.update(scores)
        final_chunk = final_chunk + total_id_scores  # .most_common(3)

    for i in final_chunk.most_common(chunk_nums):
        final_chunk_id.append(i[0])
    return final_chunk_id


async def _build_mini_query_context(
    ent_from_query,
    type_keywords,
    originalquery,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    entity_name_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    embedder,
    query_param: QueryParam,
):
    imp_ents = []
    nodes_from_query_list = []
    ent_from_query_dict = {}

    for ent in ent_from_query:
        ent_from_query_dict[ent] = []
        results_node = await entity_name_vdb.query(ent, top_k=query_param.top_k)

        nodes_from_query_list.append(results_node)
        ent_from_query_dict[ent] = [e["entity_name"] for e in results_node]

    candidate_reasoning_path = {}

    for results_node_list in nodes_from_query_list:
        candidate_reasoning_path_new = {
            key["entity_name"]: {"Score": key["distance"], "Path": []}
            for key in results_node_list
        }

        candidate_reasoning_path = {
            **candidate_reasoning_path,
            **candidate_reasoning_path_new,
        }
    for key in candidate_reasoning_path.keys():
        candidate_reasoning_path[key][
            "Path"
        ] = await knowledge_graph_inst.get_neighbors_within_k_hops(key, 2)
        imp_ents.append(key)

    short_path_entries = {
        name: entry
        for name, entry in candidate_reasoning_path.items()
        if len(entry["Path"]) < 1
    }
    sorted_short_path_entries = sorted(
        short_path_entries.items(), key=lambda x: x[1]["Score"], reverse=True
    )
    save_p = max(1, int(len(sorted_short_path_entries) * 0.2))
    top_short_path_entries = sorted_short_path_entries[:save_p]
    top_short_path_dict = {name: entry for name, entry in top_short_path_entries}
    long_path_entries = {
        name: entry
        for name, entry in candidate_reasoning_path.items()
        if len(entry["Path"]) >= 1
    }
    candidate_reasoning_path = {**long_path_entries, **top_short_path_dict}
    node_datas_from_type = await knowledge_graph_inst.get_node_from_types(
        type_keywords
    )  # entity_type, description,...

    maybe_answer_list = [n["entity_name"] for n in node_datas_from_type]
    imp_ents = imp_ents + maybe_answer_list
    scored_reasoning_path = cal_path_score_list(
        candidate_reasoning_path, maybe_answer_list
    )

    results_edge = await relationships_vdb.query(
        originalquery, top_k=len(ent_from_query) * query_param.top_k
    )
    goodedge = []
    badedge = []
    for item in results_edge:
        # Handle case where src_id and tgt_id might be in different keys or be None
        src_id = item.get("src_id") or item.get("source") or item.get("source_id")
        tgt_id = item.get("tgt_id") or item.get("target") or item.get("target_id")
        
        if src_id is not None and tgt_id is not None:
            if src_id in imp_ents or tgt_id in imp_ents:
                goodedge.append(item)
            else:
                badedge.append(item)
    scored_edged_reasoning_path, pairs_append = edge_vote_path(
        scored_reasoning_path, goodedge
    )
    scored_edged_reasoning_path = await path2chunk(
        scored_edged_reasoning_path,
        knowledge_graph_inst,
        pairs_append,
        originalquery,
        max_chunks=3,
    )

    entites_section_list = []
    node_datas = await asyncio.gather(
        *[
            knowledge_graph_inst.get_node(entity_name)
            for entity_name in scored_edged_reasoning_path.keys()
        ]
    )

    node_datas = [
        {**n, "entity_name": k, "Score": scored_edged_reasoning_path[k]["Score"]}
        for k, n in zip(scored_edged_reasoning_path.keys(), node_datas)
        if n is not None
    ]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                n["entity_name"],
                n["Score"],
                n.get("description", "UNKNOWN"),
            ]
        )
    entites_section_list = sorted(
        entites_section_list, key=lambda x: x[1], reverse=True
    )
    entites_section_list = truncate_list_by_token_size(
        entites_section_list,
        key=lambda x: x[2],
        max_token_size=query_param.max_token_for_node_context,
    )

    entites_section_list.insert(0, ["entity", "score", "description"])
    entities_context = list_of_list_to_csv(entites_section_list)

    scorednode2chunk(ent_from_query_dict, scored_edged_reasoning_path)

    results = await chunks_vdb.query(originalquery, top_k=int(query_param.top_k / 2))
    chunks_ids = [r["id"] for r in results]
    final_chunk_id = kwd2chunk(
        ent_from_query_dict, chunks_ids, chunk_nums=int(query_param.top_k / 2)
    )

    if not len(results_edge):
        return None

    use_text_units = await asyncio.gather(
        *[text_chunks_db.get_by_id(id) for id in final_chunk_id]
    )
    text_units_section_list = [["id", "content"]]

    for i, t in enumerate(use_text_units):
        if t is not None:
            text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


async def minirag_query(  # MiniRAG
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    entity_name_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    embedder,
    query_param: QueryParam,
    global_config: dict,
) -> str:
    use_model_func = global_config["llm_model_func"]
    kw_prompt_temp = PROMPTS["minirag_query2kwd"]
    TYPE_POOL, TYPE_POOL_w_CASE = await knowledge_graph_inst.get_types()
    kw_prompt = kw_prompt_temp.format(query=query, TYPE_POOL=TYPE_POOL)
    result = await use_model_func(kw_prompt)

    try:
        keywords_data = json_repair.loads(result)
    except json.JSONDecodeError:
        # attempt crude salvage
        try:
            cleaned = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            if "{" in cleaned and "}" in cleaned:
                inner = cleaned.split("{",1)[1].rsplit("}",1)[0]
                keywords_data = json_repair.loads("{"+inner+"}")
            else:
                keywords_data = {}
        except Exception as e:
            print(f"JSON parsing error: {e}")
            keywords_data = {}

    # Normalize structure: sometimes model returns a list of dicts
    if isinstance(keywords_data, list):
        picked = None
        for item in keywords_data:
            if isinstance(item, dict) and ("answer_type_keywords" in item or "entities_from_query" in item):
                picked = item
                break
        keywords_data = picked if isinstance(picked, dict) else {}
    elif not isinstance(keywords_data, dict):
        keywords_data = {}

    type_keywords = keywords_data.get("answer_type_keywords") or []
    if not isinstance(type_keywords, list):
        type_keywords = []
    entities_from_query = keywords_data.get("entities_from_query") or []
    if not isinstance(entities_from_query, list):
        entities_from_query = []
    entities_from_query = entities_from_query[:5]

    # if not entities_from_query and not type_keywords:
    #     # Fallback: extract simple capitalized tokens as pseudo entities
    #     import re as _re
    #     candidates = list({_re.sub(r"[^A-Za-z0-9_ ]","", w).strip() for w in query.split() if w.istitle()})
    #     entities_from_query = candidates[:3]

    context = await _build_mini_query_context(
        entities_from_query,
        type_keywords,
        query,
        knowledge_graph_inst,
        entities_vdb,
        entity_name_vdb,
        relationships_vdb,
        chunks_vdb,
        text_chunks_db,
        embedder,
        query_param,
    )

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    
    
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )

    # Extract entities section from context
    entities_from_response = ""
    if context:
        import re
        entities_match = re.search(r"-----Entities-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL)
        if entities_match:
            # Process CSV to remove score column
            csv_content = entities_match.group(1)
            lines = csv_content.strip().split('\n')
            processed_lines = []
            
            for line in lines:
                parts = line.split(',')
                if len(parts) >= 3:
                    # Remove the score column (index 1) and keep entity (index 0) and description (index 2)
                    processed_parts = [parts[0], parts[2]]
                    processed_lines.append(','.join(processed_parts))
                else:
                    # Keep line as is if it doesn't have enough parts
                    processed_lines.append(line)
            
            processed_csv = '\n'.join(processed_lines)
            entities_from_response = f"-----Entities-----\n```csv\n{processed_csv}\n```"

    # put response
    output = f"-----Query Engineered-----\n" + result + "\n" + entities_from_response + "\n -----LLM Response:---- \n\n" + response

    return response

async def meta_query(
    query: str,
    doc_status_storage: BaseKVStorage,
    chunks_vdb: BaseVectorStorage,  # unused but kept for signature parity
    text_chunks_db: BaseKVStorage[TextChunkSchema],  # unused but kept
    knowledge_graph_inst: BaseGraphStorage,  # unused but kept
    query_param: QueryParam,
    global_config: dict,
):
    """Query documents by metadata.

    The incoming "query" should contain a metadata filter either as:
      1. JSON object string, e.g. '{"author": "Bob", "tags": ["ai","rag"]}'
      2. Simple key=value pairs separated by commas / semicolons, e.g. 'author=Bob,year=2024'

    Matching rules (subset semantics):
      - Scalar vs scalar: equality
      - Scalar expected vs list doc value: expected element must be in list
      - List expected vs scalar doc value: doc value must be in expected list
      - List expected vs list doc value: any intersection

    Returns a markdown table summarizing matched documents (max top_k) or,
    if query_param.only_need_context is True, a JSON string of raw matches.
    """
    import json

    # 1. Parse metadata filter
    metadata_filter: dict = {}
    q = (query or "").strip()
    if not q:
        return "No metadata filter provided"
    try:
        if q.startswith("{") and q.endswith("}"):
            metadata_filter = json.loads(q)
        else:
            # Parse key=value pairs
            parts = [p.strip() for p in re.split(r"[,;]", q) if p.strip()]
            for part in parts:
                if "=" not in part:
                    continue
                k, v = part.split("=", 1)
                k = k.strip()
                v = v.strip()
                # Try JSON decode each value for list/number/bool
                try:
                    v_dec = json.loads(v)
                    metadata_filter[k] = v_dec
                except Exception:
                    metadata_filter[k] = v
    except Exception:
        return "Failed to parse metadata filter"

    if not metadata_filter:
        return "Empty metadata filter"

    # 2. Fetch all doc ids
    try:
        all_doc_ids = await doc_status_storage.all_keys()
    except Exception:
        return "Failed to read documents"

    if not all_doc_ids:
        return "No documents available"

    # 3. Matching helper replicating semantics in MiniRAG._matches_metadata_filter
    def matches(doc_meta: dict, flt: dict) -> bool:
        for k, expected in flt.items():
            if k not in doc_meta:
                return False
            dv = doc_meta[k]
            if isinstance(expected, list) and isinstance(dv, list):
                if not any(item in dv for item in expected):
                    return False
            elif isinstance(expected, list):
                if dv not in expected:
                    return False
            elif isinstance(dv, list):
                if expected not in dv:
                    return False
            else:
                if dv != expected:
                    return False
        return True

    # 4. Gather matches (respect top_k)
    matched = []
    for doc_id in all_doc_ids:
        if len(matched) >= query_param.top_k:
            break
        try:
            d = await doc_status_storage.get_by_id(doc_id)
        except Exception:
            continue
        if not d:
            continue
        if matches(d.get("metadata", {}) or {}, metadata_filter):
            matched.append({
                "doc_id": doc_id,
                "file_path": d.get("file_path"),
                "created_at": d.get("created_at"),
                "updated_at": d.get("updated_at"),
                "status": d.get("status"),
                "content_length": d.get("content_length"),
                "chunks_count": d.get("chunks_count"),
                "metadata": d.get("metadata", {}),
            })

    if query_param.only_need_context:
        return json.dumps({"matches": matched, "filter": metadata_filter}, ensure_ascii=False, indent=2)

    if not matched:
        return "No documents matched the metadata filter"

    # 5. Build markdown table
    def truncate(v, n=80):
        s = str(v)
        return s if len(s) <= n else s[: n - 3] + "..."

    headers = ["doc_id", "status", "created_at", "chunks", "length", "file_path", "metadata"]
    rows = [" | ".join(headers), " | ".join(["-" * len(h) for h in headers])]
    for m in matched:
        rows.append(
            " | ".join([
                truncate(m.get("doc_id")),
                truncate(m.get("status")),
                truncate(m.get("created_at")),
                str(m.get("chunks_count", "")),
                str(m.get("content_length", "")),
                truncate(m.get("file_path")),
                truncate(json.dumps(m.get("metadata", {}), ensure_ascii=False)),
            ])
        )
    return "Matched Documents (metadata filter)\n\n" + "\n".join(rows)
