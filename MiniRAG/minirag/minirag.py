import asyncio
import os
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
import sys
from typing import Type, cast, Any
from dotenv import load_dotenv
from minirag.prompt import PROMPTS
from .metadata_plugin import minirag_generate_metadata, MetadataExtractor


from .operate import (
    chunking_by_token_size,
    extract_entities,
    doc_query,
    hybrid_query,
    minirag_query,
    naive_query,
    meta_query,
    presidio_entity_extraction,
)
from .operate_bart_entity import extract_entities_with_bart

from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    logger,
    clean_text,
    get_content_summary,
    set_logger,
    logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
    DocStatus,
)


STORAGES = {
    "NetworkXStorage": ".kg.networkx_impl",
    "JsonKVStorage": ".kg.json_kv_impl",
    "NanoVectorDBStorage": ".kg.nano_vector_db_impl",
    "JsonDocStatusStorage": ".kg.jsondocstatus_impl",
    "Neo4JStorage": ".kg.neo4j_impl",
    "OracleKVStorage": ".kg.oracle_impl",
    "OracleGraphStorage": ".kg.oracle_impl",
    "OracleVectorDBStorage": ".kg.oracle_impl",
    "MilvusVectorDBStorge": ".kg.milvus_impl",
    "MongoKVStorage": ".kg.mongo_impl",
    "MongoGraphStorage": ".kg.mongo_impl",
    "RedisKVStorage": ".kg.redis_impl",
    "ChromaVectorDBStorage": ".kg.chroma_impl",
    "TiDBKVStorage": ".kg.tidb_impl",
    "TiDBVectorDBStorage": ".kg.tidb_impl",
    "TiDBGraphStorage": ".kg.tidb_impl",
    "PGKVStorage": ".kg.postgres_impl",
    "PGVectorStorage": ".kg.postgres_impl",
    "AGEStorage": ".kg.age_impl",
    "PGGraphStorage": ".kg.postgres_impl",
    "GremlinStorage": ".kg.gremlin_impl",
    "PGDocStatusStorage": ".kg.postgres_impl",
    "WeaviateVectorStorage": ".kg.weaviate_impl",
    "WeaviateKVStorage": ".kg.weaviate_impl",
    "WeaviateGraphStorage": ".kg.weaviate_impl",
    "run_sync": ".kg.weaviate_impl",
}

# future KG integrations

# from .kg.ArangoDB_impl import (
#     GraphStorage as ArangoDBStorage
# )

load_dotenv(dotenv_path=".env", override=False)

def lazy_external_import(module_name: str, class_name: str):
    """Lazily import a class from an external module based on the package of the caller."""

    # Get the caller's module and package
    import inspect

    caller_frame = inspect.currentframe().f_back
    module = inspect.getmodule(caller_frame)
    package = module.__package__ if module else None

    def import_class(*args, **kwargs):
        import importlib

        module = importlib.import_module(module_name, package=package)
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)

    return import_class


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        # Try to get the current event loop
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # If no event loop exists or it is closed, create a new one
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


@dataclass
class MiniRAG:
    # Use a timestamp safe for Windows paths (replace ':' with '-')
    working_dir: str = field(
        default_factory=lambda: f"./minirag_cache_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )

    # RAGmode: str = 'minirag'
    kv_storage: str = field(default="JsonKVStorage")
    vector_storage: str = field(default="NanoVectorDBStorage")
    graph_storage: str = field(default="NetworkXStorage")

    current_log_level = logger.level
    log_level: str = field(default=current_log_level)

    # text chunking
    chunk_token_size: int = 200
    chunk_overlap_token_size: int = 5
    tiktoken_model_name: str = "gpt-4o-mini"

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500
    entity_presidio_extraction: bool = False
    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    embedding_func: EmbeddingFunc = None
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    # LLM
    llm_model_func: callable = None
    llm_model_name: str = (
        "meta-llama/Llama-3.2-1B-Instruct"  #'meta-llama/Llama-3.2-1B'#'google/gemma-2-2b-it'
    )
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 16
    llm_model_kwargs: dict = field(default_factory=dict)

    # storage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)

    enable_llm_cache: bool = True

    # extension
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    # BART entity extraction configuration
    use_bart_entity_extraction: bool = field(default=True)  # Enable BART-based entity extraction with chunk summarization

    key_sentences_lexrank: bool = field(default=True)  # Use LexRank for entity extraction
    lexrank_ratio: float = field(default=0.3)  # Ratio of sentences to extract with LexRank

    # Document summarization before chunking
    summarize_before_chunking: bool = field(default=True)  # Summarize document before chunking to reduce chunk count
    document_summary_ratio: float = field(default=0.3)  # Ratio of document to retain in summary (0.1-0.5)

    # Add new field for document status storage type
    doc_status_storage: str = field(default="JsonDocStatusStorage")

    # Custom Chunking Function
    chunking_func: callable = chunking_by_token_size
    chunking_func_kwargs: dict = field(default_factory=dict)

    max_parallel_insert: int = field(default=int(os.getenv("MAX_PARALLEL_INSERT", 2)))
    # Suppress noisy external HTTP client logs (httpx / httpcore / ollama)
    suppress_httpx_logging: bool = True
    
    # BART summarizer instance (initialized in __post_init__)
    summarizer: Any = field(default=None, init=False, repr=False)
    
    # LexRank summarizer instance (initialized in __post_init__)
    lexrank_summarizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        log_file = os.path.join(self.working_dir, "minirag.log")
        set_logger(log_file)
        logger.setLevel(self.log_level)

        logger.info(f"Logger initialized for working directory: {self.working_dir}")
        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)
        
        # Optionally silence verbose third-party HTTP client loggers
        if self.suppress_httpx_logging:
            for _ln in ("httpx", "httpcore", "ollama"):
                _lg = logging.getLogger(_ln)
                _lg.setLevel(logging.CRITICAL)
                _lg.propagate = False
                # Ensure existing handlers (if any) are raised to CRITICAL
                for _h in list(_lg.handlers):
                    try:
                        _h.setLevel(logging.CRITICAL)
                    except Exception:
                        pass

        # show config
        global_config = asdict(self)
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in global_config.items()])
        logger.debug(f"MiniRAG init with param:\n  {_print_config}\n")

        # @TODO: should move all storage setup here to leverage initial start params attached to self.

        self.key_string_value_json_storage_cls: Type[BaseKVStorage] = (
            self._get_storage_class(self.kv_storage)
        )
        self.vector_db_storage_cls: Type[BaseVectorStorage] = self._get_storage_class(
            self.vector_storage
        )
        self.graph_storage_cls: Type[BaseGraphStorage] = self._get_storage_class(
            self.graph_storage
        )

        self.key_string_value_json_storage_cls = partial(
            self.key_string_value_json_storage_cls, global_config=global_config
        )

        self.vector_db_storage_cls = partial(
            self.vector_db_storage_cls, global_config=global_config
        )

        self.graph_storage_cls = partial(
            self.graph_storage_cls, global_config=global_config
        )
        self.json_doc_status_storage = self.key_string_value_json_storage_cls(
            namespace="json_doc_status_storage",
            embedding_func=None,
        )

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache",
                global_config=asdict(self),
                embedding_func=None,
            )
            if self.enable_llm_cache
            else None
        )

        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )

        ####
        # add embedding func by walter
        ####
        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        ####
        # add embedding func by walter over
        ####

        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )
        global_config = asdict(self)

        self.entity_name_vdb = self.vector_db_storage_cls(
            namespace="entities_name",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )

        self.relationships_vdb = self.vector_db_storage_cls(
            namespace="relationships",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id"},
        )
        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )

        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,
                hashing_kv=self.llm_response_cache,
                **self.llm_model_kwargs,
            )
        )
        # BM25 index state (lazy build). Keys: index, dirty
        self._bm25_index_state = {"index": None, "dirty": True}
        # Initialize document status storage
        self.doc_status_storage_cls = self._get_storage_class(self.doc_status_storage)
        self.doc_status = self.doc_status_storage_cls(
            namespace="doc_status",
            global_config=global_config,
            embedding_func=None,
        )

        # Initialize metadata extractor with lazy-loaded analyzer
        self.metadata_extractor = MetadataExtractor()
        
        # Initialize BART summarizer if enabled
        self.summarizer = None
        if self.use_bart_entity_extraction:
            try:
                from .summarization.bart_summarizer import BARTSummarizer, TRANSFORMERS_AVAILABLE
                if TRANSFORMERS_AVAILABLE:
                    self.summarizer = BARTSummarizer(language="pt")
                    logger.info("BART summarizer initialized for entity description summarization")
            except Exception as e:
                logger.warning(f"Failed to initialize BART: {e}, proceeding without BART")
                self.summarizer = None
        
        # Initialize LexRank summarizer if enabled
        self.lexrank_summarizer = None
        if self.key_sentences_lexrank:
            try:
                from .summarization.lexrank_summarizer import DocumentSummarizer
                self.lexrank_summarizer = DocumentSummarizer(language="pt")
                logger.info(f"LexRank summarizer initialized for entity extraction (ratio: {self.lexrank_ratio})")
            except Exception as e:
                logger.warning(f"Failed to initialize LexRank: {e}, falling back to standard extraction")
                self.lexrank_summarizer = None

    def _get_storage_class(self, storage_name: str) -> dict:
        import_path = STORAGES[storage_name]
        storage_class = lazy_external_import(import_path, storage_name)
        return storage_class

    def set_storage_client(self, db_client):
        # Now only tested on Oracle Database
        for storage in [
            self.vector_db_storage_cls,
            self.graph_storage_cls,
            self.doc_status,
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.key_string_value_json_storage_cls,
            self.chunks_vdb,
            self.relationships_vdb,
            self.entities_vdb,
            self.graph_storage_cls,
            self.chunk_entity_relation_graph,
            self.llm_response_cache,
        ]:
            # set client
            storage.db = db_client

    def insert(self, string_or_strings, metadata=None):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings, metadata=metadata))

    async def ainsert(
        self,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
        file_path: str | None = None,
        metadata: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> None:
        if isinstance(input, str):
            input = [input]
        if isinstance(ids, str):
            ids = [ids]

        await self.apipeline_enqueue_documents(input, ids, file_path, metadata)
        await self.apipeline_process_enqueue_documents(
            split_by_character, split_by_character_only
        )
        # Generate and attach metadata using the metadata plugin

        # Perform additional entity extraction as per original ainsert logic
        inserting_chunks = {
            compute_mdhash_id(dp["content"], prefix="chunk-"): {
                **dp,
                "full_doc_id": doc_id,
            }
            for doc_id, status_doc in (
                await self.doc_status.get_docs_by_status(DocStatus.PROCESSED)
            ).items()
            for dp in self.chunking_func(
                status_doc.content,
                self.chunk_overlap_token_size,
                self.chunk_token_size,
                self.tiktoken_model_name,
            )
        }
        if inserting_chunks:
            logger.info("Performing entity extraction on newly processed chunks")
            
            if self.entity_presidio_extraction:
                logger.info("Using Presidio for entity extraction")
                await presidio_entity_extraction(
                    inserting_chunks,
                    knowledge_graph_inst=self.chunk_entity_relation_graph,
                    entity_vdb=self.entities_vdb,
                    entity_name_vdb=self.entity_name_vdb,
                    relationships_vdb=self.relationships_vdb,
                    global_config={**asdict(self), "metadata_extractor": self.metadata_extractor},
                )
            else:
                logger.info("Using entity extraction" + (" with BART description summarization" if self.summarizer else ""))
                await extract_entities(     
                    inserting_chunks,
                    knowledge_graph_inst=self.chunk_entity_relation_graph,
                    entity_vdb=self.entities_vdb,
                    entity_name_vdb=self.entity_name_vdb,
                    relationships_vdb=self.relationships_vdb,
                    global_config=asdict(self),
                )
        # Mark BM25 index dirty after any insertion
        if hasattr(self, "_bm25_index_state"):
            self._bm25_index_state["dirty"] = True
 
        await self._insert_done()

    async def apipeline_enqueue_documents(
        self, input: str | list[str], ids: list[str] | None = None, file_path: str | None = None, metadata: dict[str, Any] | list[dict[str, Any]] | None = None
    ) -> None:
        """
        Pipeline for Processing Documents

        1. Validate ids if provided or generate MD5 hash IDs
        2. Remove duplicate contents
        3. Generate document initial status
        4. Filter out already processed documents
        5. Enqueue document in status
        """
        if isinstance(input, str):
            input = [input]
        if isinstance(ids, str):
            ids = [ids]
        if isinstance(metadata, dict):
            metadata = [metadata]

        # Validate metadata matches input length
        if metadata is not None:
            if len(metadata) != len(input):
                raise ValueError("Number of metadata dictionaries must match the number of documents")
        else:
            metadata = [{}] * len(input)  # Default empty metadata for each document

        if ids is not None:
            if len(ids) != len(input):
                raise ValueError("Number of IDs must match the number of documents")
            if len(ids) != len(set(ids)):
                raise ValueError("IDs must be unique")
            contents = {id_: (doc, meta) for id_, doc, meta in zip(ids, input, metadata)}
        else:
            # Create unique pairs using string representation for deduplication
            input_meta_pairs = list(set((clean_text(doc), str(meta)) for doc, meta in zip(input, metadata)))
            contents = {compute_mdhash_id(doc + str(meta), prefix="doc-"): (doc, eval(meta)) for doc, meta in input_meta_pairs}

        # Remove duplicates by using content + metadata string representation as key
        seen_content_meta = {}
        unique_contents = {}
        for id_, (content, meta) in contents.items():
            # Create a hashable key using content + string representation of metadata
            content_meta_key = (content, str(meta))
            if content_meta_key not in seen_content_meta:
                seen_content_meta[content_meta_key] = id_
                unique_contents[id_] = (content, meta)
        new_docs: dict[str, Any] = {
            id_: {
                "file_path": file_path,
                "content": content,
                "content_summary": get_content_summary(content),
                "content_length": len(content),
                "status": DocStatus.PENDING,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "metadata": meta,
            }
            for id_, (content, meta) in unique_contents.items()
        }

        all_new_doc_ids = set(new_docs.keys())
        unique_new_doc_ids = await self.doc_status.filter_keys(all_new_doc_ids)

        new_docs = {
            doc_id: new_docs[doc_id]
            for doc_id in unique_new_doc_ids
            if doc_id in new_docs
        }
        if not new_docs:
            logger.info("No new unique documents were found.")
            return

        await self.doc_status.upsert(new_docs)
        logger.info(f"Stored {len(new_docs)} new unique documents")

    async def _summarize_document_before_chunking(
        self,
        doc_id: str,
        content: str,
    ) -> str:
        """
        Optionally summarize document before chunking to reduce chunk count.
        
        Strategy (in priority order):
        1. Both BART & LexRank: Apply BART → LexRank pipeline
        2. Only LexRank: Apply LexRank extraction
        3. Only BART: Apply BART summarization
        4. Neither: Use original content
        
        Args:
            doc_id: Document identifier for logging
            content: Original document content
            
        Returns:
            Summarized content (or original if disabled/failed)
        """
        if not self.summarize_before_chunking:
            return content
        
        # Check which summarizers are available
        has_bart = self.summarizer is not None
        has_lexrank = self.key_sentences_lexrank and self.lexrank_summarizer is not None
        
        try:
            if has_bart and has_lexrank:
                # Both enabled: LexRank → BART pipeline for ultra-condensed summary
                logger.warning(f"Both BART and LexRank enabled for {doc_id}. Applying LexRank→BART pipeline")
                
                # Step 1: LexRank extraction
                logger.info(f"  Step 1/2: LexRank extraction (ratio: {self.lexrank_ratio})")
                lexrank_summary_sentences = self.lexrank_summarizer.extract_key_sentences(
                    content, ratio=self.lexrank_ratio
                )
                lexrank_summary = " ".join(lexrank_summary_sentences)
                
                # Step 2: BART on LexRank summary
                logger.info(f"  Step 2/2: BART abstractive summarization on LexRank result (ratio: {self.document_summary_ratio})")
                final_summary_sentences = self.summarizer.extract_key_sentences(
                    lexrank_summary, ratio=self.document_summary_ratio
                )
                content_to_chunk = " ".join(final_summary_sentences)
                
            elif has_lexrank:
                # Only LexRank: Single-pass extraction
                logger.info(f"Using LexRank for {doc_id} (ratio: {self.lexrank_ratio})")
                summary_sentences = self.lexrank_summarizer.extract_key_sentences(
                    content, ratio=self.lexrank_ratio
                )
                content_to_chunk = " ".join(summary_sentences)
                
            elif has_bart:
                # Only BART: Single-pass summarization
                logger.info(f"Using BART for {doc_id} (ratio: {self.document_summary_ratio})")
                summary_sentences = self.summarizer.extract_key_sentences(
                    content, ratio=self.document_summary_ratio
                )
                content_to_chunk = " ".join(summary_sentences)
                
            else:
                # Neither available
                logger.warning(f"Summarization requested for {doc_id} but no summarizer available, using original content")
                return content
            
            # Log reduction statistics
            original_len = len(content)
            summary_len = len(content_to_chunk)
            reduction = 1.0 - (summary_len / original_len) if original_len > 0 else 0
            logger.info(f"Document {doc_id} summarized: {original_len} → {summary_len} chars ({reduction:.1%} reduction)")
            
            return content_to_chunk
            
        except Exception as e:
            logger.warning(f"Document summarization failed for {doc_id}: {e}, using original content")
            return content

    async def apipeline_process_enqueue_documents(
        self,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
    ) -> None:
        """
        Process pending documents by splitting them into chunks, processing
        each chunk for entity and relation extraction, embedding chunks to vector database, and updating the
        document status.
        """
        processing_docs, failed_docs, pending_docs = await asyncio.gather(
            self.doc_status.get_docs_by_status(DocStatus.PROCESSING),
            self.doc_status.get_docs_by_status(DocStatus.FAILED),
            self.doc_status.get_docs_by_status(DocStatus.PENDING),
        )

        to_process_docs: dict[str, Any] = {
            **processing_docs,
            **failed_docs,
            **pending_docs,
        }
        if not to_process_docs:
            logger.info("No documents to process")
            return

        docs_batches = [
            list(to_process_docs.items())[i : i + self.max_parallel_insert]
            for i in range(0, len(to_process_docs), self.max_parallel_insert)
        ]
        logger.info(f"Number of batches to process: {len(docs_batches)}")

        for batch_idx, docs_batch in enumerate(docs_batches):
            for doc_id, status_doc in docs_batch:
                # Step 1: Optionally summarize document before chunking
                logger.info(f"Processing document {doc_id} in batch {batch_idx + 1}/{len(docs_batches)}")
                content_to_chunk = await self._summarize_document_before_chunking(
                    doc_id, status_doc.content
                )
                # Step 2: Create chunks from content
                chunks = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_id,
                        "metadata": status_doc.metadata if hasattr(status_doc, 'metadata') else {},
                    }
                    for dp in self.chunking_func(
                        content_to_chunk,
                        self.chunk_overlap_token_size,
                        self.chunk_token_size,
                        self.tiktoken_model_name,
                    )
                }
                
                # Step 3: Upsert to vector databases
                await asyncio.gather(
                    self.chunks_vdb.upsert(chunks),
                    self.full_docs.upsert({doc_id: {"content": status_doc.content}}),
                    self.text_chunks.upsert(chunks),
                )
                
                # Step 4: Extract metadata if using Presidio
                if self.entity_presidio_extraction:
                    meta = minirag_generate_metadata(doc_id, status_doc.content, status_doc.file_path, self.metadata_extractor)
                
                # Step 5: Update document status
                await self.doc_status.upsert(
                    {
                        doc_id: {
                            "status": DocStatus.PROCESSED,
                            "chunks_count": len(chunks),
                            "content": status_doc.content,
                            "content_summary": status_doc.content_summary,
                            "content_length": status_doc.content_length,
                            "file_path": status_doc.file_path,
                            "metadata": getattr(status_doc, 'metadata', {}),
                            "created_at": status_doc.created_at,
                            "updated_at": datetime.now().isoformat(),
                        }
                    }
                )
        logger.info("Document processing pipeline completed")

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.entity_name_vdb,
            self.relationships_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        if param.mode == "light":
            response = await hybrid_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
        elif param.mode == "mini":
            response = await minirag_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.entity_name_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.text_chunks,
                self.embedding_func,
                param,
                asdict(self),
            )
        elif param.mode == "naive":
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
        elif param.mode == "doc":
            response = await doc_query(
                query,
                self.doc_status,
                self.chunks_vdb,
                self.text_chunks,
                self.chunk_entity_relation_graph,
                param,
                asdict(self),
            )
        elif param.mode == "meta":
            response = await meta_query(
                query,
                self.doc_status,
                self.chunks_vdb,
                self.text_chunks,
                self.chunk_entity_relation_graph,
                param,
                asdict(self),
            )
        elif param.mode == "bm25":
            from .operate import bm25_query  # local import to avoid cycles
            response = await bm25_query(
                query,
                self.text_chunks,
                param,
                asdict(self),
                self._bm25_index_state,
            )
        elif param.mode == "bypass":
            # Bypass mode: directly use LLM without knowledge retrieval
            use_llm_func = self.llm_model_func
            param.stream = True if param.stream is None else param.stream
            response = await use_llm_func(
                query.strip(),
                system_prompt=PROMPTS["rag_response"],
                history_messages=param.conversation_history,
                stream=param.stream,
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    async def query_by_metadata(self, metadata_filter: dict[str, Any]) -> list[dict]:
        """Query documents by metadata filter"""
        try:
            # Get all document IDs
            all_doc_ids = await self.doc_status.all_keys()
            
            if not all_doc_ids:
                return []
            
            # Filter documents by metadata
            matching_doc_ids = []
            
            for doc_id in all_doc_ids:
                # Get the document data
                doc_data = await self.doc_status.get_by_id(doc_id)
                
                if doc_data is not None:
                    doc_metadata = doc_data.get('metadata', {})
                    
                    # Check if document matches the metadata filter
                    if self._matches_metadata_filter(doc_metadata, metadata_filter):
                        matching_doc_ids.append(doc_id)
            
            if not matching_doc_ids:
                return []
            
            # Get the detailed document information
            results = []
            for doc_id in matching_doc_ids:
                doc_data = await self.doc_status.get_by_id(doc_id)
                if doc_data is not None:
                    results.append({
                        "doc_id": doc_id,
                        "content": doc_data.get("content", ""),
                        "metadata": doc_data.get("metadata", {}),
                        "file_path": doc_data.get("file_path"),
                        "content_summary": doc_data.get("content_summary", ""),
                        "created_at": doc_data.get("created_at"),
                        "status": doc_data.get("status")
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying by metadata: {e}")
            return []

    def _matches_metadata_filter(self, doc_metadata: dict[str, Any], filter_criteria: dict[str, Any]) -> bool:
        """Check if document metadata matches filter criteria"""
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

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def delete_by_entity(self, entity_name: str):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str):
        entity_name = f'"{entity_name.upper()}"'

        try:
            await self.entities_vdb.delete_entity(entity_name)
            await self.relationships_vdb.delete_relation(entity_name)
            await self.chunk_entity_relation_graph.delete_node(entity_name)

            logger.info(
                f"Entity '{entity_name}' and its relationships have been deleted."
            )
            await self._delete_by_entity_done()
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")

    async def _delete_by_entity_done(self):
        tasks = []
        for storage_inst in [
            self.entities_vdb,
            self.relationships_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)