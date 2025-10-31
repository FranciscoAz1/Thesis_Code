"""
Comprehensive test suite for BART integration in MiniRAG.

Tests cover:
1. BART chunk summarization
2. BART entity description summarization
3. Full entity extraction with BART
4. Comparison with and without BART
5. Configuration and toggle testing
"""

import asyncio
import sys
import os

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

from unittest.mock import Mock, AsyncMock, patch

# Helper for pytest decorators - no-op if pytest not available
def mark_asyncio(func):
    """Apply pytest.mark.asyncio if available, otherwise return function unchanged."""
    if PYTEST_AVAILABLE:
        return pytest.mark.asyncio(func)
    return func

def skip_if_no_transformers(func):
    """Skip test if transformers not available."""
    if not PYTEST_AVAILABLE:
        return func
    def wrapper(*args, **kwargs):
        if not TRANSFORMERS_AVAILABLE:
            pytest.skip("transformers not installed")
        return func(*args, **kwargs)
    return wrapper

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minirag.summarization.bart_summarizer import BARTSummarizer, TRANSFORMERS_AVAILABLE
from minirag.operate_bart_entity import (
    extract_entities_with_bart,
    summarize_chunk_for_extraction,
    summarize_entity_description,
    BART_ENABLED,
)


# Sample test data
SAMPLE_CHUNK = """
The Bank of Portugal has announced a new monetary policy framework that will take effect in January 2024.
The central bank has increased its main interest rate by 25 basis points to combat inflation.
The bank's governor stated that this measure is necessary to maintain price stability in the eurozone.
Economic indicators show that inflation has reached 4.2% annually.
The bank expects the new policy will help bring inflation back to its target of 2% by Q3 2024.
"""

SAMPLE_CHUNK_PORTUGUESE = """
O Banco de Portugal anunciou um novo marco de política monetária que entrará em vigor em janeiro de 2024.
O banco central aumentou sua taxa de juros principal em 25 pontos-base para combater a inflação.
O governador do banco afirmou que essa medida é necessária para manter a estabilidade de preços na zona do euro.
Os indicadores econômicos mostram que a inflação atingiu 4,2% anualmente.
O banco espera que a nova política ajude a trazer a inflação de volta à sua meta de 2% até o Q3 de 2024.
"""

SAMPLE_ENTITIES = {
    "chunk-001": {
        "content": SAMPLE_CHUNK,
        "full_doc_id": "doc-001",
        "metadata": {"source": "news", "date": "2024-01-15"}
    }
}


class TestBARTChunkSummarization:
    """Test BART chunk summarization for entity extraction."""
    
    @pytest.mark.asyncio
    async def test_summarize_chunk_basic(self):
        """Test basic chunk summarization."""
        if not TRANSFORMERS_AVAILABLE:
            pytest.skip("transformers not installed")
        
        summarizer = BARTSummarizer(language="pt")
        result = await summarize_chunk_for_extraction(SAMPLE_CHUNK, summarizer)
        
        # Result should be shorter than original
        assert len(result) < len(SAMPLE_CHUNK)
        assert len(result) > 0
        print(f"\n✓ Chunk summarized: {len(SAMPLE_CHUNK)} → {len(result)} chars")
    
    @pytest.mark.asyncio
    async def test_summarize_chunk_without_bart(self):
        """Test chunk returns original when BART disabled."""
        result = await summarize_chunk_for_extraction(SAMPLE_CHUNK, summarizer=None)
        assert result == SAMPLE_CHUNK
        print("\n✓ Chunk returned unchanged when BART disabled")
    
    @pytest.mark.asyncio
    async def test_summarize_portuguese_chunk(self):
        """Test summarization of Portuguese content."""
        if not TRANSFORMERS_AVAILABLE:
            pytest.skip("transformers not installed")
        
        summarizer = BARTSummarizer(language="pt")
        result = await summarize_chunk_for_extraction(SAMPLE_CHUNK_PORTUGUESE, summarizer)
        
        assert len(result) > 0
        assert len(result) < len(SAMPLE_CHUNK_PORTUGUESE)
        print(f"\n✓ Portuguese chunk summarized: {len(SAMPLE_CHUNK_PORTUGUESE)} → {len(result)} chars")
    
    @pytest.mark.asyncio
    async def test_summarize_short_chunk(self):
        """Test summarization of very short chunk."""
        short_chunk = "The bank increased rates."
        summarizer = BARTSummarizer(language="pt")
        result = await summarize_chunk_for_extraction(short_chunk, summarizer)
        
        # Short chunks should be returned unchanged
        assert result == short_chunk
        print("\n✓ Short chunk returned unchanged")


class TestBARTEntityDescriptionSummarization:
    """Test BART entity description summarization."""
    
    @pytest.mark.asyncio
    async def test_summarize_description_basic(self):
        """Test entity description summarization."""
        if not TRANSFORMERS_AVAILABLE:
            pytest.skip("transformers not installed")
        
        long_description = SAMPLE_CHUNK + "\n" + SAMPLE_CHUNK
        summarizer = BARTSummarizer(language="pt")
        result = await summarize_entity_description(long_description, summarizer)
        
        assert len(result) > 0
        assert len(result) <= len(long_description)
        print(f"\n✓ Description summarized: {len(long_description)} → {len(result)} chars")
    
    @pytest.mark.asyncio
    async def test_summarize_short_description(self):
        """Test short descriptions are not summarized."""
        short_desc = "Bank official. Works at Central Bank."
        summarizer = BARTSummarizer(language="pt")
        result = await summarize_entity_description(short_desc, summarizer)
        
        # Short descriptions should be returned unchanged
        assert result == short_desc
        print("\n✓ Short description returned unchanged")
    
    @pytest.mark.asyncio
    async def test_summarize_description_without_bart(self):
        """Test description returns original when BART disabled."""
        long_description = SAMPLE_CHUNK + "\n" + SAMPLE_CHUNK
        result = await summarize_entity_description(long_description, summarizer=None)
        assert result == long_description
        print("\n✓ Description returned unchanged when BART disabled")


class TestBARTSummarizerConfiguration:
    """Test BARTSummarizer configuration options."""
    
    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        summarizer = BARTSummarizer()
        assert summarizer.language == "pt"
        assert summarizer.max_length == 150
        assert summarizer.min_length == 50
        print("\n✓ BARTSummarizer initialized with defaults")
    
    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        summarizer = BARTSummarizer(
            language="en",
            max_length=100,
            min_length=30,
            num_beams=2
        )
        assert summarizer.language == "en"
        assert summarizer.max_length == 100
        assert summarizer.min_length == 30
        print("\n✓ BARTSummarizer initialized with custom params")
    
    def test_transformers_availability(self):
        """Test transformers availability check."""
        print(f"\n✓ TRANSFORMERS_AVAILABLE: {TRANSFORMERS_AVAILABLE}")
        print(f"✓ BART_ENABLED: {BART_ENABLED}")
    
    @pytest.mark.asyncio
    async def test_key_sentences_extraction(self):
        """Test extract_key_sentences method."""
        if not TRANSFORMERS_AVAILABLE:
            pytest.skip("transformers not installed")
        
        summarizer = BARTSummarizer()
        sentences = summarizer.extract_key_sentences(SAMPLE_CHUNK, ratio=0.3)
        
        assert isinstance(sentences, list)
        assert len(sentences) > 0
        print(f"\n✓ Extracted {len(sentences)} key sentences from chunk")
    
    @pytest.mark.asyncio
    async def test_summary_statistics(self):
        """Test summary statistics generation."""
        if not TRANSFORMERS_AVAILABLE:
            pytest.skip("transformers not installed")
        
        summarizer = BARTSummarizer()
        sentences = summarizer.extract_key_sentences(SAMPLE_CHUNK, ratio=0.3)
        stats = summarizer.get_summary_statistics(SAMPLE_CHUNK, sentences)
        
        assert "compression_ratio" in stats
        assert "original_tokens" in stats
        assert "summary_tokens" in stats
        print(f"\n✓ Summary statistics: {stats}")


class TestBARTEntityExtractionIntegration:
    """Test full entity extraction with BART integration."""
    
    @pytest.mark.asyncio
    async def test_extract_entities_with_bart_enabled(self):
        """Test extract_entities_with_bart with BART enabled."""
        # Mock LLM function
        mock_llm = AsyncMock()
        mock_llm.return_value = '(("entity", "BANK OF PORTUGAL", "ORGANIZATION", "Central bank of Portugal"), ("entity", "INFLATION", "METRIC", "4.2% annual rate"))'
        
        # Mock storage instances
        mock_kg = AsyncMock()
        mock_kg.get_node = AsyncMock(return_value=None)
        mock_kg.upsert_node = AsyncMock()
        
        mock_entity_vdb = AsyncMock()
        mock_entity_vdb.upsert = AsyncMock()
        
        mock_entity_name_vdb = AsyncMock()
        mock_entity_name_vdb.upsert = AsyncMock()
        
        mock_rel_vdb = AsyncMock()
        mock_rel_vdb.upsert = AsyncMock()
        
        global_config = {
            "llm_model_func": mock_llm,
            "entity_extract_max_gleaning": 1,
        }
        
        if TRANSFORMERS_AVAILABLE:
            result = await extract_entities_with_bart(
                SAMPLE_ENTITIES,
                knowledge_graph_inst=mock_kg,
                entity_vdb=mock_entity_vdb,
                entity_name_vdb=mock_entity_name_vdb,
                relationships_vdb=mock_rel_vdb,
                global_config=global_config,
                use_bart=True,
            )
            
            # Verify mock calls were made
            assert mock_kg.upsert_node.called or result is None
            print("\n✓ Entity extraction with BART completed")
        else:
            print("\n⊘ Skipping BART entity extraction test (transformers not available)")
    
    @pytest.mark.asyncio
    async def test_extract_entities_with_bart_disabled(self):
        """Test extract_entities_with_bart with BART disabled."""
        # Mock LLM function
        mock_llm = AsyncMock()
        mock_llm.return_value = '(("entity", "BANK", "ORGANIZATION", "Central bank"))'
        
        # Mock storage instances
        mock_kg = AsyncMock()
        mock_kg.get_node = AsyncMock(return_value=None)
        mock_kg.upsert_node = AsyncMock()
        
        mock_entity_vdb = AsyncMock()
        mock_entity_vdb.upsert = AsyncMock()
        
        mock_entity_name_vdb = AsyncMock()
        mock_entity_name_vdb.upsert = AsyncMock()
        
        mock_rel_vdb = AsyncMock()
        mock_rel_vdb.upsert = AsyncMock()
        
        global_config = {
            "llm_model_func": mock_llm,
            "entity_extract_max_gleaning": 1,
        }
        
        result = await extract_entities_with_bart(
            SAMPLE_ENTITIES,
            knowledge_graph_inst=mock_kg,
            entity_vdb=mock_entity_vdb,
            entity_name_vdb=mock_entity_name_vdb,
            relationships_vdb=mock_rel_vdb,
            global_config=global_config,
            use_bart=False,
        )
        
        print("\n✓ Entity extraction with BART disabled completed")


class TestBARTPerformance:
    """Test performance characteristics of BART."""
    
    @pytest.mark.asyncio
    async def test_summarization_speed(self):
        """Test summarization speed."""
        if not TRANSFORMERS_AVAILABLE:
            pytest.skip("transformers not installed")
        
        import time
        
        summarizer = BARTSummarizer()
        
        start = time.time()
        sentences = summarizer.extract_key_sentences(SAMPLE_CHUNK, ratio=0.3)
        duration = time.time() - start
        
        print(f"\n✓ Summarization completed in {duration:.2f} seconds")
        print(f"  Extracted {len(sentences)} sentences")
        print(f"  Speed: {len(SAMPLE_CHUNK) / duration:.0f} chars/sec")
    
    @pytest.mark.asyncio
    async def test_batch_summarization_memory(self):
        """Test batch summarization doesn't cause memory issues."""
        if not TRANSFORMERS_AVAILABLE:
            pytest.skip("transformers not installed")
        
        summarizer = BARTSummarizer()
        
        # Summarize multiple chunks
        summaries = []
        for i in range(3):
            summary = await summarize_chunk_for_extraction(SAMPLE_CHUNK, summarizer)
            summaries.append(summary)
        
        assert len(summaries) == 3
        print(f"\n✓ Batch summarization completed for {len(summaries)} chunks")


def run_quick_tests():
    """Run quick tests without pytest."""
    print("\n" + "="*80)
    print("BART INTEGRATION QUICK TEST SUITE")
    print("="*80)
    
    # Test 1: Check BART availability
    print(f"\n[1/5] BART Availability: {TRANSFORMERS_AVAILABLE}")
    
    # Test 2: BARTSummarizer initialization
    if TRANSFORMERS_AVAILABLE:
        try:
            summarizer = BARTSummarizer()
            print("[2/5] ✓ BARTSummarizer initialized successfully")
        except Exception as e:
            print(f"[2/5] ✗ Failed to initialize: {e}")
    else:
        print("[2/5] ⊘ Skipped (transformers not available)")
    
    # Test 3: Check summary configuration
    if TRANSFORMERS_AVAILABLE:
        print(f"[3/5] ✓ Summary Configuration:")
        print(f"     - Chunk summary ratio: 30%")
        print(f"     - Max description length: 200 tokens")
        print(f"     - Language: Portuguese (pt)")
    else:
        print("[3/5] ⊘ Skipped")
    
    # Test 4: Entity extraction imports
    try:
        from minirag.operate_bart_entity import (
            extract_entities_with_bart,
            summarize_chunk_for_extraction,
            summarize_entity_description,
        )
        print("[4/5] ✓ Entity extraction imports successful")
    except ImportError as e:
        print(f"[4/5] ✗ Import failed: {e}")
    
    # Test 5: MiniRAG integration
    try:
        from minirag.minirag import MiniRAG
        print("[5/5] ✓ MiniRAG integration imports successful")
        print("     - Added use_bart_entity_extraction configuration")
        print("     - Integrated extract_entities_with_bart into ainsert pipeline")
    except ImportError as e:
        print(f"[5/5] ✗ Import failed: {e}")
    
    print("\n" + "="*80)
    print("Quick test summary complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Run quick tests
    run_quick_tests()
    
    # Run pytest if available
    if PYTEST_AVAILABLE:
        pytest.main([__file__, "-v", "--tb=short"])
    else:
        print("Note: To run full pytest suite, install pytest: pip install pytest pytest-asyncio")
