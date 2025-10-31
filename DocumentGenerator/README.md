# Benchmark Framework - RAG System Evaluation

## Overview

This module provides a comprehensive benchmarking framework for evaluating Retrieval-Augmented Generation (RAG) systems. It tests multiple retrieval strategies (Naive, Chain-of-Thought, Agent-based) against a curated question-answer dataset and measures performance using state-of-the-art NLP metrics.

## Key Features

- **Multiple Retrieval Strategies**: Naive, CoT, Agent-based
- **Comprehensive Metrics**: ROUGE, BLEU, BERT similarity, retrieval accuracy
- **Portuguese Language Support**: Specialized evaluation for pt-BR
- **Document Generation**: Azure GPT-based synthetic document creation
- **Result Visualization**: Automated analysis and comparison reports
- **Reproducible**: Seeded random number generation

## Directory Structure

```
DocumentGenerator/
├── README.md                        # This file
├── .env                            # Environment configuration
├── requirements-visualize.txt      # Visualization dependencies
│
├── scripts/                        # Core scripts
│   ├── main.py                     # Document generation with Azure GPT
│   ├── create_qa.py                # Question-answer pair generation
│   ├── populate_curated_docs.py    # Document ingestion pipeline
│   └── test_connection.py          # Connection verification
│
├── notebooks/                      # Jupyter notebooks
│   ├── benchmark_weaviate.ipynb    # ⭐ Main benchmarking notebook
│   ├── benchmark_weaviate_openAI.ipynb  # OpenAI-specific benchmarks
│   ├── benchmark_light.ipynb       # Lightweight RAG comparison
│   └── benchmark_agent.ipynb       # Agent-specific evaluation
│
├── datasets/                       # Q&A datasets
│   ├── merged_qa_dataset.json      # Main Q&A dataset (358 pairs)
│   ├── qa_dataset20.json           # Small test set (20 pairs)
│   └── qa_dataset300.json          # Large evaluation set (300 pairs)
│
├── analysis/                       # Analysis and visualization
│   ├── analyze_results.py          # Statistical analysis
│   ├── compare_agent_answers.py    # Agent output comparison
│   └── visualize_summary.py        # Generate charts and plots
│
├── results_csv/                    # ⭐ All CSV results consolidated
│   ├── Agent_Answers.csv           # Agent evaluation results
│   ├── comparison_report.csv       # Comparison report
│   ├── results/                    # Benchmark results
│   │   ├── summary.json            # Aggregate statistics
│   │   ├── *_metric_medians.csv    # Median performance metrics
│   │   ├── *_metric_threshold_differences.csv  # Threshold analysis
│   │   ├── fixed_threshold_*.csv   # Fixed threshold results
│   │   └── final_top_*.csv         # Top configurations
│   ├── good_data/                  # Best performing configurations
│   │   ├── AgentOpenAI_300.csv     # Agent approach (300 questions)
│   │   ├── weaviateCoT_3_300.csv   # CoT with k=3
│   │   ├── weaviateNaive1_300.csv  # Naive with k=1
│   │   └── weaviateNaive3_300.csv  # Naive with k=3
│   ├── better_data/                # Improved configurations
│   ├── merged_data/                # Combined datasets
│   │   ├── Total_358_Mixed.csv     # Complete combined results
│   │   ├── MixedLiHua.csv          # LiHua benchmark subset
│   │   └── MixedSynthetic.csv      # Synthetic data subset
│   ├── results_merged/             # Merged results analysis
│   ├── results_summ/               # Summary statistics
│   ├── summary_data/               # Aggregated summaries
│   └── weaviate/                   # Weaviate-specific CSV results
│
├── curated_docs/                   # Curated legal documents
├── documentos_gerados/             # Generated synthetic documents
├── storage/                        # Temporary processing storage
└── summarization/                  # Text summarization modules
```

## Setup

### Prerequisites

1. **Python 3.10+** with Jupyter
2. **Weaviate** running (Docker)
3. **Ollama** with qwen2.5:latest model
4. **Azure OpenAI** API key (for document generation)

### Installation

```powershell
# Navigate to directory
cd "DocumentGenerator"

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements-visualize.txt
pip install weaviate-client sentence-transformers torch nltk rouge-score python-docx psutil tqdm
```

### Configuration

Create `.env` file:
```env
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4
```

## Usage

### Quick Start - Running a Benchmark

1. **Start Weaviate and Ollama**:
   ```powershell
   # Terminal 1
   cd ..\..\weaviate
   docker-compose up -d
   
   # Terminal 2
   ollama serve
   ollama pull qwen2.5:latest
   ```

2. **Open the main benchmark notebook**:
   ```powershell
   jupyter notebook notebooks/benchmark_weaviate.ipynb
   ```

3. **Execute cells in order**:
   - **Cell 0**: Initialize Weaviate collection
   - **Cell 1**: Warm up Ollama (optional)
   - **Cell 2**: Index documents
   - **Cell 3**: Test connectivity
   - **Cell 4**: Run evaluation

### Configuration Parameters

Edit these in Cell 4 of `notebooks/benchmark_weaviate.ipynb`:

```python
# Dataset
QA_JSON_PATH = r"C:\...\merged_qa_dataset.json"
OUTPUT_CSV_PATH = r"C:\...\output"

# Retrieval
TOP_K = 3              # Number of documents to retrieve
RANDOM_SEED = 42       # For reproducibility
USE_BERT_SIM = True    # Compute BERT similarity (slower)

# Prompting
COT_PROMPT = True      # Enable Chain-of-Thought
PROMPT_PREFIX = "Responda de forma breve e objetiva em português (pt-BR): "

# Performance
PER_QUERY_DEADLINE = 60.0  # Seconds per query
MAX_RETRIES = 3
RETRY_BACKOFF = 10.0
MAX_Q = None           # Limit questions (None = all)
```

## Evaluation Metrics

### Retrieval Metrics

1. **Document Retrieval Accuracy**: Percentage of queries where correct document is in top-K
2. **Average Retrieval Rank**: Average position of correct document in results
3. **Top-1/Top-3/Top-5 Accuracy**: Precision at different K values

### Answer Quality Metrics

#### Lexical Overlap
- **Exact Match**: Binary - does answer exactly match expected?
- **Substring Match**: Is expected answer contained in generated answer?
- **Token Recall**: Fraction of expected tokens present in answer
- **Jaccard Similarity**: Set intersection over union of tokens

#### N-gram Overlap
- **ROUGE-1 F1**: Unigram overlap (primary metric)
- **ROUGE-L**: Longest common subsequence
- **BLEU**: Precision of n-grams with smoothing

#### Semantic Similarity
- **BERT Cosine Similarity**: Sentence embedding similarity using all-MiniLM-L6-v2
- **Overlap Score**: Best-sentence overlap with expected answer

### Context Quality Metrics

- **Context Token Recall**: How well retrieved context matches expected context
- **Context ROUGE-1**: N-gram overlap of retrieved vs. expected context
- **Context BERT Similarity**: Semantic similarity of contexts

### Performance Metrics

- **Average Latency**: Mean response time per query
- **95th Percentile Latency**: Tail latency
- **Throughput**: Questions per second
- **Timeout Rate**: Percentage of queries exceeding deadline

## Benchmark Results Interpretation

### Sample Output

```
RAG EVALUATION RESULTS - BEST METRICS ONLY
============================================================

🎯 RETRIEVAL PERFORMANCE:
  Document Retrieval Accuracy: 276/300 = 92.00%
  Average Retrieval Rank: 0.8

📝 ANSWER QUALITY:
  Exact Match: 12.33%
  Substring Match: 45.67%
  Token Recall: 0.712
  ROUGE-1 F1: 0.689
  BERT Similarity: 0.823

🔍 CONTEXT QUALITY:
  Context Token Recall: 0.654
  Context ROUGE-1 F1: 0.598
  Context BERT Similarity: 0.778

⚡ PERFORMANCE:
  Average Latency: 2341.2ms
  95th Percentile Latency: 4567.8ms
  Questions per Second: 0.43
```

### What Good Results Look Like

| Metric | Poor | Acceptable | Good | Excellent |
|--------|------|------------|------|-----------|
| Retrieval Accuracy | <70% | 70-85% | 85-95% | >95% |
| ROUGE-1 F1 | <0.4 | 0.4-0.6 | 0.6-0.8 | >0.8 |
| BERT Similarity | <0.6 | 0.6-0.75 | 0.75-0.9 | >0.9 |
| Avg Latency | >5s | 3-5s | 1-3s | <1s |

## Retrieval Strategies

### 1. Naive Retrieval

Simple hybrid search without reasoning:
```python
results = collection.query.hybrid(
    query=question,
    limit=TOP_K,
    alpha=0.5  # Balance vector + keyword
)
```

**Pros**: Fast, simple, deterministic  
**Cons**: No reasoning, no context awareness

### 2. Chain-of-Thought (CoT)

Adds reasoning prompt to guide retrieval:
```python
cot_phrase = " Vamos pensar passo a passo."
results = collection.generate.hybrid(
    query=question + cot_phrase,
    grouped_task=f"{PROMPT_PREFIX}{question}{cot_phrase}",
    limit=TOP_K
)
```

**Pros**: Better complex queries, improved reasoning  
**Cons**: Slower, higher token usage

### 3. Agent-Based (ReAct)

Autonomous agent with tool use:
```python
agent = Agent(
    instructions=system_prompt,
    tools=[weaviate_query_tool, weaviate_follow_ref_tool],
    model="gpt-4"
)
result = await Runner.run(agent, question, max_turns=20)
```

**Pros**: Multi-step reasoning, adaptive retrieval  
**Cons**: Slowest, highest cost, non-deterministic

## Comparison Reports

### Generate Comparison

```powershell
python analysis/compare_agent_answers.py
```

### Output Format

`analysis/comparison_report.csv`:
```csv
question,expected,naive_k1,naive_k3,cot_k3,agent,best_method
What are approval requirements?,Requires 2 signatures,...,...,...,...,agent
```

### Metric Aggregation

```powershell
python analysis/analyze_results.py --input results/ --output summary.json
```

Generates:
- Median metrics by configuration
- Statistical significance tests
- Threshold sensitivity analysis
- Optimal parameter recommendations

## Document Generation

### Generate Synthetic Documents

```powershell
python scripts/main.py --num_docs 100 --output documentos_gerados/
```

### Generate Q&A Pairs

```powershell
python scripts/create_qa.py --input documentos_gerados/ --output datasets/qa_dataset300.json
```

### Populate Weaviate

```powershell
python scripts/populate_curated_docs.py --input curated_docs/ --collection Dataset
```

## Advanced Features

### Custom Similarity Function

Add new metrics in notebook Cell 4:
```python
def calculate_custom_similarity(answer: str, gold: str) -> float:
    # Your implementation
    return score

# Use in metrics computation
metrics['custom'] = calculate_custom_similarity(answer, gold)
```

### Batch Evaluation

Process multiple configurations:
```python
configs = [
    {"TOP_K": 1, "COT_PROMPT": False},
    {"TOP_K": 3, "COT_PROMPT": False},
    {"TOP_K": 3, "COT_PROMPT": True},
]

for config in configs:
    results = await run_eval(**config)
    save_results(results, f"results_{config}.csv")
```

### Custom Warmup

Ensure Ollama is ready:
```python
def custom_warmup(model: str, test_prompt: str) -> bool:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": test_prompt}
    )
    return response.status_code == 200

warm_up_ollama(LLM_MODEL_NAME)
```

## Troubleshooting

### Ollama Not Responding

Check model availability:
```powershell
ollama list
ollama pull qwen2.5:latest
```

Verify service:
```powershell
curl http://localhost:11434/api/tags
```

### Weaviate Connection Failed

Test connectivity in Cell 3:
```python
meta = client.get_meta()
print(meta['modules'])
```

Check Docker:
```powershell
docker ps | Select-String weaviate
```

### Out of Memory

Reduce batch size:
```python
# In document indexing
with rag.batch.dynamic() as batch:
    batch.fixed_size = 50  # Default 100
```

Disable BERT similarity:
```python
USE_BERT_SIM = False  # Saves ~2GB RAM
```

### Timeout Errors

Increase timeouts:
```python
CUSTOM_QUERY_TIMEOUT = 1800  # 30 minutes
PER_QUERY_DEADLINE = 120.0   # 2 minutes per query
```

### Chinese Output from Ollama

Enforce Portuguese:
```python
PROMPT_PREFIX = "IMPORTANTE: Responda APENAS em português (pt-BR), NUNCA em chinês: "
```

## Best Practices

1. **Always warm up Ollama** before benchmarking (Cell 1)
2. **Test connectivity** before full run (Cell 3)
3. **Use seeded randomness** for reproducibility
4. **Save intermediate results** to avoid data loss
5. **Monitor memory usage** during large batches
6. **Validate Q&A dataset** before benchmarking
7. **Compare multiple runs** to account for variance
8. **Document configuration** in results metadata

## Performance Optimization

### Speed Up Evaluation

1. **Reduce TOP_K**: Fewer retrievals = faster
2. **Disable BERT**: Skip expensive embedding computation
3. **Limit MAX_Q**: Test on subset first
4. **Use Naive approach**: Fastest baseline
5. **Increase batch size**: Better throughput

### Improve Accuracy

1. **Increase TOP_K**: More context for LLM
2. **Enable CoT**: Better reasoning
3. **Tune alpha**: Balance semantic/keyword search
4. **Better prompts**: Clearer instructions
5. **Use agent**: Multi-step retrieval

## Related Components

- **Agent Implementation**: `../../AgenticChatbot/`
- **Weaviate Setup**: `../../weaviate/weaviate_manager.py`
- **MCP Server**: `../../mcp-server-weaviate-ts/`

## References

- [ROUGE Metric](https://github.com/google-research/google-research/tree/master/rouge)
- [BERT Score](https://github.com/Tiiiger/bert_score)
- [Sentence Transformers](https://www.sbert.net/)
- [Weaviate Hybrid Search](https://weaviate.io/developers/weaviate/search/hybrid)

---

**Last Updated**: October 31, 2025
