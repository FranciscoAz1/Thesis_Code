# Weaviate Database Manager

## Overview

This module provides a comprehensive Python interface for managing a Weaviate vector database with a complex relational schema designed for legal document workflows. It includes collection setup, data insertion, advanced querying with cross-references, and semantic search capabilities.

## Key Features

- **Rich Schema**: 5 interconnected collections (Fluxo, Etapa, Entidade, Pasta, Ficheiro)
- **Cross-References**: Bidirectional relationships between collections
- **Hybrid Search**: Combines vector similarity + keyword matching
- **Semantic Navigation**: Follow references across collections
- **Generative AI**: Integrated Ollama for answer generation
- **Batch Insertion**: Efficient bulk data loading
- **Dynamic Schema Introspection**: Query schema at runtime

## Database Schema

```
Entidade (Company/Organization)
  ↓ hasPastas
Pasta (Folder)
  ↓ hasFicheiros, hasFluxos
Ficheiro (Document)
  ↓ hasEtapas
  ├ metadados (TEXT property)
Etapa (Stage)
  ↓ belongsToFluxo
Fluxo (Workflow)
  ↓ hasEtapas, belongsToFicheiros, belongsToPastas
```

### Collection Details

#### Entidade
- **Properties**: `name` (TEXT)
- **References**: 
  - `hasPastas` → Pasta
  - `hasFicheiros` → Ficheiro
- **Description**: Represents a company or organization that owns folders

#### Pasta
- **Properties**: `name` (TEXT)
- **References**:
  - `hasFicheiros` → Ficheiro
  - `hasEntidades` → Entidade
  - `hasFluxos` → Fluxo
- **Description**: A folder belonging to an entity, containing documents

#### Ficheiro
- **Properties**: `name` (TEXT), `metadados` (TEXT)
- **References**:
  - `hasEtapas` → Etapa
  - `hasPastas` → Pasta
  - `hasEntidades` → Entidade
- **Description**: A document with metadata, part of workflows

#### Etapa
- **Properties**: `name` (TEXT)
- **References**:
  - `belongsToFluxo` → Fluxo
  - `hasFicheiros` → Ficheiro
- **Description**: A stage within a workflow

#### Fluxo
- **Properties**: `name` (TEXT)
- **References**:
  - `hasEtapas` → Etapa
  - `belongsToFicheiros` → Ficheiro
  - `belongsToPastas` → Pasta
- **Description**: A workflow containing multiple stages

## Directory Structure

```
weaviate/
├── weaviate_manager.py              # ⭐ Main database manager (root for easy import)
├── README.md                        # This file
├── .env                            # Environment configuration
├── docker-compose.yml              # Weaviate services configuration
├── Weaviate.md                     # Additional documentation
├── tsweaviate.ts                   # TypeScript client example
├── Modelfile                       # Ollama model config
├── output.txt                      # Debug output
│
├── managers/                       # Additional manager implementations
│   ├── weaviate_manager_benchmark.py  # Benchmarking utilities
│   └── weaviate_manager_paralyzed.py  # Alternative implementation
│
├── notebooks/                      # Interactive notebooks
│   ├── weaviate.ipynb              # Interactive exploration
│   ├── cross_reference.ipynb       # Reference navigation examples
│   └── weavaddobj.ipynb            # Object insertion examples
│
├── app/                           # Application scripts
│   ├── main_script.py             # Complete workflow example
│   ├── setup_collections.py       # Initialize schema
│   ├── data_insertion.py          # Bulk data loading
│   ├── data_queries.py            # Example queries
│   └── weaviate_manager.py        # Manager reference
│
├── data_structure/                # Data models
│   ├── main.py                    # Model examples
│   └── models/                    # Python dataclasses
│       ├── documento.py           # Document model
│       ├── entidade.py            # Entity model
│       ├── etapa.py               # Stage model
│       ├── fluxo.py               # Workflow model
│       └── pasta.py               # Folder model
│
├── tests/                         # Testing and verification
│   ├── checkConnectivity.py      # ⭐ Connection verification
│   ├── ollamacheck.py             # Ollama integration test
│   ├── word_tokens.py             # Tokenization test
│   ├── benchmark_configs.py       # Performance testing
│   └── benchmark_results*.csv     # Benchmark data
│
├── scripts/                       # Utility scripts
│   ├── plot_latency_vs_files.py   # Performance visualization
│   └── outputs/                   # Script outputs
│
├── backups_old/                   # Old configuration backups
│   └── .env copy1                 # Backup environment file
│
└── .vscode/                       # VS Code configuration
    ├── launch.json                # Debug configuration
    └── settings.json              # Editor settings
```

## Files

### Core Implementation

- **`weaviate_manager.py`** (root) - Main database manager class (⭐ USE THIS)
  - Collection setup and teardown
  - Data insertion with relationships
  - Advanced querying
  - Semantic search
  - Schema introspection

- **`managers/weaviate_manager_benchmark.py`** - Benchmarking utilities
- **`managers/weaviate_manager_paralyzed.py`** - Alternative implementation

### Application Scripts

- **`app/setup_collections.py`** - Initialize database schema
- **`app/data_insertion.py`** - Bulk data loading
- **`app/data_queries.py`** - Example queries
- **`app/main_script.py`** - Complete workflow example

### Data Models

- **`data_structure/models/`** - Python dataclasses
  - `documento.py`
  - `entidade.py`
  - `etapa.py`
  - `fluxo.py`
  - `pasta.py`

### Testing

- **`tests/checkConnectivity.py`** - Connection verification
- **`tests/benchmark_configs.py`** - Performance testing
- **`tests/ollamacheck.py`** - Ollama integration test
- **`tests/benchmark_results*.csv`** - Performance data

### Notebooks

- **`weaviate.ipynb`** - Interactive exploration
- **`cross_reference.ipynb`** - Reference navigation examples
- **`weavaddobj.ipynb`** - Object insertion examples

### Configuration

- **`docker-compose.yml`** - Weaviate + services setup
- **`.env`** - Environment variables

## Setup

### Prerequisites

1. **Docker & Docker Compose**
2. **Python 3.10+**
3. **Ollama** (optional, for generation)

### Installation

```powershell
# Start Weaviate
docker-compose up -d

# Install Python dependencies
pip install weaviate-client sentence-transformers
```

### Docker Configuration

The `docker-compose.yml` includes:
- **Weaviate**: Vector database (port 8080)
- **t2v-transformers**: Embedding service (all-MiniLM-L6-v2)
- **Ollama** (optional): LLM generation (port 11434)

## Usage

### Basic Setup

```python
from weaviate_manager import WeaviateDataManager

# Initialize manager
manager = WeaviateDataManager(connect_to_local=True)

try:
    # Setup schema
    manager.setup_collections(clean_start=True)
    
    # Insert sample data
    data = manager.insert_sample_data(
        num_entidades=2,
        num_pastas_per_entidade=5,
        num_ficheiros_per_pasta=8,
        num_fluxos=5,
        num_etapas_per_fluxo=10
    )
    
    print(f"Inserted {len(data['ficheiros'])} documents")
    
finally:
    manager.close()
```

### Creating Entities with Relationships

```python
# Create an entity
entidade = manager.add_entidade("Acme Corporation")

# Create a folder for the entity
pasta = manager.add_pasta(
    "Contracts 2024",
    entidade_obj=entidade
)

# Create a document in the folder
ficheiro = manager.add_ficheiro(
    ficheiro_name="Contract_ABC123.pdf",
    metadados_name="Contract type: Purchase, Date: 2024-01-15",
    pasta_obj=pasta,
    entidade_obj=entidade
)

# Create a workflow
fluxo = manager.add_fluxo(
    "Contract Approval Process",
    pasta_obj=pasta,
    ficheiro_obj=ficheiro
)

# Add workflow stages
etapa1 = manager.add_etapa(
    "Legal Review",
    fluxo_obj=fluxo,
    ficheiro_obj=ficheiro
)

etapa2 = manager.add_etapa(
    "Executive Approval",
    fluxo_obj=fluxo,
    ficheiro_obj=ficheiro
)
```

### Querying with References

#### Query Workflows and Their Stages

```python
results = manager.query_fluxo_etapas(limit=10)

for fluxo in results:
    print(f"Workflow: {fluxo['fluxo_name']}")
    for etapa in fluxo['etapas']:
        print(f"  - Stage: {etapa}")
```

#### Query Entity Hierarchy

```python
results = manager.query_entidade_hierarchy(limit=5)

for ent in results:
    print(f"Entity: {ent['entidade_name']}")
    for pasta in ent['pastas']:
        print(f"  Folder: {pasta['pasta_name']}")
        for fich in pasta['ficheiros']:
            print(f"    Document: {fich['ficheiro_name']}")
            print(f"    Metadata: {fich['metadados']}")
```

#### Deep Hierarchy Navigation

```python
results = manager.query_entidade_deep_hierarchy(limit=10)

# Traverses: Entidade → Pasta → Ficheiro → Etapa → Fluxo
```

### Semantic Search

#### Global Search Across All Collections

```python
results = manager.global_semantic_search(
    query_text="contract approval process",
    alpha=0.2,  # 0.0=pure vector, 1.0=pure keyword, 0.5=balanced
    limit_per_collection=5
)

for result in results:
    print(f"[{result['class']}] {result['name']}")
    print(f"  Score: {result['score']:.4f}")
    print(f"  Metadata: {result['metadados']}")
```

#### Search with Explainability

```python
results = manager.entity_flux_semantic_search(
    entidade_name="Acme Corporation",  # Filter by entity
    fluxo_name="Approval Process",     # Filter by workflow
    query_text="legal review stage",
    alpha=0.3,
    limit=10
)

for result in results:
    print(f"[{result['class']}] {result['name']} (score={result['score']})")
    print(f"  Workflows: {', '.join(result['fluxos'])}")
    print(f"  Entities: {', '.join(result['entidades'])}")
```

## Advanced Features

### Dynamic Reference Planning

Build nested query plans automatically from schema:

```python
# Plan references 2 levels deep for all collections
plan = manager.build_reference_plan(
    classes=["Pasta", "Fluxo", "Etapa", "Ficheiro", "Entidade"],
    depth=2
)

# Use in query
results = manager.pasta.query.fetch_objects(
    return_references=plan["Pasta"],
    limit=10
)
```

### Schema Introspection

```python
# Get all reference properties for a collection
refs = manager.get_class_references("Ficheiro")

for ref in refs:
    print(f"{ref['name']} → {ref['target_collection']}")
```

Output:
```
hasEtapas → Etapa
hasPastas → Pasta
hasEntidades → Entidade
```

### Dense Connection Data

Create a maximally connected dataset:

```python
data = manager.insert_conection_data(
    num_entidades=2,
    num_pastas_per_entidade=5,
    num_ficheiros_per_pasta=8,
    num_fluxos=5,
    num_etapas_per_fluxo=10
)

# This creates bidirectional references across all relationships
```

## Benchmarking

### Run Performance Tests

```python
from weaviate_manager import benchmark_sample_configs

configs = [
    {
        "num_entidades": 2,
        "num_pastas_per_entidade": 5,
        "num_ficheiros_per_pasta": 8,
        "num_fluxos": 5,
        "num_etapas_per_fluxo": 10,
    },
    {
        "num_entidades": 5,
        "num_pastas_per_entidade": 10,
        "num_ficheiros_per_pasta": 20,
        "num_fluxos": 10,
        "num_etapas_per_fluxo": 15,
    },
]

results = benchmark_sample_configs(
    configs,
    limit_fluxo_etapas=3,
    limit_entidade_hierarchy=2,
    global_query="contract approval",
    clean_start_each=True
)

# Results contain timing for:
# - setup_collections
# - insert_sample_data
# - query_fluxo_etapas
# - query_entidade_deep_hierarchy
# - global_semantic_search
```

### Performance Metrics

Typical performance (local Docker, 80 documents):

| Operation | Time |
|-----------|------|
| Setup Collections | ~0.5s |
| Insert Sample Data | ~2-5s |
| Query Fluxo Etapas | ~0.1s |
| Deep Hierarchy Query | ~0.3s |
| Semantic Search | ~0.5s |

## Configuration

### Vectorization

```python
Configure.NamedVectors.text2vec_transformers(
    name="text_vector",
    source_properties=["text"],
    pooling_strategy="masked_mean",
)
```

- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Dimensions**: 384
- **Source**: Docker container (t2v-transformers)

### Generative AI

```python
Configure.Generative.ollama(
    api_endpoint="http://host.docker.internal:11434",
    model="qwen2.5:latest"
)
```

## Data Insertion Patterns

### Single Object

```python
# Simple insertion
ficheiro_obj = manager.ficheiro.data.insert({
    "name": "document.pdf",
    "metadados": "Contract, 2024-01-15"
})
```

### With References

```python
# Insert and link
ficheiro_obj = manager.add_ficheiro(
    ficheiro_name="document.pdf",
    metadados_name="Contract metadata",
    pasta_obj=pasta_obj,
    entidade_obj=entidade_obj
)
```

### Batch Insertion

```python
with manager.ficheiro.batch.dynamic() as batch:
    for doc in documents:
        batch.add_object({
            "name": doc['name'],
            "metadados": doc['metadata']
        })
```

## Query Patterns

### Fetch with References

```python
from weaviate.classes.query import QueryReference, MetadataQuery

results = manager.entidade.query.fetch_objects(
    return_properties=["name"],
    return_references=QueryReference(
        link_on="hasPastas",
        return_properties=["name"],
        return_references=QueryReference(
            link_on="hasFicheiros",
            return_properties=["name", "metadados"]
        )
    ),
    limit=10
)

for obj in results.objects:
    print(obj.properties['name'])
    if hasattr(obj, 'references'):
        for pasta in obj.references['hasPastas'].objects:
            print(f"  {pasta.properties['name']}")
```

### Hybrid Search

```python
results = manager.ficheiro.query.hybrid(
    query="contract approval",
    alpha=0.5,  # Balance semantic + keyword
    return_properties=["name", "metadados"],
    return_metadata=MetadataQuery(score=True, distance=True),
    limit=10
)

for obj in results.objects:
    print(f"{obj.properties['name']} (score={obj.metadata.score})")
```

### Generative Search

```python
results = manager.ficheiro.generate.hybrid(
    query="What is the approval process?",
    grouped_task="Summarize the approval process based on these documents:",
    limit=5
)

# Access generated answer
answer = results.generative.text
print(answer)

# Access source documents
for obj in results.objects:
    print(f"Source: {obj.properties['name']}")
```

## Troubleshooting

### Connection Failed

Check Weaviate status:
```powershell
docker ps | Select-String weaviate
curl http://localhost:8080/v1/meta
```

Restart services:
```powershell
docker-compose down
docker-compose up -d
```

### Slow Queries

1. **Reduce limit**: Query fewer objects
2. **Limit reference depth**: Avoid deep nesting
3. **Use filters**: Narrow search space
4. **Check indexes**: Ensure vectorization complete

### Reference Errors

Verify reference exists:
```python
config = manager.ficheiro.config.get()
print(config.references)
```

Add missing reference:
```python
from weaviate.classes.config import ReferenceProperty

manager.ficheiro.config.add_reference(
    ReferenceProperty(
        name="hasEtapas",
        target_collection="Etapa"
    )
)
```

### Memory Issues

For large datasets:
```python
# Use dynamic batching with smaller size
with collection.batch.dynamic() as batch:
    batch.batch_size = 50  # Default 100
    batch.num_workers = 2  # Default 1
    for item in data:
        batch.add_object(item)
```

## Best Practices

1. **Always close connections**: Use try-finally or context managers
2. **Use batch insertion**: Much faster than individual inserts
3. **Plan references ahead**: Schema changes require collection recreation
4. **Index before querying**: Wait for vectorization to complete
5. **Test with small data**: Validate schema before large imports
6. **Monitor Docker resources**: Weaviate needs sufficient memory
7. **Use alpha tuning**: Experiment with hybrid search balance
8. **Leverage schema introspection**: Build dynamic queries

## Integration with Other Components

### With MCP Servers

The MCP servers (`mcp-server-weaviate-ts`) use this schema:

```typescript
// Query with references
const result = await weaviateManager.entity_flux_semantic_search(
    "Acme Corp",
    "Approval Process",
    "contract terms",
    0.3,
    5
);
```

### With Benchmarking

```python
# Populate Weaviate for benchmarks
from weaviate_manager import WeaviateDataManager

manager = WeaviateDataManager()
manager.setup_collections(clean_start=True)

# Load documents from benchmark dataset
with manager.ficheiro.batch.dynamic() as batch:
    for doc in benchmark_docs:
        batch.add_object({
            "name": doc['filename'],
            "metadados": doc['metadata']
        })
```

## Related Components

- **MCP Server**: `../mcp-server-weaviate-ts/` - Protocol adapter
- **Benchmarking**: `../gerador_documentos_gpt_azure (1)/gerador_documentos_gpt_azure/`
- **Agent**: `../AgenticChatbot/` - Uses this database

## References

- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [Python Client Reference](https://weaviate.io/developers/weaviate/client-libraries/python)
- [Hybrid Search Guide](https://weaviate.io/developers/weaviate/search/hybrid)
- [Cross-References](https://weaviate.io/developers/weaviate/manage-data/cross-references)

---

**Last Updated**: October 31, 2025
