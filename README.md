# Thesis Project: RAG Systems with Agentic Capabilities and Weaviate Integration

**Author:** Francisco Azeredo  
**Institution:** Técnico Lisboa  
**Year:** 2025

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Repository Structure](#-repository-structure)
3. [Quick Start](#-quick-start)
4. [Installation Guide](#-installation-guide)
5. [Key Components](#-key-components)
6. [File Organization](#-file-organization)
7. [Usage Examples](#-usage-examples)
8. [Benchmarking Results](#-benchmarking-results)
9. [Development Guide](#-development-guide)
10. [Troubleshooting](#-troubleshooting)
11. [Documentation](#-documentation)

---

## 📚 Project Overview

This repository contains the complete implementation for a Master's thesis exploring Retrieval-Augmented Generation (RAG) systems with agentic capabilities, integrated with Weaviate vector database.

### Research Goals

1. **Implement and compare multiple RAG approaches**: Naive retrieval, Chain-of-Thought prompting, and agent-based retrieval
2. **Integrate Weaviate vector database**: Leverage semantic search and graph-based relationships
3. **Develop MCP (Model Context Protocol) servers**: Enable structured communication between agents and databases
4. **Benchmark performance**: Comprehensive evaluation of different retrieval strategies
5. **Create agentic systems**: Implement autonomous agents capable of multi-step reasoning

### Technologies Used

- **LLMs**: OpenAI GPT-4/5, Ollama (Qwen2.5)
- **Vector DB**: Weaviate (v4+)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Frameworks**: OpenAI Agents SDK, LangChain components
- **Languages**: Python, TypeScript, Go
- **Evaluation**: ROUGE, BLEU, BERT Similarity, Custom metrics

---

## 📁 Repository Structure

```
Thesis_Code/
│
├── AgenticChatbot/                    # Agentic RAG system with OpenAI
│   ├── legal_document_agent_openai.py # Main agent implementation
│   ├── scripts/                       # Benchmarks and utilities
│   ├── datasets/                      # Q&A datasets
│   ├── results/                       # Benchmark results
│   └── sessions/                      # Session storage
│
├── DocumentGenerator/                 # Benchmark framework
│   ├── scripts/                       # Core execution scripts
│   ├── notebooks/                     # Jupyter benchmark notebooks
│   ├── datasets/                      # Q&A datasets
│   ├── analysis/                      # Analysis and visualization
│   ├── results/                       # Benchmark results
│   └── curated_docs/                  # Source documents
│
├── MiniRAG/                           # Lightweight RAG framework
│   ├── main.py                        # MiniRAG entry point
│   ├── minirag/                       # Core RAG implementation
│   └── dataset/                       # Test datasets
│
├── weaviate/                          # Weaviate database management
│   ├── weaviate_manager.py            # Main database manager
│   ├── managers/                      # Alternative implementations
│   ├── notebooks/                     # Interactive notebooks
│   ├── app/                           # Application scripts
│   └── tests/                         # Connection and benchmark tests
│
├── mcp-server-weaviate-ts/            # TypeScript MCP server (recommended)
│   ├── src/                           # Server implementation
│   └── test/                          # Integration tests
│
├── mcp-server-weaviate/               # Go MCP server (alternative)
│   └── *.go                           # Go source files
│
└── Nougat/                            # PDF processing utilities
```

> **Note**: The codebase was reorganized on October 31, 2025 for better clarity. Files are now organized into logical folders: `scripts/`, `datasets/`, `results/`, `analysis/`, etc.

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** with virtual environment support
- **Node.js 18+** (for TypeScript MCP server)
- **Docker & Docker Compose** (for Weaviate)
- **Ollama** (for local LLM inference) or OpenAI API key

### Quick Setup

```powershell
# 1. Start Weaviate with Docker
cd weaviate
docker-compose up -d

# 2. Start Ollama (in a separate terminal)
ollama serve
ollama pull qwen2.5:latest

# 3. Setup Python environment for benchmarking
cd ..\DocumentGenerator
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 4. Start MCP server (TypeScript version)
cd ..\mcp-server-weaviate-ts
npm install
npm run build
npm start
```

### Verification

```powershell
# Test Weaviate
cd weaviate\tests
python checkConnectivity.py

# Test Ollama
python ollamacheck.py
```

Expected: ✅ All checks pass

---

## 🔧 Installation Guide

### System Requirements

**Minimum**:
- **OS**: Windows 10/11, macOS 12+, or Linux
- **RAM**: 16 GB (32 GB recommended)
- **Disk**: 50 GB free space
- **CPU**: 4 cores (8 cores recommended)

### 1. Install Docker Desktop

**Windows**:
1. Download from https://www.docker.com/products/docker-desktop
2. Run installer and restart computer
3. Verify: `docker --version`

**macOS**:
```bash
brew install --cask docker
```

**Linux**:
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

### 2. Install Python 3.10+

**Windows**:
1. Download from https://www.python.org/downloads/
2. ✅ Check "Add Python to PATH"
3. Verify: `python --version`

**macOS**:
```bash
brew install python@3.10
```

**Linux**:
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
```

### 3. Install Node.js 18+

**Windows**: Download from https://nodejs.org/

**macOS**:
```bash
brew install node@18
```

**Linux**:
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### 4. Install Ollama

**Windows**: Download from https://ollama.com/download/windows

**macOS**:
```bash
brew install ollama
```

**Linux**:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 5. Setup Weaviate

```powershell
cd weaviate

# Review docker-compose.yml configuration
# Start services
docker-compose up -d

# Check status
docker-compose ps

# Test endpoint
curl http://localhost:8080/v1/meta
```

### 6. Setup Ollama Models

```powershell
# Start Ollama service
ollama serve

# Pull required models (in another terminal)
ollama pull qwen2.5:latest    # Primary model (3.7GB)
ollama pull qwen2:latest      # Alternative (1.5GB)

# Verify
ollama list
```

### 7. Setup Python Environments

#### For DocumentGenerator

```powershell
cd DocumentGenerator

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install weaviate-client sentence-transformers torch
pip install nltk rouge-score python-docx psutil tqdm
pip install jupyter notebook ipykernel

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Register Jupyter kernel
python -m ipykernel install --user --name=venv --display-name "Thesis Benchmark"
```

#### For AgenticChatbot

```powershell
cd ..\AgenticChatbot
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install agents openai python-dotenv requests
```

#### For MiniRAG

```powershell
cd ..\MiniRAG
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e .
```

### 8. Setup MCP Server (TypeScript)

```powershell
cd ..\mcp-server-weaviate-ts

# Install dependencies
npm install

# Create .env file
echo "WEAVIATE_HOST=localhost" > .env
echo "WEAVIATE_PORT=8080" >> .env
echo "WEAVIATE_SCHEME=http" >> .env
echo "MCP_TRANSPORT=http" >> .env
echo "MCP_HTTP_HOST=127.0.0.1" >> .env
echo "MCP_HTTP_PORT=3000" >> .env
echo "LOG_LEVEL=info" >> .env
echo "READ_ONLY=false" >> .env

# Build
npm run build

# Start server
npm start
```

Expected output:
```
🚀 Starting Weaviate MCP Server v0.1.0
✅ HTTP server listening on 127.0.0.1:3000
```

### 9. Configure Environment Variables

#### AgenticChatbot/.env
```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4
MCP_TRANSPORT=http
MCP_HTTP_URL=http://127.0.0.1:3000/mcp
```

#### DocumentGenerator/.env
```env
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4
```

---

## 🔑 Key Components

### 1. AgenticChatbot - Intelligent Document Agent

An autonomous agent built with OpenAI's Agents SDK that:
- Uses ReAct (Reasoning + Acting) pattern
- Maintains conversation context across turns
- Leverages MCP for structured database access
- Supports retry logic and error handling

**Key Files**:
- `legal_document_agent_openai.py` - Main agent implementation
- `scripts/benchmark_openAI_1.py` - Single-question benchmark
- `scripts/benchmark_openAI_2.py` - Batch benchmark
- `datasets/merged_qa_dataset.json` - 358 Q&A pairs
- `sessions/` - Persisted conversation sessions

**Quick Start**:
```powershell
cd AgenticChatbot
python scripts/demo.py
```

### 2. DocumentGenerator - Comprehensive Evaluation

A rigorous benchmarking system evaluating:
- **Retrieval Accuracy**: Document retrieval precision
- **Answer Quality**: ROUGE, BLEU, BERT similarity, token recall
- **Latency**: Response time metrics
- **Approaches**: Naive, Chain-of-Thought, Agentic

**Key Files**:
- `notebooks/benchmark_weaviate.ipynb` - Main benchmarking notebook
- `notebooks/benchmark_weaviate_openAI.ipynb` - OpenAI-specific benchmarks
- `datasets/merged_qa_dataset.json` - 358 Q&A pairs
- `results/` - Detailed metrics and comparisons
- `good_data/` - Best performing configurations

**Quick Start**:
```powershell
cd DocumentGenerator
.\venv\Scripts\Activate.ps1
jupyter notebook notebooks/benchmark_weaviate.ipynb
```

### 3. MiniRAG - Lightweight RAG Framework

Integration with MiniRAG for:
- Knowledge graph construction
- Entity and relationship extraction
- Text summarization (LexRank, BART)
- Hybrid retrieval strategies

**Key Files**:
- `main.py` - Entry point
- `minirag/minirag.py` - Core RAG implementation
- `dataset/LiHua-World/` - LiHua benchmark dataset

**Quick Start**:
```powershell
cd MiniRAG
.\venv\Scripts\Activate.ps1
python main.py
```

### 4. Weaviate Integration - Vector Database

Sophisticated database schema with:
- **Collections**: Fluxo, Etapa, Entidade, Pasta, Ficheiro
- **Cross-references**: Rich relationship modeling
- **Hybrid search**: Combines semantic + keyword search
- **Generative AI**: Ollama integration for answer generation

**Key Files**:
- `weaviate_manager.py` - Complete database manager
- `app/setup_collections.py` - Schema definition
- `tests/checkConnectivity.py` - Connection verification
- `docker-compose.yml` - Docker services

**Quick Start**:
```powershell
cd weaviate
python weaviate_manager.py
```

### 5. MCP Servers - Protocol Implementation

Two implementations of Model Context Protocol:
- **TypeScript** (recommended): Full-featured, HTTP + stdio
- **Go**: Alternative implementation, stdio only

Both provide:
- Query tools for semantic search
- Reference navigation across collections
- Read-only and read-write modes

**Quick Start (TypeScript)**:
```powershell
cd mcp-server-weaviate-ts
npm start
```

---

## 📂 File Organization

The codebase follows a **function-based organization** strategy:

### Organization Principles

1. **scripts/** - Executable files (.py scripts, notebooks)
2. **datasets/** - Data files (JSON, CSV with Q&A pairs)
3. **results/** - Benchmark output (CSV results)
4. **analysis/** - Analysis and visualization scripts
5. **logs/** - Output and log files
6. **legacy/** - Deprecated/old versions
7. **managers/** - Core implementation variants

### Quick File Lookup

| What you need | Where to find it |
|---------------|------------------|
| **AgenticChatbot** | |
| Main agent code | `legal_document_agent_openai.py` (root) |
| Run benchmarks | `scripts/benchmark_openAI_*.py` |
| Q&A datasets | `datasets/*.json` |
| Benchmark results | `results/*.csv` |
| **DocumentGenerator** | |
| Main benchmark notebook | `notebooks/benchmark_weaviate.ipynb` |
| Generate documents | `scripts/main.py` |
| Q&A datasets | `datasets/*.json` |
| Analysis scripts | `analysis/*.py` |
| Best configurations | `good_data/*.csv` |
| **weaviate** | |
| Main manager | `weaviate_manager.py` (root) |
| Alternative managers | `managers/*.py` |
| Test connection | `tests/checkConnectivity.py` |
| Interactive notebooks | `notebooks/*.ipynb` |

### Path Updates After Reorganization

If you have existing scripts, update paths:

```python
# OLD
"benchmark_openAI_1.py"
"qa_dataset300.json"
"benchmark_results_openai.csv"

# NEW
"scripts/benchmark_openAI_1.py"
"datasets/qa_dataset300.json"
"results/benchmark_results_openai.csv"
```

**Note**: Main module imports still work unchanged:
```python
from weaviate_manager import WeaviateDataManager  # ✅ Still works
from legal_document_agent_openai import openai_agent  # ✅ Still works
```

---

## 💻 Usage Examples

### Running a Benchmark

```powershell
# Full benchmark (358 questions)
cd DocumentGenerator
.\venv\Scripts\Activate.ps1
jupyter notebook notebooks/benchmark_weaviate.ipynb
# Follow notebook instructions

# Agent benchmark
cd ..\AgenticChatbot
.\venv\Scripts\Activate.ps1
python scripts/benchmark_openAI_2.py
```

### Querying with the Agent

```python
from legal_document_agent_openai import openai_agent

answer = await openai_agent("What are the requirements for contract approval?")
print(answer.final_output)
```

### Using Weaviate Manager

```python
from weaviate_manager import WeaviateDataManager

manager = WeaviateDataManager()
manager.setup_collections(clean_start=True)

results = manager.entity_flux_semantic_search(
    None, None, 
    "contract approval process", 
    alpha=0.3, 
    limit=5
)

for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Entity: {result['entity_name']}")
    print(f"Context: {result['context'][:200]}...")
    print()

manager.close()
```

### Testing MCP Server

```powershell
# Start server
cd mcp-server-weaviate-ts
npm start

# In another terminal, test connection
curl http://127.0.0.1:3000/health

# Or run integration test
node test/test-server.js
```

---

## 📈 Benchmarking Results

Evaluation of multiple configurations across 358 Q&A pairs:

| Configuration | Retrieval Accuracy | Avg Latency | ROUGE-1 F1 | BERT Similarity |
|---------------|-------------------|-------------|------------|-----------------|
| Naive (k=1)   | 85%              | 1.2s        | 0.65       | 0.78            |
| Naive (k=3)   | 92%              | 1.8s        | 0.71       | 0.82            |
| CoT (k=3)     | 94%              | 2.5s        | 0.74       | 0.85            |
| Agent-based   | 96%              | 3.2s        | 0.78       | 0.87            |

*See `DocumentGenerator/good_data/` for detailed configuration reports.*

### Key Findings

1. **Agent-based approach achieves highest accuracy** (96%) but with increased latency
2. **Chain-of-Thought improves accuracy** by 2% over naive retrieval
3. **Top-K=3 significantly better** than K=1 for naive approaches
4. **BERT similarity correlates strongly** with human evaluation

---

## 🛠️ Development Guide

### Project Structure Best Practices

#### Adding New Scripts
Place in appropriate `scripts/` folder:
```powershell
# AgenticChatbot
cd AgenticChatbot\scripts
# Add your_new_script.py

# DocumentGenerator
cd DocumentGenerator\scripts
# Add your_new_script.py
```

#### Adding New Results
Save to `results/` folder:
```python
import pandas as pd

results_df = pd.DataFrame(results)
results_df.to_csv("results/my_benchmark_results.csv", index=False)
```

#### Adding New Datasets
Place in `datasets/` folder:
```python
import json

with open("datasets/my_qa_dataset.json", "w", encoding="utf-8") as f:
    json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
```

### Running Tests

```powershell
# Test Weaviate connectivity
cd weaviate\tests
python checkConnectivity.py

# Test Ollama
python ollamacheck.py

# Test MCP server
cd ..\..\mcp-server-weaviate-ts
npm test

# Run agent benchmarks
cd ..\AgenticChatbot
python scripts/benchmark_openAI_1.py
```

### Git Workflow

```powershell
# Check status (will show organized structure)
git status

# Add changes
git add .

# Commit with descriptive message
git commit -m "Add new benchmark configuration"

# Push to remote
git push origin main
```

---

## 🔍 Troubleshooting

### Docker Issues

**Problem**: Docker daemon not running
```powershell
# Start Docker Desktop
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
```

**Problem**: Port 8080 already in use
```powershell
# Find process
netstat -ano | findstr :8080

# Kill process (replace <PID>)
taskkill /PID <PID> /F
```

### Python Issues

**Problem**: ModuleNotFoundError
```powershell
# Verify virtual environment is activated (look for (venv) in prompt)
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

**Problem**: NLTK data not found
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Ollama Issues

**Problem**: Model not found
```powershell
# Pull model
ollama pull qwen2.5:latest

# Verify
ollama list
```

**Problem**: Connection refused
```powershell
# Start Ollama
ollama serve

# Check port
netstat -ano | findstr :11434
```

### Weaviate Issues

**Problem**: Connection timeout
```powershell
# Check container status
cd weaviate
docker-compose ps

# Restart services
docker-compose restart

# View logs
docker-compose logs weaviate
```

**Problem**: Vectorization slow
Edit `docker-compose.yml` to increase memory:
```yaml
services:
  weaviate:
    deploy:
      resources:
        limits:
          memory: 4G
```

### MCP Server Issues

**Problem**: Port 3000 in use
```powershell
# Find process
netstat -ano | findstr :3000

# Change port in .env
echo "MCP_HTTP_PORT=3001" >> .env
```

**Problem**: Build fails
```powershell
# Clean and rebuild
npm run clean
npm install
npm run build
```

---

## 📖 Documentation

### Component Documentation

Each major component has detailed documentation:

- **AgenticChatbot/README.md** - Agent implementation details
- **DocumentGenerator/README.md** - Benchmarking methodology
- **MiniRAG/README.md** - RAG framework integration
- **weaviate/README.md** - Database schema and queries
- **mcp-server-weaviate-ts/README.md** - MCP server usage

### Research Contributions

1. **Comprehensive RAG Comparison**: Direct comparison of naive, CoT, and agentic approaches
2. **MCP Integration**: Novel use of Model Context Protocol for agent-database communication
3. **Portuguese Language Focus**: Specialized evaluation for pt-BR document retrieval
4. **Hybrid Retrieval**: Combines semantic search with graph-based navigation
5. **Reproducible Benchmarks**: Complete framework for RAG system evaluation

---

## 🤝 Contributing

This is a thesis project. For questions or collaboration:
- Open an issue on the repository
- Contact: francisco.azeredo@tecnico.ulisboa.pt

---

## 📄 License

This project is part of academic research. Please cite if you use any components:

```bibtex
@mastersthesis{azeredo2025rag,
  author  = {Francisco Azeredo},
  title   = {RAG Systems with Agentic Capabilities and Weaviate Integration},
  school  = {Instituto Superior Técnico, Universidade de Lisboa},
  year    = {2025}
}
```

---

## 🙏 Acknowledgments

- Thesis advisor and committee
- OpenAI for the Agents SDK
- Weaviate community
- MiniRAG framework authors

---

## 📧 Contact

**Francisco Azeredo**  
Instituto Superior Técnico, Universidade de Lisboa  
francisco.azeredo@tecnico.ulisboa.pt

---

**Last Updated:** October 31, 2025  
**Version:** 2.0.0 (Consolidated)
