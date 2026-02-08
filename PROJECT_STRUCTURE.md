# Project Structure

Complete file structure for the AI Resume Evaluator deployment package.

```
resume_evaluator_deployment/
│
├── README.md                          # Main documentation
├── METRICS.md                         # Comprehensive metrics guide
├── DEPLOYMENT.md                      # Deployment instructions
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Docker container definition
├── docker-compose.yml                 # Docker Compose configuration
├── .env.example                       # Environment variables template
├── .gitignore                         # Git ignore rules
│
├── src/                               # Source code
│   ├── __init__.py                   # Package initialization
│   ├── main.py                       # CLI entry point
│   ├── api.py                        # FastAPI REST API
│   ├── evaluator.py                  # Main evaluator class
│   ├── config.py                     # Configuration management
│   │
│   ├── models/                       # Model management
│   │   ├── __init__.py
│   │   ├── llm.py                   # LLM wrapper (Qwen)
│   │   └── embeddings.py            # Embedding models
│   │
│   ├── agents/                       # LangGraph agents
│   │   ├── __init__.py
│   │   ├── requirement_extractor.py # Extract JD requirements
│   │   ├── resume_evaluator.py      # Evaluate requirements
│   │   └── decision_maker.py        # Final decision logic
│   │
│   ├── rag/                          # RAG components
│   │   ├── __init__.py
│   │   ├── chunker.py               # Text chunking
│   │   ├── vectorstore.py           # FAISS management
│   │   └── retriever.py             # Evidence retrieval
│   │
│   └── utils/                        # Utilities
│       ├── __init__.py
│       ├── logger.py                 # Logging setup
│       ├── types.py                  # Type definitions
│       ├── metrics.py                # Metrics calculation
│       └── validators.py             # Input validation
│
├── scripts/                           # Utility scripts
│   ├── evaluate_metrics.py           # Run benchmarks
│   ├── generate_report.py            # Generate PDF reports
│   ├── deploy_aws.sh                 # AWS deployment
│   ├── deploy_gcp.sh                 # GCP deployment
│   └── create_test_dataset.py        # Generate test data
│
├── data/                              # Data directory
│   ├── job_descriptions/             # Sample JDs
│   │   └── sample_jd.txt
│   ├── resumes/                      # Sample resumes
│   │   └── sample_resume.txt
│   └── benchmark/                    # Test datasets
│       └── test_set.json             # Evaluation dataset
│
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── test_evaluator.py             # Evaluator tests
│   ├── test_rag.py                   # RAG tests
│   ├── test_api.py                   # API tests
│   ├── test_agents.py                # Agent tests
│   └── conftest.py                   # Pytest configuration
│
├── notebooks/                         # Jupyter notebooks
│   ├── Resume_Evaluator_Colab.ipynb  # Google Colab version
│   ├── Development.ipynb             # Development notebook
│   └── Analysis.ipynb                # Results analysis
│
├── logs/                              # Log files (gitignored)
│   └── .gitkeep
│
├── results/                           # Evaluation results (gitignored)
│   └── .gitkeep
│
├── metrics/                           # Metrics outputs
│   └── .gitkeep
│
└── docs/                              # Additional documentation
    ├── API.md                        # API reference
    ├── CONTRIBUTING.md               # Contribution guide
    ├── DEPLOYMENT.md                 # Deployment guide
    └── DEVELOPMENT.md                # Development guide
```

## File Descriptions

### Root Level

| File | Purpose |
|------|---------|
| `README.md` | Main project documentation |
| `METRICS.md` | Comprehensive metrics guide |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container image definition |
| `docker-compose.yml` | Multi-container setup |

### Source Code (`src/`)

| File | Purpose |
|------|---------|
| `main.py` | CLI interface for command-line usage |
| `api.py` | FastAPI REST API server |
| `evaluator.py` | Main evaluator orchestrating workflow |
| `config.py` | Configuration management and settings |

### Models (`src/models/`)

| File | Purpose |
|------|---------|
| `llm.py` | LLM wrapper for Qwen model |
| `embeddings.py` | Embedding model management |

### Agents (`src/agents/`)

| File | Purpose |
|------|---------|
| `requirement_extractor.py` | Extract requirements from JD |
| `resume_evaluator.py` | Evaluate resume against requirements |
| `decision_maker.py` | Generate final hiring decision |

### RAG (`src/rag/`)

| File | Purpose |
|------|---------|
| `chunker.py` | Text chunking strategies |
| `vectorstore.py` | FAISS vector store management |
| `retriever.py` | Evidence retrieval from vectors |

### Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `evaluate_metrics.py` | Run comprehensive benchmarks |
| `generate_report.py` | Generate PDF evaluation reports |
| `deploy_aws.sh` | Deploy to AWS EC2 |
| `deploy_gcp.sh` | Deploy to Google Cloud |

### Tests (`tests/`)

| File | Purpose |
|------|---------|
| `test_evaluator.py` | Test main evaluator |
| `test_rag.py` | Test RAG components |
| `test_api.py` | Test API endpoints |
| `test_agents.py` | Test individual agents |

## Directory Structure Creation

To create the complete directory structure:

```bash
# Clone repository
git clone https://github.com/yourusername/ai-resume-evaluator.git
cd ai-resume-evaluator

# Create all directories
mkdir -p src/{models,agents,rag,utils}
mkdir -p scripts
mkdir -p data/{job_descriptions,resumes,benchmark}
mkdir -p tests
mkdir -p notebooks
mkdir -p logs results metrics docs

# Create __init__.py files
touch src/__init__.py
touch src/{models,agents,rag,utils}/__init__.py
touch tests/__init__.py

# Create .gitkeep files for empty directories
touch {logs,results,metrics}/.gitkeep
```

## Recommended IDE Setup

### VS Code
```json
{
  "editor.formatOnSave": true,
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true
}
```

### PyCharm
- Mark `src/` as Sources Root
- Mark `tests/` as Test Sources Root
- Enable pytest as test runner
- Configure Black as code formatter

## Development Workflow

1. **Setup Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Development**
   ```bash
   # Make changes to src/
   python src/main.py --jd data/job_descriptions/sample_jd.txt \
                      --resume data/resumes/sample_resume.txt
   ```

3. **Testing**
   ```bash
   pytest tests/
   ```

4. **Run API**
   ```bash
   python src/api.py
   ```

5. **Evaluate Metrics**
   ```bash
   python scripts/evaluate_metrics.py \
     --dataset data/benchmark/test_set.json \
     --output metrics/results.json
   ```

## Deployment Workflow

1. **Local Testing**
   ```bash
   python src/api.py
   ```

2. **Docker Build**
   ```bash
   docker build -t resume-evaluator .
   ```

3. **Docker Run**
   ```bash
   docker-compose up -d
   ```

4. **Cloud Deployment**
   ```bash
   ./scripts/deploy_aws.sh  # or deploy_gcp.sh
   ```

---

**Last Updated:** 2024-02-08
