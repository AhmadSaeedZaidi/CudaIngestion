# CUDA Kernel Ingestion Pipeline

Automated data engineering pipeline for scraping, filtering, annotating, and storing high-quality CUDA C++ kernels from GitHub. The resulting dataset is used for Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) of an open-weights LLM.

## Features

- **GitHub Scraper**: Searches for `.cu` and `.cuh` files across diverse computational domains
- **Domain Balancing**: Dynamically cycles through search terms (robotics, signal processing, ML inference, physics simulations) to prevent mode collapse
- **Heuristic Filtering**: Regex-based filters to remove host wrappers and dummy code
- **MiniMax M2.7 Annotation**: Semantic annotation with exponential backoff and JSON schema enforcement
- **Neon PostgreSQL Storage**: Efficient deduplication via SHA-256 hashing

## Technical Stack

- **Language:** Python 3.11+
- **Dependency Management:** Poetry
- **Orchestration:** GitHub Actions (Cron-triggered)
- **Database:** Neon PostgreSQL
- **External APIs:** GitHub REST API, MiniMax M2.7 API

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd cuda-ingest-pipeline

# Install dependencies with Poetry
poetry install
```

## Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Required environment variables:

- `NEON_URI` - Neon PostgreSQL connection URI
- `GITHUB_TOKEN` - GitHub personal access token
- `MINIMAX_API_KEY` - MiniMax API key

## Usage

### Local Execution

```bash
# Run the pipeline
poetry run cuda-ingest

# Or with custom kernel limit
poetry run cuda-ingest --max-kernels 20
```

### GitHub Actions (Scheduled)

The pipeline runs automatically via GitHub Actions on a cron schedule (daily at midnight by default). See `.github/workflows/ingest_cron.yml`.

## Pipeline Flow

1. **Initialize**: Load configuration, initialize DB schema
2. **Search**: Query GitHub for CUDA files using domain-balancing queries
3. **Fetch**: Download file content from repositories
4. **Deduplicate**: Check SHA-256 hashes against existing records
5. **Filter**: Apply heuristic filters (device keywords, kernel patterns, length checks)
6. **Annotate**: Send to MiniMax M2.7 for semantic annotation
7. **Store**: Batch insert annotated kernels into Neon PostgreSQL

## Database Schema

```sql
kernels (
    id SERIAL PRIMARY KEY,
    repo_name VARCHAR(255),
    file_path TEXT,
    commit_hash VARCHAR(40),
    raw_code TEXT,
    code_hash VARCHAR(64) UNIQUE,  -- SHA-256 for deduplication
    domain_tag VARCHAR(100),
    algorithmic_intent TEXT,
    memory_pattern TEXT,
    hardware_utilization TEXT,
    ingested_at TIMESTAMP
)
```

## Project Structure

```
cuda-ingest-pipeline/
├── .github/
│   └── workflows/
│       └── ingest_cron.yml
├── src/
│   ├── core/
│   │   ├── config.py
│   │   └── logger.py
│   ├── scraper/
│   │   ├── github_client.py
│   │   └── query_builder.py
│   ├── processor/
│   │   ├── filter.py
│   │   └── annotator.py
│   ├── db/
│   │   ├── client.py
│   │   └── schema.sql
│   └── main.py
├── pyproject.toml
├── .env.example
└── README.md
```

## License

MIT
