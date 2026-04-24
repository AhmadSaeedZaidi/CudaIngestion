-- CUDA Kernel Ingestion Pipeline - Database Schema
-- Neon PostgreSQL schema for storing annotated CUDA kernels

-- Main kernels table
CREATE TABLE IF NOT EXISTS kernels (
    id SERIAL PRIMARY KEY,
    repo_name VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    commit_hash VARCHAR(40) NOT NULL,
    raw_code TEXT NOT NULL,
    code_hash VARCHAR(64) UNIQUE NOT NULL,  -- SHA-256 hash for deduplication
    domain_tag VARCHAR(100),
    algorithmic_intent TEXT,
    memory_pattern TEXT,
    hardware_utilization TEXT,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for deduplication lookups
CREATE INDEX IF NOT EXISTS idx_code_hash ON kernels(code_hash);

-- Index for domain-based queries
CREATE INDEX IF NOT EXISTS idx_domain_tag ON kernels(domain_tag);

-- Index for time-based queries and ordering
CREATE INDEX IF NOT EXISTS idx_ingested_at ON kernels(ingested_at);

-- Index for repository-based queries
CREATE INDEX IF NOT EXISTS idx_repo_name ON kernels(repo_name);

-- Composite index for common query patterns
CREATE INDEX IF NOT EXISTS idx_domain_ingested ON kernels(domain_tag, ingested_at DESC);

-- Comments for documentation
COMMENT ON TABLE kernels IS 'CUDA kernels scraped from GitHub and annotated with MiniMax M2.7';
COMMENT ON COLUMN kernels.code_hash IS 'SHA-256 hash of raw_code for efficient deduplication';
COMMENT ON COLUMN kernels.domain_tag IS 'Computational domain (e.g., machine_learning, signal_processing)';
COMMENT ON COLUMN kernels.algorithmic_intent IS 'Description of the algorithms purpose (2-3 sentences)';
COMMENT ON COLUMN kernels.memory_pattern IS 'Memory access pattern (e.g., row-major, tiled, shared)';
COMMENT ON COLUMN kernels.hardware_utilization IS 'Expected hardware utilization hints';
