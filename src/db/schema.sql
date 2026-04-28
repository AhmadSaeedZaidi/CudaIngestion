-- CUDA Kernel Ingestion Schema
-- Supports multi-domain kernel collection with comprehensive annotations

CREATE TABLE IF NOT EXISTS kernels (
    id SERIAL PRIMARY KEY,
    repo_name VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    commit_hash VARCHAR(40) NOT NULL,
    raw_code TEXT NOT NULL,
    code_hash VARCHAR(64) UNIQUE NOT NULL,
    
    -- Basic classification
    domain_tag VARCHAR(100),
    algorithmic_intent TEXT,
    
    -- Memory and hardware analysis
    memory_pattern TEXT,
    hardware_utilization TEXT,
    
    -- Advanced analysis fields for RL training
    mathematical_formulation TEXT,
    thread_to_data_mapping TEXT,
    bottleneck_analysis TEXT,
    edge_case_vulnerabilities TEXT,
    
    -- Metadata
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_code_hash ON kernels(code_hash);
CREATE INDEX IF NOT EXISTS idx_domain_tag ON kernels(domain_tag);
CREATE INDEX IF NOT EXISTS idx_ingested_at ON kernels(ingested_at);

-- Checkpoint/State tracking for resumable ingestion
CREATE TABLE IF NOT EXISTS ingestion_state (
    id SERIAL PRIMARY KEY,
    state_key VARCHAR(100) UNIQUE NOT NULL,
    state_value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_state_key ON ingestion_state(state_key);

-- Tracks pagination state for GitHub search queries (cursor-based pagination)
CREATE TABLE IF NOT EXISTS search_progress (
    id SERIAL PRIMARY KEY,
    query VARCHAR(500) NOT NULL,
    domain VARCHAR(100),
    current_page INTEGER DEFAULT 1,
    last_signature VARCHAR(500),
    last_result_count INTEGER DEFAULT 0,
    total_processed INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'in_progress',
    rate_limit_reset TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(query)
);

CREATE INDEX IF NOT EXISTS idx_search_progress_status ON search_progress(status);
CREATE INDEX IF NOT EXISTS idx_search_progress_domain ON search_progress(domain);

-- Discovered Repos table for two-phase ingestion
CREATE TABLE IF NOT EXISTS discovered_repos (
    id SERIAL PRIMARY KEY,
    repo_name VARCHAR(255) UNIQUE NOT NULL,
    domain_tag VARCHAR(100),
    stargazers_count INTEGER DEFAULT 0,
    last_commit_hash VARCHAR(40),
    processed_page INTEGER DEFAULT 1,
    available_kernels INTEGER DEFAULT 0,
    explored_kernels INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'pending',
    filter_version VARCHAR(10) DEFAULT 'v1',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_discovered_repos_status ON discovered_repos(status);
CREATE INDEX IF NOT EXISTS idx_discovered_repos_filter_version ON discovered_repos(filter_version);
