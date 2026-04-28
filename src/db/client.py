"""Neon PostgreSQL client with connection pooling, batching, and deduplication."""

import hashlib
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class KernelRecord:
    """A CUDA kernel record for database operations."""
    repo_name: str
    file_path: str
    commit_hash: str
    raw_code: str
    domain_tag: str | None = None
    algorithmic_intent: str | None = None
    memory_pattern: str | None = None
    hardware_utilization: str | None = None
    mathematical_formulation: str | None = None
    thread_to_data_mapping: str | None = None
    bottleneck_analysis: str | None = None
    edge_case_vulnerabilities: str | None = None


class DatabaseClient:
    """
    Neon PostgreSQL database client with connection pooling.
    Handles deduplication via code_hash to avoid redundant API costs.
    Uses batch inserts for high-throughput operations.
    """

    DEFAULT_INSERT_BATCH = 10

    def __init__(self, connection_uri: str, insert_batch_size: int | None = None):
        self.insert_batch_size = insert_batch_size if insert_batch_size is not None else self.DEFAULT_INSERT_BATCH
        self.engine: Engine = create_engine(
            connection_uri,
            poolclass=NullPool,
            connect_args={"connect_timeout": 30},
        )
        logger.info("Database client initialized")

    def init_schema(self) -> None:
        """Initialize database schema if not exists."""
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS kernels (
                    id SERIAL PRIMARY KEY,
                    repo_name VARCHAR(255) NOT NULL,
                    file_path TEXT NOT NULL,
                    commit_hash VARCHAR(40) NOT NULL,
                    raw_code TEXT NOT NULL,
                    code_hash VARCHAR(64) UNIQUE NOT NULL,
                    domain_tag VARCHAR(100),
                    algorithmic_intent TEXT,
                    memory_pattern TEXT,
                    hardware_utilization TEXT,
                    mathematical_formulation TEXT,
                    thread_to_data_mapping TEXT,
                    bottleneck_analysis TEXT,
                    edge_case_vulnerabilities TEXT,
                    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_code_hash ON kernels(code_hash)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_domain_tag ON kernels(domain_tag)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_ingested_at ON kernels(ingested_at)"))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS ingestion_state (
                    id SERIAL PRIMARY KEY,
                    state_key VARCHAR(100) UNIQUE NOT NULL,
                    state_value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_state_key ON ingestion_state(state_key)"))

            # Search progress table for GitHub pagination
            conn.execute(text("""
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
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_search_progress_status ON search_progress(status)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_search_progress_domain ON search_progress(domain)"))

            # Discovered Repos table for two-phase ingestion
            conn.execute(text("""
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
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_discovered_repos_status ON discovered_repos(status)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_discovered_repos_filter_version ON discovered_repos(filter_version)"))

            try:
                conn.execute(text("ALTER TABLE discovered_repos ADD COLUMN IF NOT EXISTS filter_version VARCHAR(10) DEFAULT 'v1'"))
            except Exception:
                pass

            conn.commit()
        logger.info("Database schema initialized")

    def compute_code_hash(self, code: str) -> str:
        return hashlib.sha256(code.encode("utf-8")).hexdigest()

    def check_duplicate(self, code_hash: str) -> bool:
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT 1 FROM kernels WHERE code_hash = :hash LIMIT 1"),
                {"hash": code_hash}
            )
            return result.fetchone() is not None

    def get_existing_hashes(self, code_hashes: list[str]) -> set[str]:
        if not code_hashes:
            return set()
        with self.engine.connect() as conn:
            placeholders = ", ".join([f":hash{i}" for i in range(len(code_hashes))])
            params = {f"hash{i}": h for i, h in enumerate(code_hashes)}
            result = conn.execute(
                text(f"SELECT code_hash FROM kernels WHERE code_hash IN ({placeholders})"),
                params
            )
            return {row[0] for row in result.fetchall()}

    def insert_kernel(self, record: KernelRecord) -> bool:
        code_hash = self.compute_code_hash(record.raw_code)
        if self.check_duplicate(code_hash):
            logger.debug(f"Duplicate kernel detected: {record.repo_name}/{record.file_path}")
            return False
        with self.engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO kernels
                    (repo_name, file_path, commit_hash, raw_code, code_hash,
                     domain_tag, algorithmic_intent, memory_pattern, hardware_utilization,
                     mathematical_formulation, thread_to_data_mapping, bottleneck_analysis, edge_case_vulnerabilities)
                    VALUES
                    (:repo_name, :file_path, :commit_hash, :raw_code, :code_hash,
                     :domain_tag, :algorithmic_intent, :memory_pattern, :hardware_utilization,
                     :mathematical_formulation, :thread_to_data_mapping, :bottleneck_analysis, :edge_case_vulnerabilities)
                """),
                {
                    "repo_name": record.repo_name,
                    "file_path": record.file_path,
                    "commit_hash": record.commit_hash,
                    "raw_code": record.raw_code,
                    "code_hash": code_hash,
                    "domain_tag": record.domain_tag,
                    "algorithmic_intent": record.algorithmic_intent,
                    "memory_pattern": record.memory_pattern,
                    "hardware_utilization": record.hardware_utilization,
                    "mathematical_formulation": record.mathematical_formulation,
                    "thread_to_data_mapping": record.thread_to_data_mapping,
                    "bottleneck_analysis": record.bottleneck_analysis,
                    "edge_case_vulnerabilities": record.edge_case_vulnerabilities,
                }
            )
            conn.commit()
            logger.info(f"Inserted kernel: {record.repo_name}/{record.file_path}")
            return True

    def _bulk_insert_sqlalchemy_core(self, records: list[KernelRecord], code_hashes: list[str]) -> int:
        values_list = []
        for i, record in enumerate(records):
            values_list.append({
                "repo_name": record.repo_name,
                "file_path": record.file_path,
                "commit_hash": record.commit_hash,
                "raw_code": record.raw_code,
                "code_hash": code_hashes[i],
                "domain_tag": record.domain_tag,
                "algorithmic_intent": record.algorithmic_intent,
                "memory_pattern": record.memory_pattern,
                "hardware_utilization": record.hardware_utilization,
                "mathematical_formulation": record.mathematical_formulation,
                "thread_to_data_mapping": record.thread_to_data_mapping,
                "bottleneck_analysis": record.bottleneck_analysis,
                "edge_case_vulnerabilities": record.edge_case_vulnerabilities,
            })
        with self.engine.connect() as conn:
            for i in range(0, len(values_list), self.insert_batch_size):
                batch = values_list[i:i + self.insert_batch_size]
                stmt = text("""
                    INSERT INTO kernels
                    (repo_name, file_path, commit_hash, raw_code, code_hash,
                     domain_tag, algorithmic_intent, memory_pattern, hardware_utilization,
                     mathematical_formulation, thread_to_data_mapping, bottleneck_analysis, edge_case_vulnerabilities)
                    VALUES
                    (:repo_name, :file_path, :commit_hash, :raw_code, :code_hash,
                     :domain_tag, :algorithmic_intent, :memory_pattern, :hardware_utilization,
                     :mathematical_formulation, :thread_to_data_mapping, :bottleneck_analysis, :edge_case_vulnerabilities)
                    ON CONFLICT (code_hash) DO NOTHING
                """)
                for values in batch:
                    conn.execute(stmt, values)
            conn.commit()
        return len(records)

    def insert_batch(self, records: list[KernelRecord]) -> int:
        if not records:
            return 0
        records_with_hashes = []
        for record in records:
            code_hash = self.compute_code_hash(record.raw_code)
            records_with_hashes.append((code_hash, record))
        all_hashes = [r[0] for r in records_with_hashes]
        existing = self.get_existing_hashes(all_hashes)
        new_records = []
        new_hashes = []
        for code_hash, record in records_with_hashes:
            if code_hash in existing:
                continue
            new_records.append(record)
            new_hashes.append(code_hash)
        if not new_records:
            return 0
        try:
            inserted = self._bulk_insert_sqlalchemy_core(new_records, new_hashes)
            return inserted
        except Exception as e:
            logger.error(f"Bulk insert failed: {e}")
            inserted_count = 0
            for _code_hash, record in zip(new_hashes, new_records, strict=True):
                try:
                    if self.insert_kernel(record):
                        inserted_count += 1
                except Exception:
                    pass
            return inserted_count

    def get_stats(self) -> dict[str, Any]:
        with self.engine.connect() as conn:
            total = conn.execute(text("SELECT COUNT(*) FROM kernels")).scalar()
            by_domain = conn.execute(text("""
                SELECT domain_tag, COUNT(*) as count
                FROM kernels WHERE domain_tag IS NOT NULL
                GROUP BY domain_tag ORDER BY count DESC
            """)).fetchall()
            return {"total_kernels": total, "by_domain": dict(by_domain)}

    def get_state(self, key: str) -> str | None:
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT state_value FROM ingestion_state WHERE state_key = :key"),
                {"key": key}
            )
            row = result.fetchone()
            return row[0] if row else None

    def set_state(self, key: str, value: str) -> None:
        with self.engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO ingestion_state (state_key, state_value, updated_at)
                    VALUES (:key, :value, CURRENT_TIMESTAMP)
                    ON CONFLICT (state_key)
                    DO UPDATE SET state_value = :value, updated_at = CURRENT_TIMESTAMP
                """),
                {"key": key, "value": value}
            )
            conn.commit()

    def delete_state(self, key: str) -> None:
        with self.engine.connect() as conn:
            conn.execute(text("DELETE FROM ingestion_state WHERE state_key = :key"), {"key": key})
            conn.commit()

    # ---- Search Progress Pagination Methods ----

    def get_search_progress(self, query: str) -> dict[str, Any] | None:
        """
        Get current progress for a search query.

        Returns:
            Dict with keys: query, domain, current_page, last_signature,
            last_result_count, total_processed, status, rate_limit_reset
            or None if query not found.
        """
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT query, domain, current_page, last_signature,
                           last_result_count, total_processed, status, rate_limit_reset
                    FROM search_progress WHERE query = :query
                """),
                {"query": query}
            )
            row = result.fetchone()
            if row:
                return {
                    "query": row[0],
                    "domain": row[1],
                    "current_page": row[2],
                    "last_signature": row[3],
                    "last_result_count": row[4],
                    "total_processed": row[5],
                    "status": row[6],
                    "rate_limit_reset": row[7],
                }
            return None

    def upsert_search_progress(
        self,
        query: str,
        domain: str | None = None,
        current_page: int = 1,
        last_signature: str = "",
        last_result_count: int = 0,
        total_processed: int = 0,
        status: str = "in_progress",
        rate_limit_reset: str | None = None,
    ) -> None:
        """
        Upsert search progress for a query.
        Creates new entry or updates existing one.
        """
        with self.engine.connect() as conn:
            # Use NULL for rate_limit_reset when None to avoid PostgreSQL cast issues
            reset_value = rate_limit_reset if rate_limit_reset else None
            conn.execute(
                text("""
                    INSERT INTO search_progress
                    (query, domain, current_page, last_signature, last_result_count,
                     total_processed, status, rate_limit_reset, updated_at)
                    VALUES
                    (:query, :domain, :current_page, :last_signature, :last_result_count,
                     :total_processed, :status, :rate_limit_reset, CURRENT_TIMESTAMP)
                    ON CONFLICT (query)
                    DO UPDATE SET
                        domain = COALESCE(:domain, search_progress.domain),
                        current_page = :current_page,
                        last_signature = :last_signature,
                        last_result_count = :last_result_count,
                        total_processed = :total_processed,
                        status = :status,
                        rate_limit_reset = :rate_limit_reset,
                        updated_at = CURRENT_TIMESTAMP
                """),
                {
                    "query": query,
                    "domain": domain,
                    "current_page": current_page,
                    "last_signature": last_signature,
                    "last_result_count": last_result_count,
                    "total_processed": total_processed,
                    "status": status,
                    "rate_limit_reset": reset_value,
                }
            )
            conn.commit()
        logger.debug(f"Upserted search progress: query={query}, page={current_page}, status={status}")

    def mark_search_completed(self, query: str) -> None:
        """Mark a search query as completed."""
        self.upsert_search_progress(query, current_page=0, status="completed", last_result_count=0)
        logger.info(f"Marked search completed: {query}")

    def mark_search_failed(self, query: str, rate_limit_reset: str) -> None:
        """Mark a search query as failed due to rate limit."""
        self.upsert_search_progress(
            query,
            status="failed",
            rate_limit_reset=rate_limit_reset
        )
        logger.warning(f"Marked search failed: {query}, reset at {rate_limit_reset}")

    def get_pending_searches(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get searches that are in_progress or failed but rate limit has reset.

        Returns:
            List of dicts with search progress info.
        """
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT query, domain, current_page, last_signature,
                           last_result_count, total_processed, status, rate_limit_reset
                    FROM search_progress
                    WHERE status IN ('in_progress', 'failed')
                    ORDER BY updated_at ASC
                    LIMIT :limit
                """),
                {"limit": limit}
            )
            rows = result.fetchall()
            return [
                {
                    "query": row[0],
                    "domain": row[1],
                    "current_page": row[2],
                    "last_signature": row[3],
                    "last_result_count": row[4],
                    "total_processed": row[5],
                    "status": row[6],
                    "rate_limit_reset": row[7],
                }
                for row in rows
            ]

    def cleanup_completed_searches(self, older_than_days: int = 7) -> int:
        """Remove completed queries older than specified days to keep table lean."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                    DELETE FROM search_progress
                    WHERE status = 'completed'
                    AND updated_at < CURRENT_TIMESTAMP - INTERVAL ':days days'
                    RETURNING query
                """),
                {"days": older_than_days}
            )
            deleted = len(result.fetchall())
            conn.commit()
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} completed search entries")
            return deleted

    def delete_completed_searches(self) -> int:
        """Delete ALL completed search entries to allow fresh runs to re-search queries."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                    DELETE FROM search_progress
                    WHERE status = 'completed'
                    RETURNING query
                """)
            )
            deleted = len(result.fetchall())
            conn.commit()
            logger.info(f"Deleted {deleted} completed search entries for fresh run")
            return deleted

    # ---- Two-Phase Repo Discovery Methods ----

    def upsert_discovered_repo(
        self,
        repo_name: str,
        domain_tag: str | None = None,
        stargazers_count: int = 0,
        filter_version: str = "v1",
    ) -> None:
        """Upsert a discovered repository."""
        with self.engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO discovered_repos
                    (repo_name, domain_tag, stargazers_count, filter_version, created_at, updated_at)
                    VALUES
                    (:repo_name, :domain_tag, :stargazers_count, :filter_version, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT (repo_name)
                    DO NOTHING
                """),
                {
                    "repo_name": repo_name,
                    "domain_tag": domain_tag,
                    "stargazers_count": stargazers_count,
                    "filter_version": filter_version,
                }
            )
            conn.commit()

    def get_next_repo_to_process(self) -> dict[str, Any] | None:
        """Get the next pending or processing repository, prioritized by stars."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT repo_name, domain_tag, stargazers_count, last_commit_hash, processed_page, status, available_kernels, explored_kernels
                    FROM discovered_repos
                    WHERE status IN ('pending', 'processing')
                    ORDER BY stargazers_count DESC
                    LIMIT 1
                """)
            )
            row = result.fetchone()
            if row:
                # Mark it as processing if it was pending
                if row[5] == 'pending':
                    conn.execute(
                        text("UPDATE discovered_repos SET status = 'processing', updated_at = CURRENT_TIMESTAMP WHERE repo_name = :repo_name"),
                        {"repo_name": row[0]}
                    )
                    conn.commit()
                return {
                    "repo_name": row[0],
                    "domain_tag": row[1],
                    "stargazers_count": row[2],
                    "last_commit_hash": row[3],
                    "processed_page": row[4],
                    "status": 'processing',
                    "available_kernels": row[6],
                    "explored_kernels": row[7],
                }
            return None

    def update_repo_progress(
        self,
        repo_name: str,
        processed_page: int,
        last_commit_hash: str | None = None,
        available_kernels: int | None = None,
        explored_kernels_delta: int = 0,
    ) -> None:
        """Update processing progress for a repository."""
        with self.engine.connect() as conn:
            params = {"repo_name": repo_name, "processed_page": processed_page, "explored_delta": explored_kernels_delta}
            update_hash_sql = ""
            if last_commit_hash:
                update_hash_sql = ", last_commit_hash = :last_commit_hash"
                params["last_commit_hash"] = last_commit_hash

            update_avail_sql = ""
            if available_kernels is not None:
                update_avail_sql = ", available_kernels = :available_kernels"
                params["available_kernels"] = available_kernels

            conn.execute(
                text(f"""
                    UPDATE discovered_repos
                    SET processed_page = :processed_page,
                        explored_kernels = explored_kernels + :explored_delta,
                        updated_at = CURRENT_TIMESTAMP
                        {update_hash_sql}
                        {update_avail_sql}
                    WHERE repo_name = :repo_name
                """),
                params
            )
            conn.commit()

    def mark_repo_completed(self, repo_name: str) -> None:
        """Mark a repository as completed."""
        with self.engine.connect() as conn:
            conn.execute(
                text("UPDATE discovered_repos SET status = 'completed', updated_at = CURRENT_TIMESTAMP WHERE repo_name = :repo_name"),
                {"repo_name": repo_name}
            )
            conn.commit()
            logger.info(f"Marked repository {repo_name} as completed.")

    def get_repos_for_reprocessing(self, min_stars: int = 50) -> list[dict[str, Any]]:
        """
        Get high-star repos that were processed with v1 filters and need reprocessing.

        Args:
            min_stars: Minimum star count to consider a repo for reprocessing

        Returns:
            List of dicts with repo info
        """
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT repo_name, domain_tag, stargazers_count, processed_page, available_kernels, explored_kernels
                    FROM discovered_repos
                    WHERE status = 'completed'
                      AND filter_version = 'v1'
                      AND stargazers_count >= :min_stars
                    ORDER BY stargazers_count DESC
                """),
                {"min_stars": min_stars}
            )
            rows = result.fetchall()
            return [
                {
                    "repo_name": row[0],
                    "domain_tag": row[1],
                    "stargazers_count": row[2],
                    "processed_page": row[3],
                    "available_kernels": row[4],
                    "explored_kernels": row[5],
                }
                for row in rows
            ]

    def reset_repos_for_v2_filter(self, repo_names: list[str] | None = None) -> int:
        """
        Reset completed repos with v1 filter to pending status for reprocessing with v2 filters.
        Optionally filter to specific repo names.

        Args:
            repo_names: Optional list of specific repo names to reset (for targeted reprocessing)

        Returns:
            Number of repos reset
        """
        with self.engine.connect() as conn:
            if repo_names:
                placeholders = ", ".join([f":name{i}" for i in range(len(repo_names))])
                params = {f"name{i}": name for i, name in enumerate(repo_names)}
                result = conn.execute(
                    text(f"""
                        UPDATE discovered_repos
                        SET status = 'pending',
                            processed_page = 1,
                            explored_kernels = 0,
                            filter_version = 'v2',
                            updated_at = CURRENT_TIMESTAMP
                        WHERE status = 'completed'
                          AND filter_version = 'v1'
                          AND repo_name IN ({placeholders})
                        RETURNING repo_name
                    """),
                    params
                )
            else:
                result = conn.execute(
                    text("""
                        UPDATE discovered_repos
                        SET status = 'pending',
                            processed_page = 1,
                            explored_kernels = 0,
                            filter_version = 'v2',
                            updated_at = CURRENT_TIMESTAMP
                        WHERE status = 'completed'
                          AND filter_version = 'v1'
                          AND stargazers_count >= 50
                        RETURNING repo_name
                    """)
                )
            updated = len(result.fetchall())
            conn.commit()
            logger.info(f"Reset {updated} repos for v2 filter reprocessing")
            return updated

    def close(self) -> None:
        self.engine.dispose()

    @contextmanager
    def connection(self) -> Generator:
        with self.engine.connect() as conn:
            yield conn
