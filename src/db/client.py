"""Neon PostgreSQL client with connection pooling, batching, and deduplication."""

import hashlib
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional
from dataclasses import dataclass

from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from sqlalchemy.engine import Engine

try:
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class KernelRecord:
    """A CUDA kernel record for database operations."""
    repo_name: str
    file_path: str
    commit_hash: str
    raw_code: str
    domain_tag: Optional[str] = None
    algorithmic_intent: Optional[str] = None
    memory_pattern: Optional[str] = None
    hardware_utilization: Optional[str] = None


class DatabaseClient:
    """
    Neon PostgreSQL database client with connection pooling.
    Handles deduplication via code_hash to avoid redundant API costs.
    Uses batch inserts for high-throughput operations.
    """

    # Batch size for bulk inserts
    BATCH_SIZE = 100

    def __init__(self, connection_uri: str):
        """
        Initialize database client.

        Args:
            connection_uri: Neon PostgreSQL connection URI
        """
        self.engine: Engine = create_engine(
            connection_uri,
            poolclass=NullPool,  # Neon serverless works better with NullPool
            connect_args={
                "connect_timeout": 30,
            },
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
                    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            # Create indexes for common queries
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_code_hash ON kernels(code_hash)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_domain_tag ON kernels(domain_tag)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_ingested_at ON kernels(ingested_at)"))
            
            # Create ingestion state table for checkpointing
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS ingestion_state (
                    id SERIAL PRIMARY KEY,
                    state_key VARCHAR(100) UNIQUE NOT NULL,
                    state_value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_state_key ON ingestion_state(state_key)"))
            
            conn.commit()
        logger.info("Database schema initialized")

    def compute_code_hash(self, code: str) -> str:
        """
        Compute SHA-256 hash of code for deduplication.

        Args:
            code: Raw CUDA code

        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(code.encode("utf-8")).hexdigest()

    def check_duplicate(self, code_hash: str) -> bool:
        """
        Check if a kernel with this hash already exists.

        Args:
            code_hash: SHA-256 hash of the code

        Returns:
            True if duplicate exists
        """
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT 1 FROM kernels WHERE code_hash = :hash LIMIT 1"),
                {"hash": code_hash}
            )
            return result.fetchone() is not None

    def get_existing_hashes(self, code_hashes: List[str]) -> set[str]:
        """
        Check multiple hashes at once for efficiency.

        Args:
            code_hashes: List of code hashes to check

        Returns:
            Set of hashes that already exist
        """
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
        """
        Insert a kernel record into the database.

        Args:
            record: KernelRecord to insert

        Returns:
            True if inserted successfully, False if duplicate
        """
        code_hash = self.compute_code_hash(record.raw_code)

        # Check for duplicate before inserting
        if self.check_duplicate(code_hash):
            logger.debug(f"Duplicate kernel detected: {record.repo_name}/{record.file_path}")
            return False

        with self.engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO kernels 
                    (repo_name, file_path, commit_hash, raw_code, code_hash, 
                     domain_tag, algorithmic_intent, memory_pattern, hardware_utilization)
                    VALUES 
                    (:repo_name, :file_path, :commit_hash, :raw_code, :code_hash,
                     :domain_tag, :algorithmic_intent, :memory_pattern, :hardware_utilization)
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
                }
            )
            conn.commit()
            logger.info(f"Inserted kernel: {record.repo_name}/{record.file_path}")
            return True

    def _bulk_insert_psycopg2(self, records: List[KernelRecord], code_hashes: List[str]) -> int:
        """
        Use psycopg2.extras.execute_values for high-throughput batch inserts.
        This is the preferred method for Neon PostgreSQL.
        """
        if not HAS_PSYCOPG2:
            raise ImportError("psycopg2 is required for bulk inserts")
        
        # Get raw connection for execute_values
        raw_conn = self.engine.raw_connection()
        try:
            # Prepare data tuples
            data = [
                (
                    r.repo_name, r.file_path, r.commit_hash, r.raw_code, code_hashes[i],
                    r.domain_tag, r.algorithmic_intent, r.memory_pattern, r.hardware_utilization
                )
                for i, r in enumerate(records)
            ]
            
            psycopg2.extras.execute_values(
                raw_conn,
                """
                INSERT INTO kernels 
                (repo_name, file_path, commit_hash, raw_code, code_hash,
                 domain_tag, algorithmic_intent, memory_pattern, hardware_utilization)
                VALUES %s
                ON CONFLICT (code_hash) DO NOTHING
                """,
                data,
                page_size=100,
            )
            raw_conn.commit()
            return len(records)
        finally:
            raw_conn.close()

    def _bulk_insert_sqlalchemy(self, records: List[KernelRecord], code_hashes: List[str]) -> int:
        """
        Fallback: Use SQLAlchemy Core for batch inserts.
        Less efficient than psycopg2.extras but works without psycopg2.
        """
        from sqlalchemy import insert
        
        values = [
            {
                "repo_name": r.repo_name,
                "file_path": r.file_path,
                "commit_hash": r.commit_hash,
                "raw_code": r.raw_code,
                "code_hash": code_hashes[i],
                "domain_tag": r.domain_tag,
                "algorithmic_intent": r.algorithmic_intent,
                "memory_pattern": r.memory_pattern,
                "hardware_utilization": r.hardware_utilization,
            }
            for i, r in enumerate(records)
        ]
        
        with self.engine.connect() as conn:
            # Use execute_values via text for ON CONFLICT support
            for i in range(0, len(values), self.BATCH_SIZE):
                batch = values[i:i + self.BATCH_SIZE]
                placeholders = ", ".join([
                    "(%(repo_name)s, %(file_path)s, %(commit_hash)s, %(raw_code)s, %(code_hash)s, %(domain_tag)s, %(algorithmic_intent)s, %(memory_pattern)s, %(hardware_utilization)s)"
                ] * len(batch))
                
                params = {}
                for j, v in enumerate(batch):
                    for k, val in v.items():
                        params[f"{k}_{j}"] = val
                
                query = text(f"""
                    INSERT INTO kernels 
                    (repo_name, file_path, commit_hash, raw_code, code_hash,
                     domain_tag, algorithmic_intent, memory_pattern, hardware_utilization)
                    VALUES {placeholders}
                    ON CONFLICT (code_hash) DO NOTHING
                """)
                conn.execute(query, params)
            conn.commit()
        
        return len(records)

    def insert_batch(self, records: List[KernelRecord]) -> int:
        """
        Insert multiple kernel records efficiently using batch inserts.

        Args:
            records: List of KernelRecord to insert

        Returns:
            Number of records actually inserted (excluding duplicates)
        """
        if not records:
            return 0

        # Compute hashes and check for existing ones
        records_with_hashes = []
        for record in records:
            code_hash = self.compute_code_hash(record.raw_code)
            records_with_hashes.append((code_hash, record))

        # Batch check for duplicates
        all_hashes = [r[0] for r in records_with_hashes]
        existing = self.get_existing_hashes(all_hashes)

        # Filter out duplicates
        new_records = []
        new_hashes = []
        for code_hash, record in records_with_hashes:
            if code_hash in existing:
                logger.debug(f"Skipping duplicate: {record.repo_name}/{record.file_path}")
                continue
            new_records.append(record)
            new_hashes.append(code_hash)

        if not new_records:
            logger.info("All records were duplicates")
            return 0

        # Use bulk insert method
        try:
            if HAS_PSYCOPG2:
                inserted = self._bulk_insert_psycopg2(new_records, new_hashes)
            else:
                inserted = self._bulk_insert_sqlalchemy(new_records, new_hashes)
            logger.info(f"Batch insert complete: {inserted}/{len(records)} inserted")
            return inserted
        except Exception as e:
            logger.error(f"Bulk insert failed, falling back to row-by-row: {e}")
            # Fallback to row-by-row
            inserted_count = 0
            for code_hash, record in zip(new_hashes, new_records):
                try:
                    with self.engine.connect() as conn:
                        conn.execute(
                            text("""
                                INSERT INTO kernels 
                                (repo_name, file_path, commit_hash, raw_code, code_hash,
                                 domain_tag, algorithmic_intent, memory_pattern, hardware_utilization)
                                VALUES 
                                (:repo_name, :file_path, :commit_hash, :raw_code, :code_hash,
                                 :domain_tag, :algorithmic_intent, :memory_pattern, :hardware_utilization)
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
                            }
                        )
                        conn.commit()
                        inserted_count += 1
                except Exception as e:
                    logger.error(f"Failed to insert kernel {record.repo_name}/{record.file_path}: {e}")
            return inserted_count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with kernel counts by domain and total
        """
        with self.engine.connect() as conn:
            total = conn.execute(text("SELECT COUNT(*) FROM kernels")).scalar()

            by_domain = conn.execute(text("""
                SELECT domain_tag, COUNT(*) as count 
                FROM kernels 
                WHERE domain_tag IS NOT NULL 
                GROUP BY domain_tag 
                ORDER BY count DESC
            """)).fetchall()

            return {
                "total_kernels": total,
                "by_domain": dict(by_domain),
            }

    # ============ State Management for Checkpointing ============

    def get_state(self, key: str) -> Optional[str]:
        """
        Get ingestion state value by key.

        Args:
            key: State key (e.g., 'last_page', 'last_query')

        Returns:
            State value or None if not found
        """
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT state_value FROM ingestion_state WHERE state_key = :key"),
                {"key": key}
            )
            row = result.fetchone()
            return row[0] if row else None

    def set_state(self, key: str, value: str) -> None:
        """
        Set ingestion state value.

        Args:
            key: State key
            value: State value
        """
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
        logger.debug(f"State saved: {key} = {value}")

    def delete_state(self, key: str) -> None:
        """
        Delete ingestion state by key.

        Args:
            key: State key to delete
        """
        with self.engine.connect() as conn:
            conn.execute(
                text("DELETE FROM ingestion_state WHERE state_key = :key"),
                {"key": key}
            )
            conn.commit()

    def close(self) -> None:
        """Close database connections."""
        self.engine.dispose()
        logger.info("Database connections closed")

    @contextmanager
    def connection(self) -> Generator:
        """
        Context manager for database connections.

        Yields:
            Database connection
        """
        with self.engine.connect() as conn:
            yield conn
