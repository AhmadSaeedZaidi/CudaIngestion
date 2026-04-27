"""Script to backfill discovered_repos table from existing kernels."""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy import text
from src.core.config import get_config
from src.db.client import DatabaseClient
from src.core.logger import setup_logger, get_logger

setup_logger("")
logger = get_logger(__name__)

def main():
    config = get_config()
    db_client = DatabaseClient(config.neon_uri)
    
    logger.info("Connecting to database for backfill...")
    
    with db_client.engine.connect() as conn:
        # 1. Ensure columns exist (ALTER TABLE)
        logger.info("Ensuring available_kernels and explored_kernels columns exist...")
        try:
            conn.execute(text("ALTER TABLE discovered_repos ADD COLUMN IF NOT EXISTS available_kernels INTEGER DEFAULT 0;"))
            conn.execute(text("ALTER TABLE discovered_repos ADD COLUMN IF NOT EXISTS explored_kernels INTEGER DEFAULT 0;"))
            conn.commit()
        except Exception as e:
            logger.warning(f"Column modification note: {e}")
            conn.rollback() # Rollback if error so we can continue
            
        # 2. Get distinct repos and count from kernels
        logger.info("Fetching repository stats from kernels table...")
        result = conn.execute(text("""
            SELECT repo_name, MAX(domain_tag), COUNT(*) as ingested_count
            FROM kernels
            GROUP BY repo_name
        """))
        repos = result.fetchall()
        
        logger.info(f"Found {len(repos)} unique repositories in kernels table.")
        
        # 3. Upsert into discovered_repos
        upsert_sql = text("""
            INSERT INTO discovered_repos 
            (repo_name, domain_tag, stargazers_count, processed_page, available_kernels, explored_kernels, status, created_at, updated_at)
            VALUES 
            (:repo_name, :domain_tag, 0, 1, 0, :explored_kernels, 'processing', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT (repo_name) 
            DO UPDATE SET 
                explored_kernels = GREATEST(discovered_repos.explored_kernels, EXCLUDED.explored_kernels),
                domain_tag = COALESCE(discovered_repos.domain_tag, EXCLUDED.domain_tag)
        """)
        
        processed = 0
        for row in repos:
            repo_name = row[0]
            domain_tag = row[1] or "general"
            ingested_count = row[2]
            
            conn.execute(upsert_sql, {
                "repo_name": repo_name,
                "domain_tag": domain_tag,
                "explored_kernels": ingested_count
            })
            processed += 1
            
        conn.commit()
        logger.info(f"Successfully backfilled {processed} repositories into discovered_repos.")
        
    db_client.close()

if __name__ == "__main__":
    main()
