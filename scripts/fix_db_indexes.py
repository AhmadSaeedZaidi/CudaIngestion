"""Script to fix gaps in the kernels table ID sequence."""
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
    
    logger.info("Connecting to database to fix kernel IDs...")
    
    with db_client.engine.connect() as conn:
        # Check current count
        count = conn.execute(text("SELECT COUNT(*) FROM kernels")).scalar()
        logger.info(f"Found {count} entries in kernels table.")
        
        # SQL to renumber IDs and remove gaps
        renumber_sql = text("""
            WITH renumbered AS (
                SELECT code_hash, ROW_NUMBER() OVER (ORDER BY ingested_at, id) as new_id
                FROM kernels
            )
            UPDATE kernels
            SET id = renumbered.new_id
            FROM renumbered
            WHERE kernels.code_hash = renumbered.code_hash;
        """)
        
        logger.info("Executing renumbering SQL...")
        result = conn.execute(renumber_sql)
        logger.info(f"Updated {result.rowcount} rows in kernels table.")
        
        # Update the sequence to the new MAX(id)
        seq_sql = text("SELECT setval('kernels_id_seq', COALESCE((SELECT MAX(id) FROM kernels), 1));")
        logger.info("Updating sequence...")
        conn.execute(seq_sql)
        
        conn.commit()
        logger.info("Database ID gap fix completed successfully.")
        
    db_client.close()

if __name__ == "__main__":
    main()
