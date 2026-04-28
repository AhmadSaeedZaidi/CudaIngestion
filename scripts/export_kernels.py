#!/usr/bin/env python3
"""Export kernels from database to Parquet file."""

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from src.core.config import get_config
from src.core.logger import setup_logger, get_logger

setup_logger("")
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export kernels to Parquet")
    parser.add_argument(
        "--output",
        type=str,
        default="kernels.parquet",
        help="Output file path (default: kernels.parquet)",
    )
    args = parser.parse_args()

    load_dotenv()
    config = get_config()

    logger.info("Connecting to database...")
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        db_url = config.neon_uri
    engine = create_engine(db_url, pool_pre_ping=True)
    logger.info("Connected!")

    output_path = Path(args.output)
    logger.info(f"Exporting to {output_path}...")

    import pyarrow as pa
    import pyarrow.parquet as pq

    # Stream results with chunksize=10 to prevent high memory usage during fetch
    with engine.connect().execution_options(stream_results=True, yield_per=10) as conn:
        query = text("SELECT * FROM kernels")
        writer = None
        total_rows = 0

        for chunk_idx, chunk in enumerate(pd.read_sql(query, conn, chunksize=10)):
            table = pa.Table.from_pandas(chunk)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')
            
            writer.write_table(table)
            total_rows += len(chunk)
            logger.info(f"Written chunk {chunk_idx + 1} ({len(chunk)} rows)...")
            
        if writer:
            writer.close()
    
    engine.dispose()
    
    logger.info(f"Successfully exported {total_rows} total rows to {output_path}")


if __name__ == "__main__":
    main()
