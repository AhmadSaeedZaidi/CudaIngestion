"""CUDA Kernel Ingestion Pipeline - Main Entry Point."""

import base64
import sys
import time
from typing import Any, Dict, List, Optional

from src.core.config import get_config
from src.core.logger import setup_logger, get_logger
from src.scraper.github_client import GitHubClient
from src.scraper.query_builder import QueryBuilder
from src.processor.filter import CUDAFilter
from src.processor.annotator import MiniMaxAnnotator, AnnotationSchema
from src.db.client import DatabaseClient, KernelRecord

# Setup logging
setup_logger("cuda-ingest")
logger = get_logger(__name__)


class IngestionPipeline:
    """
    Main pipeline for CUDA kernel ingestion and annotation.
    Orchestrates scraping, filtering, annotation, and storage.
    """

    def __init__(self):
        """Initialize the pipeline with configuration."""
        self.config = get_config()
        self.github_client = GitHubClient(self.config.github_token)
        self.query_builder = QueryBuilder()
        self.cuda_filter = CUDAFilter(
            min_length=self.config.min_kernel_length,
            max_length=self.config.max_kernel_length,
        )
        self.annotator = MiniMaxAnnotator(
            api_key=self.config.minimax_api_key,
            api_base=self.config.minimax_api_base,
        )
        self.db_client = DatabaseClient(self.config.neon_uri)

        logger.info("Pipeline initialized")

    def initialize(self) -> None:
        """Initialize database schema."""
        self.db_client.init_schema()
        logger.info("Database schema ready")

    def decode_file_content(self, file_data: Dict[str, Any]) -> Optional[str]:
        """
        Decode base64-encoded file content from GitHub API.

        Args:
            file_data: File metadata from GitHub API

        Returns:
            Decoded file content or None on failure
        """
        try:
            if "content" not in file_data:
                return None

            content = file_data["content"]
            # GitHub returns base64 encoded content with line breaks
            encoded = content.replace("\n", "")
            decoded = base64.b64decode(encoded).decode("utf-8")
            return decoded

        except Exception as e:
            logger.error(f"Failed to decode file content: {e}")
            return None

    def fetch_kernel(self, search_item: Dict[str, Any]) -> Optional[tuple[str, str, str]]:
        """
        Fetch kernel code and metadata from GitHub.

        Args:
            search_item: Search result item from GitHub API

        Returns:
            Tuple of (repo_name, file_path, raw_code) or None on failure
        """
        try:
            repo = search_item.get("repository", {}).get("full_name", "")
            file_path = search_item.get("path", "")
            sha = search_item.get("sha", "")

            if not repo or not file_path:
                return None

            # Fetch file content
            file_data = self.github_client.get_file_content(repo, file_path)
            raw_code = self.decode_file_content(file_data)

            if not raw_code:
                return None

            return repo, file_path, raw_code

        except Exception as e:
            logger.warning(f"Failed to fetch kernel {search_item.get('path')}: {e}")
            return None

    def process_kernel(self, repo: str, file_path: str, raw_code: str) -> Optional[KernelRecord]:
        """
        Process a single kernel: filter and annotate.

        Args:
            repo: Repository name
            file_path: Path to the file
            raw_code: Raw CUDA code

        Returns:
            KernelRecord with annotation or None if filtered out
        """
        # Apply heuristic filters
        passed, reason = self.cuda_filter.filter(raw_code)
        if not passed:
            logger.debug(f"Filtered out {repo}/{file_path}: {reason}")
            return None

        # Get commit hash for this file
        try:
            commits = self.github_client.get_commits(repo, per_page=1)
            commit_hash = commits[0].get("sha", "unknown") if commits else "unknown"
        except Exception:
            commit_hash = "unknown"

        # Annotate with MiniMax M2.7
        annotation = self.annotator.annotate(raw_code)

        # Create kernel record
        record = KernelRecord(
            repo_name=repo,
            file_path=file_path,
            commit_hash=commit_hash,
            raw_code=raw_code,
            domain_tag=annotation.domain_tag if annotation else None,
            algorithmic_intent=annotation.algorithmic_intent if annotation else None,
            memory_pattern=annotation.memory_pattern if annotation else None,
            hardware_utilization=annotation.hardware_utilization if annotation else None,
        )

        return record

    def run_batch(self, max_kernels: int = 10) -> Dict[str, int]:
        """
        Run a batch of the ingestion pipeline.

        Args:
            max_kernels: Maximum number of kernels to process

        Returns:
            Statistics dictionary
        """
        logger.info(f"Starting batch ingestion (max: {max_kernels})")

        stats = {
            "fetched": 0,
            "duplicates": 0,
            "filtered": 0,
            "annotated": 0,
            "inserted": 0,
        }

        # Generate diverse search query
        query = self.query_builder.get_next_query()
        logger.info(f"Using search query: {query}")

        # Search for CUDA files
        search_results = self.github_client.search_cuda_files(query, max_results=max_kernels * 2)
        logger.info(f"Found {len(search_results)} search results")

        # Collect records to insert
        records_to_insert: List[KernelRecord] = []

        for item in search_results:
            if stats["fetched"] >= max_kernels:
                break

            stats["fetched"] += 1

            # Fetch kernel content
            kernel_data = self.fetch_kernel(item)
            if not kernel_data:
                continue

            repo, file_path, raw_code = kernel_data

            # Process kernel (filter + annotate)
            record = self.process_kernel(repo, file_path, raw_code)
            if not record:
                stats["filtered"] += 1
                continue

            stats["annotated"] += 1
            records_to_insert.append(record)

            # Respectful delay between API calls
            time.sleep(1)

        # Batch insert into database
        if records_to_insert:
            inserted = self.db_client.insert_batch(records_to_insert)
            stats["inserted"] = inserted
            stats["duplicates"] = len(records_to_insert) - inserted

        logger.info(f"Batch complete: {stats}")
        return stats

    def run(self, max_kernels: int = 10) -> Dict[str, Any]:
        """
        Run the full ingestion pipeline.

        Args:
            max_kernels: Maximum number of kernels to process

        Returns:
            Final statistics
        """
        try:
            self.initialize()
            batch_stats = self.run_batch(max_kernels=max_kernels)
            db_stats = self.db_client.get_stats()

            return {
                "batch": batch_stats,
                "database": db_stats,
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources."""
        self.db_client.close()
        logger.info("Pipeline cleanup complete")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="CUDA Kernel Ingestion Pipeline")
    parser.add_argument(
        "--max-kernels",
        type=int,
        default=10,
        help="Maximum number of kernels to process (default: 10)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("CUDA Kernel Ingestion Pipeline Starting")
    logger.info("=" * 60)

    pipeline = IngestionPipeline()
    results = pipeline.run(max_kernels=args.max_kernels)

    logger.info("=" * 60)
    logger.info("Pipeline Results:")
    logger.info(f"  Total kernels in DB: {results['database']['total_kernels']}")
    logger.info(f"  Batch fetched: {results['batch']['fetched']}")
    logger.info(f"  Batch filtered: {results['batch']['filtered']}")
    logger.info(f"  Batch annotated: {results['batch']['annotated']}")
    logger.info(f"  Batch inserted: {results['batch']['inserted']}")
    logger.info(f"  Duplicates skipped: {results['batch']['duplicates']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
