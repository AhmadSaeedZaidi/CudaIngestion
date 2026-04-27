"""CUDA Kernel Ingestion Pipeline - Main Entry Point."""

import base64
import time
from typing import Any

from src.core.config import get_config
from src.core.logger import get_logger, setup_logger
from src.db.client import DatabaseClient, KernelRecord
from src.processor.annotator import MiniMaxAnnotator
from src.processor.filter import CUDAFilter
from src.scraper.github_client import GitHubClient
from src.scraper.query_builder import QueryBuilder

# Setup logging
setup_logger("")
logger = get_logger(__name__)


class IngestionPipeline:
    """
    Main pipeline for CUDA kernel ingestion and annotation.
    Orchestrates scraping, filtering, annotation, and storage.
    """

    def __init__(self, dry_run: bool = False):
        """Initialize the pipeline with configuration."""
        self.config = get_config()
        self.dry_run = dry_run or self.config.dry_run
        self.github_client = GitHubClient(self.config.github_token)
        self.query_builder = QueryBuilder()
        self.cuda_filter = CUDAFilter(
            min_length=self.config.min_kernel_length,
            max_length=self.config.max_kernel_length,
        )
        self.annotator = MiniMaxAnnotator(
            api_key=self.config.minimax_api_key,
            api_base=self.config.minimax_api_base,
            batch_size=self.config.batch_size,
        )
        self.db_client = DatabaseClient(
            self.config.neon_uri,
            insert_batch_size=self.config.batch_size,
        )

        logger.info(f"Pipeline initialized (dry_run={self.dry_run})")

    def initialize(self) -> None:
        """Initialize database schema."""
        if self.dry_run:
            logger.info("Dry run: skipping database schema initialization")
            return
        self.db_client.init_schema()
        logger.info("Database schema ready")

    def decode_file_content(self, file_data: dict[str, Any]) -> str | None:
        """Decode base64-encoded file content from GitHub API."""
        try:
            if "content" not in file_data:
                return None
            content = file_data["content"]
            encoded = content.replace("\n", "")
            return base64.b64decode(encoded).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to decode file content: {e}")
            return None

    def fetch_kernel(self, search_item: dict[str, Any]) -> tuple[str, str, str] | None:
        """Fetch kernel code and metadata from GitHub."""
        try:
            repo = search_item.get("repository", {}).get("full_name", "")
            file_path = search_item.get("path", "")
            if not repo or not file_path:
                return None
            file_data = self.github_client.get_file_content(repo, file_path)
            raw_code = self.decode_file_content(file_data)
            if not raw_code:
                return None
            return repo, file_path, raw_code
        except Exception as e:
            logger.warning(f"Failed to fetch kernel {search_item.get('path')}: {e}")
            return None

    def discover_repos(self) -> None:
        """Phase 1: Discover new repositories across different domains."""
        queries = self.query_builder.repo_discovery_queries()
        logger.info(f"Discovering repos using {len(queries)} base queries...")
        for query in queries:
            try:
                response = self.github_client.search_repositories(
                    query, per_page=30, page=1, sort="stars", order="desc"
                )
                items = response.get("items", [])
                for item in items:
                    repo_name = item.get("full_name")
                    stars = item.get("stargazers_count", 0)

                    # Determine basic domain tag from repo description/topics or query
                    domain_tag = "general"
                    topics = item.get("topics", [])
                    desc = (item.get("description") or "").lower()
                    if "machine-learning" in topics or "deep-learning" in topics or "ml" in desc:
                        domain_tag = "machine_learning"
                    elif "simulation" in topics or "physics" in desc:
                        domain_tag = "simulation"
                    elif "hpc" in topics or "hpc" in desc:
                        domain_tag = "hpc"

                    if repo_name:
                        self.db_client.upsert_discovered_repo(repo_name, domain_tag, stars)
            except Exception as e:
                logger.error(f"Error discovering repos for query '{query}': {e}")

            # Respect rate limits
            time.sleep(self.github_client._min_request_delay)

    def run_batch(self, max_kernels: int = 50) -> dict[str, int]:
        """
        Run the ingestion pipeline in two phases.
        Loop until exactly max_kernels unique kernels are annotated and inserted.
        """
        logger.info(f"Starting batch ingestion (target: {max_kernels}, dry_run={self.dry_run})")
        stats = {
            "fetched": 0,
            "duplicates": 0,
            "filtered": 0,
            "annotated": 0,
            "inserted": 0,
        }

        total_inserted = 0
        batch_size = self.config.batch_size

        while total_inserted < max_kernels:
            # PHASE 1: Repo Discovery (if no pending repos)
            repo_info = self.db_client.get_next_repo_to_process()
            if not repo_info and not self.dry_run:
                logger.info("No pending repos found. Running discovery phase...")
                self.discover_repos()
                repo_info = self.db_client.get_next_repo_to_process()

            if not repo_info:
                logger.warning("No repos available to process even after discovery.")
                break

            repo_name = repo_info["repo_name"]
            processed_page = repo_info["processed_page"]
            last_commit_hash = repo_info.get("last_commit_hash")
            available_kernels = repo_info.get("available_kernels", 0)
            explored_kernels = repo_info.get("explored_kernels", 0)

            logger.info(f"Processing repo: {repo_name} (page {processed_page}, explored {explored_kernels}/{available_kernels})")

            # Fetch latest commit if we don't have it
            if not last_commit_hash and not self.dry_run:
                try:
                    commits = self.github_client.get_commits(repo_name, per_page=1)
                    last_commit_hash = commits[0].get("sha", "unknown") if commits else "unknown"
                    self.db_client.update_repo_progress(repo_name, processed_page, last_commit_hash)
                except Exception as e:
                    logger.warning(f"Failed to get commit hash for {repo_name}: {e}")
                    last_commit_hash = "unknown"

            # PHASE 2: Fetch Kernels
            query = f"extension:cu repo:{repo_name}"
            raw_codes_to_annotate = []
            records_for_batch = []

            # Fetch kernels until we have enough for a batch OR run out of results
            # We want to fill the batch, but if the target (max_kernels - total_inserted) is less than batch_size,
            # we should cap it there.
            target_for_this_batch = min(batch_size, max_kernels - total_inserted)

            while len(raw_codes_to_annotate) < target_for_this_batch:
                try:
                    results = self.github_client.search_code(query, per_page=100, page=processed_page)
                    items = results.get("items", [])

                    if available_kernels == 0 and "total_count" in results:
                        available_kernels = results["total_count"]

                    if not items or (available_kernels > 0 and explored_kernels >= available_kernels):
                        if not self.dry_run:
                            self.db_client.mark_repo_completed(repo_name)
                        logger.info(f"Finished processing repo {repo_name} (explored {explored_kernels}/{available_kernels})")
                        break

                    explored_this_page = 0
                    for item in items:
                        if len(raw_codes_to_annotate) >= target_for_this_batch:
                            break

                        explored_this_page += 1

                        file_path = item.get("path", "")
                        if not file_path:
                            continue

                        # Fetch raw code
                        kernel_data = self.fetch_kernel(item)
                        if not kernel_data:
                            continue

                        _, _, raw_code = kernel_data

                        # Pre-filter logic
                        passed, reason = self.cuda_filter.filter(raw_code)
                        if not passed:
                            stats["filtered"] += 1
                            continue

                        # Duplicate check BEFORE annotation
                        if not self.dry_run:
                            code_hash = self.db_client.compute_code_hash(raw_code)
                            if self.db_client.check_duplicate(code_hash):
                                stats["duplicates"] += 1
                                continue

                        stats["fetched"] += 1

                        record = KernelRecord(
                            repo_name=repo_name,
                            file_path=file_path,
                            commit_hash=last_commit_hash or "unknown",
                            raw_code=raw_code,
                        )
                        raw_codes_to_annotate.append(raw_code)
                        records_for_batch.append(record)

                    processed_page += 1
                    explored_kernels += explored_this_page
                    if not self.dry_run:
                        self.db_client.update_repo_progress(
                            repo_name,
                            processed_page,
                            last_commit_hash,
                            available_kernels=available_kernels,
                            explored_kernels_delta=explored_this_page
                        )

                    time.sleep(self.github_client._min_request_delay)

                except Exception as e:
                    logger.error(f"Error fetching kernels from {repo_name} page {processed_page}: {e}")
                    # If it's a rate limit or other error, mark as completed or try next repo
                    break

            # Batch Annotate and Insert
            if raw_codes_to_annotate:
                if self.dry_run:
                    stats["annotated"] += len(raw_codes_to_annotate)
                    stats["inserted"] += len(raw_codes_to_annotate)
                    total_inserted += len(raw_codes_to_annotate)
                else:
                    logger.info(f"Annotating batch of {len(raw_codes_to_annotate)} kernels...")
                    annotations = self.annotator.annotate_batch(raw_codes_to_annotate)

                    valid_records = []
                    for record, annotation in zip(records_for_batch, annotations, strict=True):
                        if annotation:
                            record.domain_tag = annotation.domain_tag
                            record.algorithmic_intent = annotation.algorithmic_intent
                            record.memory_pattern = annotation.memory_pattern
                            record.hardware_utilization = annotation.hardware_utilization
                            record.mathematical_formulation = annotation.mathematical_formulation
                            record.thread_to_data_mapping = annotation.thread_to_data_mapping
                            record.bottleneck_analysis = annotation.bottleneck_analysis
                            record.edge_case_vulnerabilities = annotation.edge_case_vulnerabilities
                            valid_records.append(record)
                            stats["annotated"] += 1
                        else:
                            logger.warning(f"Annotation failed for {record.repo_name}/{record.file_path}")

                    if valid_records:
                        inserted = self.db_client.insert_batch(valid_records)
                        stats["inserted"] += inserted
                        total_inserted += inserted
                        logger.info(f"Inserted {inserted} kernels (Total this run: {total_inserted}/{max_kernels})")

        logger.info(f"Batch complete: {stats}")
        return stats

    def run(self, max_kernels: int = 50) -> dict[str, Any]:
        """Run the full ingestion pipeline."""
        try:
            self.initialize()
            batch_stats = self.run_batch(max_kernels=max_kernels)

            if self.dry_run:
                db_stats = {"total_kernels": 0, "by_domain": {}}
            else:
                db_stats = self.db_client.get_stats()

            return {
                "batch": batch_stats,
                "database": db_stats,
                "dry_run": self.dry_run,
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

        finally:
            if not self.dry_run:
                self.cleanup()


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="CUDA Kernel Ingestion Pipeline")
    parser.add_argument(
        "--max-kernels",
        type=int,
        default=50,
        help="Maximum number of kernels to process (default: 50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run without making external API calls or database writes",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("CUDA Kernel Ingestion Pipeline Starting")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 60)

    pipeline = IngestionPipeline(dry_run=args.dry_run)
    results = pipeline.run(max_kernels=args.max_kernels)

    logger.info("=" * 60)
    logger.info("Pipeline Results:")
    logger.info(f"  Total kernels in DB: {results['database']['total_kernels']}")
    logger.info(f"  Batch fetched: {results['batch']['fetched']}")
    logger.info(f"  Batch filtered: {results['batch']['filtered']}")
    logger.info(f"  Batch annotated: {results['batch']['annotated']}")
    logger.info(f"  Batch inserted: {results['batch']['inserted']}")
    logger.info(f"  Duplicates skipped: {results['batch']['duplicates']}")
    logger.info(f"  Dry run: {results['dry_run']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
