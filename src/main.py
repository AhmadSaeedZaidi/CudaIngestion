"""CUDA Kernel Ingestion Pipeline - Main Entry Point."""

import base64
import json
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
setup_logger("cuda-ingest")
logger = get_logger(__name__)

# Persisted 1-based page for /search/repositories (50 repos per page, max ~1000 results).
GITHUB_REPO_SEARCH_PAGE_KEY = "github_repo_search_page"
# Index into QueryBuilder.repo_discovery_queries() after exhausting paginated window.
GITHUB_REPO_QUERY_INDEX_KEY = "github_repo_query_index"


def _repo_search_max_pages(repos_per_page: int) -> int:
    """GitHub repository search returns at most ~1000 items."""
    rpp = max(1, repos_per_page)
    return max(1, min(100, (1000 + rpp - 1) // rpp))


class IngestionPipeline:
    """
    Main pipeline for CUDA kernel ingestion and annotation.
    Orchestrates scraping, filtering, annotation, and storage.
    """

    # Checkpoint keys
    CHECKPOINT_KEY = "github_search_checkpoint"

    def __init__(self, dry_run: bool = False):
        """
        Initialize the pipeline with configuration.

        Args:
            dry_run: If True, skip external API calls (MiniMax, database writes)
        """
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

    def fetch_kernel(self, search_item: dict[str, Any]) -> tuple[str, str, str] | None:
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

    def process_kernel(self, repo: str, file_path: str, raw_code: str) -> KernelRecord | None:
        """
        Process a single kernel: filter only (annotation is batched later).

        Args:
            repo: Repository name
            file_path: Path to the file
            raw_code: Raw CUDA code

        Returns:
            KernelRecord without annotation (annotation is added in batch later)
        """
        # Apply heuristic filters
        passed, reason = self.cuda_filter.filter(raw_code)
        if not passed:
            logger.debug(f"Filtered out {repo}/{file_path}: {reason}")
            return None

        # Get commit hash for this file (with delay to avoid rate limits)
        commit_hash = "unknown"
        try:
            # Add small delay before commit API call to spread requests
            time.sleep(0.5)
            commits = self.github_client.get_commits(repo, per_page=1)
            commit_hash = commits[0].get("sha", "unknown") if commits else "unknown"
        except Exception:
            pass  # Keep "unknown" if commit fetch fails

        # Create kernel record WITHOUT annotation (annotation is batched in run_batch)
        record = KernelRecord(
            repo_name=repo,
            file_path=file_path,
            commit_hash=commit_hash,
            raw_code=raw_code,
            domain_tag=None,
            algorithmic_intent=None,
            memory_pattern=None,
            hardware_utilization=None,
            mathematical_formulation=None,
            thread_to_data_mapping=None,
            bottleneck_analysis=None,
            edge_case_vulnerabilities=None,
        )

        return record

    def _get_checkpoint(self) -> dict[str, Any] | None:
        """Load checkpoint from database."""
        if self.dry_run:
            return None
        try:
            checkpoint_json = self.db_client.get_state(self.CHECKPOINT_KEY)
            if checkpoint_json:
                return json.loads(checkpoint_json)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
        return None

    def _save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Save checkpoint to database."""
        if self.dry_run:
            logger.debug(f"Dry run: would save checkpoint: {checkpoint}")
            return
        try:
            self.db_client.set_state(self.CHECKPOINT_KEY, json.dumps(checkpoint))
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _clear_checkpoint(self) -> None:
        """Clear checkpoint after successful completion."""
        if self.dry_run:
            return
        try:
            self.db_client.delete_state(self.CHECKPOINT_KEY)
        except Exception as e:
            logger.warning(f"Failed to clear checkpoint: {e}")

    def run_batch(self, max_kernels: int = 10) -> dict[str, int]:
        """
        Run a batch of the ingestion pipeline.
        Uses multi-query approach to get diverse kernels across different computational domains.

        Args:
            max_kernels: Maximum number of kernels to process

        Returns:
            Statistics dictionary
        """
        logger.info(f"Starting batch ingestion (max: {max_kernels}, dry_run={self.dry_run})")

        stats = {
            "fetched": 0,
            "duplicates": 0,
            "filtered": 0,
            "annotated": 0,
            "inserted": 0,
        }

        # Load checkpoint if exists
        checkpoint = self._get_checkpoint()
        if checkpoint:
            logger.info(f"Resuming from checkpoint: {checkpoint}")

        # Diverse /search/code queries (valid qualifiers only); combined with repo: in the client.
        domain_queries = self.query_builder.get_diverse_batch(num_queries=5)
        logger.info(f"Using {len(domain_queries)} diverse domain queries: {domain_queries}")

        repos_per_run = self.config.repos_per_run
        repo_page = 1
        repo_queries = self.query_builder.repo_discovery_queries()
        query_index = 0
        if not self.dry_run:
            if self.config.reset_github_repo_discovery:
                self.db_client.set_state(GITHUB_REPO_SEARCH_PAGE_KEY, "1")
                self.db_client.set_state(GITHUB_REPO_QUERY_INDEX_KEY, "0")
                logger.info("RESET_GITHUB_REPO_DISCOVERY: repo page and query index reset")
            raw_page = self.db_client.get_state(GITHUB_REPO_SEARCH_PAGE_KEY)
            if raw_page and raw_page.strip().isdigit():
                repo_page = max(1, int(raw_page.strip()))
            raw_qi = self.db_client.get_state(GITHUB_REPO_QUERY_INDEX_KEY)
            if raw_qi and raw_qi.strip().isdigit():
                query_index = max(0, int(raw_qi.strip())) % len(repo_queries)

        repo_query = repo_queries[query_index]
        print(
            f"Fetching up to {repos_per_run} CUDA repositories "
            f"(variant {query_index + 1}/{len(repo_queries)}, page {repo_page})...",
            flush=True,
        )
        logger.info(f"Repo discovery query: {repo_query}")
        search_results: list[dict[str, Any]] = []
        repo_items: list[dict[str, Any]] = []
        next_repo_page: int | None = None
        next_query_index: int | None = None

        try:
            repo_response = self.github_client.search_repositories(
                repo_query,
                per_page=repos_per_run,
                page=repo_page,
                sort="stars",
                order="desc",
            )
            repo_items = list(repo_response.get("items") or [])
        except Exception as e:
            logger.error(f"Repository search failed: {e}")
            print(f"  -> Repository search failed: {type(e).__name__}: {e}", flush=True)
            repo_items = []

        if not repo_items:
            print("  -> No repositories returned; check token or query.", flush=True)
        else:
            print(f"  -> {len(repo_items)} repositories; searching for .cu hits per repo...", flush=True)
            search_results = self.github_client.collect_cuda_hits_from_repos(
                repo_items,
                domain_queries,
                code_hits_per_repo=30,
                max_total_candidates=max(max_kernels * 6, 200),
            )
            if not self.dry_run:
                wrapped = False
                max_pages = _repo_search_max_pages(repos_per_run)
                if len(repo_items) >= repos_per_run:
                    nxt = repo_page + 1
                    if nxt > max_pages:
                        nxt = 1
                        wrapped = True
                        logger.info(
                            f"Repo search page exceeded window ({max_pages}); wrapping to page 1"
                        )
                else:
                    nxt = 1
                next_repo_page = nxt
                if wrapped:
                    next_query_index = (query_index + 1) % len(repo_queries)
                    logger.info(
                        f"Rotating repo discovery variant -> {(next_query_index or 0) + 1}/{len(repo_queries)}"
                    )
                else:
                    next_query_index = query_index

        print(f"Total ranked code hits (deduped): {len(search_results)}", flush=True)
        logger.info(f"Collected {len(search_results)} code search hits from {len(repo_items)} repos")

        # Collect raw codes and records for batch annotation
        raw_codes_to_annotate: list[str] = []
        records_for_batch: list[tuple[KernelRecord, int]] = []  # (record, original_index)

        for item in search_results:
            if stats["fetched"] >= max_kernels:
                break

            repo = item.get('repository', {}).get('full_name', '')
            path = item.get('path', '')
            print(f"[{stats['fetched']+1}/{max_kernels}] Fetching: {repo}/{path}", flush=True)

            stats["fetched"] += 1

            # Fetch kernel content
            kernel_data = self.fetch_kernel(item)
            if not kernel_data:
                print("  -> FAILED to fetch", flush=True)
                continue

            repo, file_path, raw_code = kernel_data
            print(f"  -> Fetched {len(raw_code)} chars", flush=True)

            # Process kernel (filter only, no annotation yet)
            record = self.process_kernel(repo, file_path, raw_code)
            if not record:
                stats["filtered"] += 1
                continue

            # Collect for batch annotation
            raw_codes_to_annotate.append(raw_code)
            records_for_batch.append((record, len(raw_codes_to_annotate) - 1))

            # Respectful delay between GitHub API calls
            time.sleep(1)

        # Batch annotate all collected kernels in one API call (if any)
        if raw_codes_to_annotate:
            if self.dry_run:
                logger.info(f"Dry run: would batch annotate {len(raw_codes_to_annotate)} kernels")
                stats["annotated"] = len(raw_codes_to_annotate)
            else:
                print(f"  -> Batch annotating {len(raw_codes_to_annotate)} kernels with MiniMax...", flush=True)
                batch_annotations = self.annotator.annotate_batch(raw_codes_to_annotate)

                # Assign annotations to records
                for record, idx in records_for_batch:
                    annotation = batch_annotations[idx]
                    if annotation:
                        record.domain_tag = annotation.domain_tag
                        record.algorithmic_intent = annotation.algorithmic_intent
                        record.memory_pattern = annotation.memory_pattern
                        record.hardware_utilization = annotation.hardware_utilization
                        record.mathematical_formulation = annotation.mathematical_formulation
                        record.thread_to_data_mapping = annotation.thread_to_data_mapping
                        record.bottleneck_analysis = annotation.bottleneck_analysis
                        record.edge_case_vulnerabilities = annotation.edge_case_vulnerabilities
                        print(f"  -> Annotated [{idx+1}/{len(raw_codes_to_annotate)}]: domain={annotation.domain_tag}", flush=True)
                    else:
                        print(f"  -> Annotation [{idx+1}/{len(raw_codes_to_annotate)}] failed", flush=True)

                stats["annotated"] = len(raw_codes_to_annotate)

        # Collect records to insert
        records_to_insert = [rec for rec, _ in records_for_batch]

        # Batch insert into database (skip in dry run)
        if records_to_insert:
            if self.dry_run:
                logger.info(f"Dry run: would insert {len(records_to_insert)} records")
                stats["inserted"] = len(records_to_insert)
            else:
                inserted = self.db_client.insert_batch(records_to_insert)
                stats["inserted"] = inserted
                stats["duplicates"] = len(records_to_insert) - inserted

        # Clear checkpoint on successful completion
        if stats["inserted"] > 0 or stats["annotated"] > 0:
            self._clear_checkpoint()

        if next_repo_page is not None and not self.dry_run:
            self.db_client.set_state(GITHUB_REPO_SEARCH_PAGE_KEY, str(next_repo_page))
            if next_query_index is not None:
                self.db_client.set_state(GITHUB_REPO_QUERY_INDEX_KEY, str(next_query_index))
            logger.info(
                f"Saved repo discovery cursor page={next_repo_page} query_index={next_query_index}"
            )

        logger.info(f"Batch complete: {stats}")
        return stats

    def run(self, max_kernels: int = 10) -> dict[str, Any]:
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
