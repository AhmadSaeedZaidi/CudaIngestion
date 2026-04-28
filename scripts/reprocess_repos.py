#!/usr/bin/env python3
"""
Script to reset high-star repos for reprocessing with relaxed (v2) filters.

This allows re-scraping repos that were previously filtered out due to strict
kernel filters, but have high star counts indicating they may contain valuable kernels.

Usage:
    python scripts/reprocess_repos.py [--min-stars N] [--dry-run]
"""

import argparse

from dotenv import load_dotenv

from src.core.config import get_config
from src.core.logger import get_logger, setup_logger
from src.db.client import DatabaseClient

setup_logger("")
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Reset high-star repos for reprocessing with v2 filters"
    )
    parser.add_argument(
        "--min-stars",
        type=int,
        default=50,
        help="Minimum star count to consider for reprocessing (default: 50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be reset without actually resetting",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list repos that would be reset (implies --dry-run)",
    )
    args = parser.parse_args()

    load_dotenv()
    config = get_config()

    if args.list_only:
        args.dry_run = True

    logger.info("Connecting to database...")
    db = DatabaseClient(config.neon_uri)
    try:
        with db.engine.connect() as conn:
            from sqlalchemy import text
            try:
                conn.execute(text("ALTER TABLE discovered_repos ADD COLUMN IF NOT EXISTS filter_version VARCHAR(10) DEFAULT 'v1'"))
            except Exception:
                pass

            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_discovered_repos_filter_version ON discovered_repos(filter_version)"))

            conn.commit()
    except Exception as e:
        logger.debug(f"Database setup skipped: {e}")

    repos_for_reprocess = db.get_repos_for_reprocessing(min_stars=args.min_stars)
    logger.info(f"Found {len(repos_for_reprocess)} repos eligible for reprocessing (stars >= {args.min_stars})")

    if args.list_only or args.dry_run:
        logger.info("=" * 60)
        logger.info("Repos that would be reset for v2 filter reprocessing:")
        logger.info("=" * 60)
        for repo in repos_for_reprocess:
            logger.info(
                f"  - {repo['repo_name']} "
                f"(stars: {repo['stargazers_count']}, "
                f"explored: {repo['explored_kernels']}/{repo['available_kernels']})"
            )
        logger.info("=" * 60)
        if args.list_only:
            db.close()
            return

    if args.dry_run:
        logger.info("(Dry run - no changes made)")
        db.close()
        return

    # Actually reset the repos
    reset_count = db.reset_repos_for_v2_filter()
    logger.info(f"Successfully reset {reset_count} repos for v2 filter reprocessing")

    db.close()
    logger.info("Done!")


if __name__ == "__main__":
    main()
